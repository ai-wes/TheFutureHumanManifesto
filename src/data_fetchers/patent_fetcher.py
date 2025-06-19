# src/data_fetchers/patent_fetcher.py
import logging
import requests
import time
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import redis
import hashlib
import os

from utils.models import RawDataItem, DataSource
from utils.config_loader import ConfigLoader
from utils.logging import get_logger

logger = get_logger("patent_fetcher_serpapi")

@dataclass
class SerpApiPatentResult:
    """Structure for patent data directly from SerpApi organic_results"""
    position: Optional[int] = None
    rank: Optional[int] = None
    title: Optional[str] = None
    snippet: Optional[str] = None # This is often the abstract or part of it
    publication_date: Optional[str] = None # Also grant_date, filing_date, priority_date
    patent_id: Optional[str] = None # e.g., "patent/US8110241B2/en"
    patent_link: Optional[str] = None
    priority_date: Optional[str] = None
    filing_date: Optional[str] = None
    grant_date: Optional[str] = None
    inventor: Optional[str] = None # SerpApi often lists one primary, or a string of them
    assignee: Optional[str] = None # SerpApi often lists one primary, or a string of them
    publication_number: Optional[str] = None # e.g., US8110241B2
    # Note: SerpApi's 'inventor' and 'assignee' fields might be single strings.
    # For multiple, you might need to parse them or use a different API if SerpApi doesn't provide structured lists.
    # For this implementation, we'll assume they are strings.

class PatentFetcher:
    def __init__(self, config_loader_instance: Optional[ConfigLoader] = None):
        if config_loader_instance:
            self.config_loader = config_loader_instance
        else:
            effective_config_path = os.getenv("GAPS_CONFIG_PATH", "config/config.yaml")
            self.config_loader = ConfigLoader(config_path=effective_config_path)

        redis_params = self.config_loader.get_redis_config()
        try:
            self.redis_client = redis.Redis(**redis_params)
            self.redis_client.ping() # Test connection
        except redis.exceptions.ConnectionError as e:
            logger.error(f"PatentFetcher: Redis connection failed: {e}. Fetcher will not be able to publish.")
            self.redis_client = None # Ensure it's None if connection fails

        self.stream_name = self.config_loader.get("redis.streams.raw_patents", "stream:raw_patents")

        # SerpApi Configuration
        serpapi_conf = self.config_loader.get('data_sources.serpapi', {})
        self.serpapi_api_key = self.config_loader.get_env('SERPAPI_API_KEY', serpapi_conf.get('api_key'))
        self.serpapi_base_url = "https://serpapi.com/search" # Fixed base URL for SerpApi
        self.serpapi_engine = serpapi_conf.get('google_patents_engine', 'google_patents')
        
        # Rate limiting for SerpApi
        self.api_last_request_time: float = 0.0
        # Default to 1 sec/req, adjust based on your SerpApi plan (e.g., paid plans allow faster rates)
        self.api_rate_limit_seconds: float = float(serpapi_conf.get('rate_limit_seconds', 1.0)) 
        
        # Default search parameters from config (data_sources.patents.search_settings)
        self.search_settings = self.config_loader.get('data_sources.patents.search_settings', {})
        self.default_queries = [
                "(neural interface) OR (brain computer interface)",
                "(CRISPR OR gene editing) AND (longevity OR aging reversal)",
                "(quantum computing) AND (artificial intelligence OR drug discovery)"
            ] # Fallback

    def _respect_rate_limit(self):
        if self.api_rate_limit_seconds <= 0: # No rate limiting if set to 0 or less
            return

        current_time = time.time()
        time_since_last = current_time - self.api_last_request_time
        if time_since_last < self.api_rate_limit_seconds:
            sleep_duration = self.api_rate_limit_seconds - time_since_last
            logger.debug(f"SerpApi Rate Limiting: sleeping for {sleep_duration:.2f}s")
            time.sleep(sleep_duration)
        self.api_last_request_time = time.time()

    def search_google_patents_via_serpapi(self, query: str, num_results: int = 10, page: int = 0) -> List[SerpApiPatentResult]:
        if not self.serpapi_api_key:
            logger.warning("SerpApi API key not configured. Skipping Google Patents search.")
            return []

        self._respect_rate_limit()
        
        params = {
            "engine": self.serpapi_engine,
            "q": query,
            "api_key": self.serpapi_api_key,
            "num": str(num_results), # Number of results per page (10-100)
            "page": str(page),     # Page number (0 is first page for SerpApi)
            # Add other parameters from self.search_settings
            "sort": self.search_settings.get("sort_by", "relevance"), # relevance, new, old
            "patents": str(self.search_settings.get("include_patents", "true")).lower(),
            "scholar": str(self.search_settings.get("include_scholar", "false")).lower(),
        }
        
        # Optional date filters
        if self.search_settings.get("date_after_value"):
            params["after"] = f"{self.search_settings.get('date_after_type', 'publication')}:{self.search_settings.get('date_after_value')}"
        if self.search_settings.get("date_before_value"):
            params["before"] = f"{self.search_settings.get('date_before_type', 'publication')}:{self.search_settings.get('date_before_value')}"
        
        # Other optional filters
        if self.search_settings.get("country_codes"): params["country"] = self.search_settings.get("country_codes")
        if self.search_settings.get("languages"): params["language"] = self.search_settings.get("languages")
        if self.search_settings.get("status_filter"): params["status"] = self.search_settings.get("status_filter")
        if self.search_settings.get("type_filter"): params["type"] = self.search_settings.get("type_filter")


        try:
            logger.debug(f"Querying SerpApi Google Patents: {self.serpapi_base_url} with query: '{query}', page: {page}")
            response = requests.get(self.serpapi_base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error(f"SerpApi returned an error for query '{query}': {data['error']}")
                return []

            organic_results = data.get("organic_results", [])
            patent_results = []
            for result in organic_results:
                if result.get("is_scholar", False): # Skip scholar results if not desired
                    if params["scholar"] == "false":
                        continue
                
                # Extracting multiple inventors/assignees if they are semi-colon separated in the string
                inventors_str = result.get("inventor", "")
                assignees_str = result.get("assignee", "")

                patent_api_result = SerpApiPatentResult(
                    position=result.get("position"),
                    rank=result.get("rank"),
                    title=result.get("title"),
                    snippet=result.get("snippet"), # Often the abstract
                    publication_date=result.get("publication_date"), # This is the primary one usually
                    patent_id=result.get("patent_id"), # e.g., "patent/US12345B2/en"
                    patent_link=result.get("patent_link"),
                    priority_date=result.get("priority_date"),
                    filing_date=result.get("filing_date"),
                    grant_date=result.get("grant_date"),
                    inventor=inventors_str, # Keep as string, KG consumer can parse
                    assignee=assignees_str, # Keep as string
                    publication_number=result.get("publication_number") # e.g., US12345B2
                )
                patent_results.append(patent_api_result)
            
            logger.info(f"Retrieved {len(patent_results)} patent results from SerpApi for query: '{query}', page: {page}")
            return patent_results

        except requests.exceptions.RequestException as e:
            logger.error(f"SerpApi request failed for query '{query}': {e}")
        except json.JSONDecodeError as e:
            logger.error(f"SerpApi JSON decode error for query '{query}': {e}. Response: {response.text[:500] if 'response' in locals() else 'N/A'}")
        except Exception as e:
            logger.error(f"Error processing SerpApi results for query '{query}': {e}")
        return []

    def _generate_id(self, unique_string: str) -> str:
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str: return None
        # SerpApi dates are usually YYYY-MM-DD
        try:
            dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return dt_obj.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning(f"Could not parse date string from SerpApi: {date_str}")
            return None # Fallback to other parsing if needed, but SerpApi is usually consistent

    def create_raw_data_item(self, serp_patent_result: SerpApiPatentResult) -> RawDataItem:
        # Use publication_number for a cleaner ID if available, else the full patent_id string
        original_patent_id = serp_patent_result.publication_number or serp_patent_result.patent_id or "UNKNOWN_ID"
        
        # For author field, try to split inventor/assignee strings if they contain multiple names
        inventors_list = [inv.strip() for inv in serp_patent_result.inventor.split(';') if inv.strip()] if serp_patent_result.inventor else []
        assignees_list = [asg.strip() for asg in serp_patent_result.assignee.split(';') if asg.strip()] if serp_patent_result.assignee else []
        
        authors_display_list = []
        if inventors_list: authors_display_list.extend(inventors_list[:2]) # Max 2 inventors for display
        if assignees_list: authors_display_list.extend(assignees_list[:1]) # Max 1 assignee for display
        author_string = "; ".join(authors_display_list) or None

        metadata = {
            "original_patent_id_str": serp_patent_result.patent_id, # Full SerpApi ID
            "publication_number": serp_patent_result.publication_number,
            "inventors_str": serp_patent_result.inventor, # Raw string
            "assignees_str": serp_patent_result.assignee, # Raw string
            "priority_date": serp_patent_result.priority_date,
            "filing_date": serp_patent_result.filing_date,
            "grant_date": serp_patent_result.grant_date,
            "source_api": "serpapi_google_patents",
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "serpapi_position": serp_patent_result.position,
            "serpapi_rank": serp_patent_result.rank
        }
        
        # Use patent_link as the primary URL
        url = serp_patent_result.patent_link or f"https://patents.google.com/patent/{original_patent_id}"

        # Use snippet as content, it's often the abstract
        content = serp_patent_result.snippet or ""

        # Determine the most relevant date for 'published_date'
        # Prefer grant_date, then publication_date (from SerpApi), then filing_date
        primary_date_str = serp_patent_result.grant_date or serp_patent_result.publication_date or serp_patent_result.filing_date

        return RawDataItem(
            id=self._generate_id(url), # Use URL for a more stable hash ID
            source=DataSource.PATENTS,
            title=serp_patent_result.title or "No Title Provided",
            content=content,
            url=url,
            author=author_string,
            published_date=self._parse_date(primary_date_str),
            metadata=metadata
        )

    def publish_to_stream(self, raw_items: List[RawDataItem]) -> int:
        if not self.redis_client:
            logger.error("Redis client not available. Cannot publish patent items.")
            return 0
        published_count = 0
        for item in raw_items:
            try:
                item_data = item.dict()
                item_data['published_date'] = item_data['published_date'].isoformat() if item_data['published_date'] else None
                item_data['extracted_date'] = item_data['extracted_date'].isoformat()
                item_data['source'] = item_data['source'].value
                item_data['metadata'] = json.dumps(item_data['metadata'])

                message_id = self.redis_client.xadd(self.stream_name, item_data)
                logger.debug(f"Published patent item {item.id} (Orig: {item.metadata.get('original_patent_id_str')}) to stream {self.stream_name} with ID {message_id}")
                published_count += 1
            except Exception as e:
                logger.error(f"Error publishing patent item {item.id} to Redis: {e}")
        logger.info(f"Published {published_count} patent items to Redis stream {self.stream_name}")
        return published_count

    def run(self):
        logger.info("Starting patent fetching process using SerpApi Google Patents...")
        if not self.serpapi_api_key:
            logger.error("SERPAPI_API_KEY is not configured. Patent fetching cannot proceed.")
            return

        queries = self.search_settings.get('future_tech_queries', self.default_queries)
        max_results_per_query = self.search_settings.get('max_results_per_query', 10)
        # SerpApi uses 'page' starting at 0, but often people think of page 1.
        # For simplicity, let's fetch just the first page of results (page=0).
        # To get more, you'd loop through pages.
        num_pages_to_fetch = self.search_settings.get('num_pages_per_query', 1)


        all_patent_api_results: List[SerpApiPatentResult] = []
        # Using publication_number or patent_id from SerpApi as the unique identifier here
        processed_unique_ids = set()

        for query in queries:
            for page_num in range(num_pages_to_fetch): # 0-indexed for SerpApi
                logger.info(f"Fetching patents for query: '{query}', page: {page_num}")
                serpapi_results = self.search_google_patents_via_serpapi(query, num_results=max_results_per_query, page=page_num)
                
                for pres in serpapi_results:
                    # Use publication_number if available and unique, otherwise full patent_id
                    unique_id_key = pres.publication_number or pres.patent_id
                    if unique_id_key and unique_id_key not in processed_unique_ids:
                        all_patent_api_results.append(pres)
                        processed_unique_ids.add(unique_id_key)
                
                if len(serpapi_results) < max_results_per_query: # No more results on subsequent pages
                    logger.debug(f"Fewer results than max_results for query '{query}' on page {page_num}, stopping pagination for this query.")
                    break 
            time.sleep(0.2) # Small delay between distinct keyword queries

        if not all_patent_api_results:
            logger.info("No patents found across all queries using SerpApi.")
            return

        logger.info(f"Fetched a total of {len(all_patent_api_results)} unique patent API results via SerpApi.")
        raw_data_items_to_publish = [self.create_raw_data_item(p_res) for p_res in all_patent_api_results]
        
        if self.redis_client:
            self.publish_to_stream(raw_data_items_to_publish)
        else:
            logger.warning("Redis client not initialized, skipping publishing.")
            # Optionally print items if no Redis
            # for item in raw_data_items_to_publish:
            #     print(json.dumps(item.dict(), indent=2, default=str))


        logger.info("Patent fetching and publishing process (SerpApi) completed.")

def main():
    fetcher = PatentFetcher()
    fetcher.run()

if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()