
import arxiv
import redis
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
from ..utils.models import RawDataItem, DataSource
from ..utils.config import ConfigLoader
from ..utils.logging import get_logger

class ArxivFetcher:
    """Fetches research papers from ArXiv"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigLoader(config_path)
        self.redis_client = redis.Redis(**self.config.get_redis_config())
        self.logger = get_logger("arxiv_fetcher")
        self.stream_name = self.config.get('redis.streams.raw_arxiv')

    def fetch_recent_papers(self, days_back: int = 1) -> List[RawDataItem]:
        """Fetch recent papers from ArXiv"""
        self.logger.info(f"Fetching ArXiv papers from last {days_back} days")

        categories = self.config.get('data_sources.arxiv.categories', [])
        max_results = self.config.get('data_sources.arxiv.max_results', 100)

        papers = []
        start_date = datetime.now() - timedelta(days=days_back)

        for category in categories:
            try:
                # Construct search query
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=max_results // len(categories),
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )

                for result in search.results():
                    # Filter by date
                    if result.published.replace(tzinfo=None) < start_date:
                        continue

                    # Create data item
                    paper_id = self._generate_id(result.entry_id)

                    raw_item = RawDataItem(
                        id=paper_id,
                        source=DataSource.ARXIV,
                        title=result.title,
                        content=result.summary,
                        url=result.entry_id,
                        author=", ".join([author.name for author in result.authors]),
                        published_date=result.published.replace(tzinfo=None),
                        metadata={
                            "categories": [cat for cat in result.categories],
                            "primary_category": result.primary_category,
                            "pdf_url": result.pdf_url,
                            "doi": result.doi,
                            "journal_ref": result.journal_ref,
                            "comment": result.comment
                        }
                    )

                    papers.append(raw_item)

            except Exception as e:
                self.logger.error(f"Error fetching from category {category}: {e}")
                continue

        self.logger.info(f"Fetched {len(papers)} papers from ArXiv")
        return papers

    def publish_to_stream(self, papers: List[RawDataItem]) -> int:
        """Publish papers to Redis stream"""
        published_count = 0

        for paper in papers:
            try:
                # Convert to dict for Redis
                paper_data = paper.dict()
                paper_data['published_date'] = paper_data['published_date'].isoformat() if paper_data['published_date'] else None
                paper_data['extracted_date'] = paper_data['extracted_date'].isoformat()

                # Add to Redis stream
                stream_id = self.redis_client.xadd(
                    self.stream_name,
                    paper_data
                )

                published_count += 1
                self.logger.debug(f"Published paper {paper.id} to stream with ID {stream_id}")

            except Exception as e:
                self.logger.error(f"Error publishing paper {paper.id}: {e}")
                continue

        self.logger.info(f"Published {published_count} papers to Redis stream {self.stream_name}")
        return published_count

    def _generate_id(self, url: str) -> str:
        """Generate unique ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def run(self) -> int:
        """Main execution method"""
        self.logger.info("Starting ArXiv fetcher")
        papers = self.fetch_recent_papers()
        return self.publish_to_stream(papers)

if __name__ == "__main__":
    fetcher = ArxivFetcher()
    fetcher.run()
