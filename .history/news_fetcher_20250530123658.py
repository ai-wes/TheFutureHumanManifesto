import requests
import redis
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from newspaper import Article, ArticleException
from ..utils.models import RawDataItem, DataSource
from ..utils.config_loader import ConfigLoader # Adjusted import
from ..utils.logging import get_logger

class NewsFetcher:
    """Fetches news articles using News API and newspaper3k"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigLoader(config_path)
        self.redis_client = redis.Redis(**self.config.get_redis_config())
        self.logger = get_logger("news_fetcher")
        self.stream_name = self.config.get('redis.streams.raw_news')
        self.api_key = self.config.get_env('NEWS_API_KEY')
        if not self.api_key:
            self.logger.warning("NEWS_API_KEY not found in environment or config. NewsFetcher may not work.")


    def fetch_news_articles(self, days_back: int = 1) -> List[RawDataItem]:
        """Fetch news articles from News API"""
        if not self.api_key:
            self.logger.error("News API key not configured. Cannot fetch news.")
            return []

        self.logger.info(f"Fetching news articles from last {days_back} days")

        keywords = self.config.get('data_sources.news_api.keywords', [])
        sources = self.config.get('data_sources.news_api.sources', [])
        language = self.config.get('data_sources.news_api.language', 'en')
        page_size = self.config.get('data_sources.news_api.page_size', 100)


        articles = []
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        base_url = "https://newsapi.org/v2/everything"

        for keyword in keywords:
            try:
                params = {
                    'q': keyword,
                    'from': from_date,
                    'sortBy': 'publishedAt', # 'relevancy' or 'popularity' also possible
                    'pageSize': page_size,
                    'apiKey': self.api_key,
                    'language': language
                }
                if sources: # Add sources only if specified, otherwise search all
                    params['sources'] = ','.join(sources)


                response = requests.get(base_url, params=params)
                response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)

                data = response.json()

                for article_data in data.get('articles', []):
                    if not article_data.get('url'): # Skip articles without a URL
                        self.logger.debug(f"Skipping article without URL: {article_data.get('title')}")
                        continue

                    # Parse full article content
                    full_content = self._extract_full_content(article_data['url'])

                    article_id = self._generate_id(article_data['url'])

                    published_at_str = article_data.get('publishedAt')
                    published_date_dt = None
                    if published_at_str:
                        try:
                            # Handle different datetime formats, common one is ISO with Z
                            if 'Z' in published_at_str:
                                published_date_dt = datetime.fromisoformat(published_at_str.replace('Z', '+00:00')).replace(tzinfo=None)
                            else:
                                published_date_dt = datetime.fromisoformat(published_at_str).replace(tzinfo=None)
                        except ValueError:
                            self.logger.warning(f"Could not parse date {published_at_str} for article {article_data['url']}")


                    raw_item = RawDataItem(
                        id=article_id,
                        source=DataSource.NEWS,
                        title=article_data.get('title', 'No Title Provided'),
                        content=full_content or article_data.get('description') or article_data.get('content') or "", # Fallback for content
                        url=article_data['url'],
                        author=article_data.get('author'),
                        published_date=published_date_dt,
                        metadata={
                            'source_name': article_data.get('source', {}).get('name'),
                            'description': article_data.get('description'),
                            'url_to_image': article_data.get('urlToImage'),
                            'keyword': keyword
                        }
                    )
                    articles.append(raw_item)

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching news for keyword '{keyword}' from NewsAPI: {e}")
            except Exception as e: # Catch other potential errors
                self.logger.error(f"Unexpected error processing keyword '{keyword}': {e}")
                continue

        self.logger.info(f"Fetched {len(articles)} news articles")
        return articles

    def _extract_full_content(self, url: str) -> str:
        """Extract full article content using newspaper3k"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except ArticleException as e: # newspaper3k specific exception
            self.logger.debug(f"newspaper3k ArticleException for {url}: {e}")
            return ""
        except Exception as e: # Catch any other error during download/parse
            self.logger.warning(f"Could not extract full content from {url} using newspaper3k: {e}")
            return ""

    def publish_to_stream(self, articles: List[RawDataItem]) -> int:
        """Publish articles to Redis stream"""
        published_count = 0
        for article in articles:
            try:
                article_data = article.dict()
                article_data['published_date'] = article_data['published_date'].isoformat() if article_data['published_date'] else None
                article_data['extracted_date'] = article_data['extracted_date'].isoformat()
                article_data['source'] = article_data['source'].value # Enum to string

                stream_id = self.redis_client.xadd(
                    self.stream_name,
                    article_data
                )
                published_count += 1
                self.logger.debug(f"Published article {article.id} to stream {self.stream_name} with ID {stream_id}")
            except Exception as e:
                self.logger.error(f"Error publishing article {article.id} to {self.stream_name}: {e}")
                continue
        self.logger.info(f"Published {published_count} articles to Redis stream {self.stream_name}")
        return published_count

    def _generate_id(self, url: str) -> str:
        """Generate unique ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def run(self) -> int:
        """Main execution method"""
        self.logger.info("Starting News fetcher")
        articles = self.fetch_news_articles()
        if articles: # Only publish if articles were fetched
            return self.publish_to_stream(articles)
        return 0

if __name__ == "__main__":
    fetcher = NewsFetcher(config_path="../../config/config.yaml") # Adjust path
    fetcher.run()