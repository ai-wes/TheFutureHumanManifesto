
import requests
import redis
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
from newspaper import Article
from ..utils.models import RawDataItem, DataSource
from ..utils.config import ConfigLoader
from ..utils.logging import get_logger

class NewsFetcher:
    """Fetches news articles using News API and newspaper3k"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigLoader(config_path)
        self.redis_client = redis.Redis(**self.config.get_redis_config())
        self.logger = get_logger("news_fetcher")
        self.stream_name = self.config.get('redis.streams.raw_news')
        self.api_key = self.config.get_env('NEWS_API_KEY')

    def fetch_news_articles(self, days_back: int = 1) -> List[RawDataItem]:
        """Fetch news articles from News API"""
        if not self.api_key:
            self.logger.error("News API key not configured")
            return []

        self.logger.info(f"Fetching news articles from last {days_back} days")

        keywords = self.config.get('data_sources.news_api.keywords', [])
        sources = self.config.get('data_sources.news_api.sources', [])

        articles = []
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        base_url = "https://newsapi.org/v2/everything"

        for keyword in keywords:
            try:
                params = {
                    'q': keyword,
                    'sources': ','.join(sources),
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'pageSize': 100,
                    'apiKey': self.api_key
                }

                response = requests.get(base_url, params=params)
                response.raise_for_status()

                data = response.json()

                for article_data in data.get('articles', []):
                    # Parse full article content
                    full_content = self._extract_full_content(article_data['url'])

                    article_id = self._generate_id(article_data['url'])

                    raw_item = RawDataItem(
                        id=article_id,
                        source=DataSource.NEWS,
                        title=article_data['title'],
                        content=full_content or article_data['description'] or "",
                        url=article_data['url'],
                        author=article_data.get('author'),
                        published_date=datetime.fromisoformat(
                            article_data['publishedAt'].replace('Z', '+00:00')
                        ).replace(tzinfo=None) if article_data['publishedAt'] else None,
                        metadata={
                            'source_name': article_data['source']['name'],
                            'description': article_data['description'],
                            'url_to_image': article_data.get('urlToImage'),
                            'keyword': keyword
                        }
                    )

                    articles.append(raw_item)

            except Exception as e:
                self.logger.error(f"Error fetching news for keyword '{keyword}': {e}")
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
        except Exception as e:
            self.logger.debug(f"Could not extract full content from {url}: {e}")
            return ""

    def publish_to_stream(self, articles: List[RawDataItem]) -> int:
        """Publish articles to Redis stream"""
        published_count = 0

        for article in articles:
            try:
                # Convert to dict for Redis
                article_data = article.dict()
                article_data['published_date'] = article_data['published_date'].isoformat() if article_data['published_date'] else None
                article_data['extracted_date'] = article_data['extracted_date'].isoformat()

                # Add to Redis stream
                stream_id = self.redis_client.xadd(
                    self.stream_name,
                    article_data
                )

                published_count += 1
                self.logger.debug(f"Published article {article.id} to stream")

            except Exception as e:
                self.logger.error(f"Error publishing article {article.id}: {e}")
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
        return self.publish_to_stream(articles)

if __name__ == "__main__":
    fetcher = NewsFetcher()
    fetcher.run()
