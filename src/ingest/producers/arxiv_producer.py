# src/ingest/producers/arxiv_producer.py
import asyncio
import aiohttp
import feedparser
import json
from confluent_kafka import Producer
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime
import uuid

KAFKA_TOPIC_RAW_DOCUMENTS = "raw_documents"

class FeedItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    link: HttpUrl
    published_date: str # Consider datetime object after parsing
    summary: str
    source: str
    topic: str # e.g., "AI", "Longevity", "BCI"
    fetch_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

async def fetch_arxiv_feed(feed_url: str, topic: str, producer_config: dict):
    producer = Producer(producer_config)
    async with aiohttp.ClientSession() as session:
        async with session.get(feed_url) as response:
            raw_rss = await response.text()
    
    feed = feedparser.parse(raw_rss)
    for entry in feed.entries[:10]: # Process a limited number for example
        item_data = {
            "title": entry.title,
            "link": entry.link,
            "published_date": entry.get("published", datetime.utcnow().isoformat()),
            "summary": entry.get("summary", ""),
            "source": "arXiv",
            "topic": topic,
        }
        try:
            feed_item = FeedItem(**item_data)
            producer.produce(
                KAFKA_TOPIC_RAW_DOCUMENTS,
                key=feed_item.id,
                value=feed_item.model_dump_json()
            )
            print(f"Produced to Kafka: {feed_item.title[:50]}...")
        except Exception as e:
            print(f"Error processing entry {entry.title}: {e}")
    producer.flush()

async def main_ingestion_loop():
    producer_conf = {'bootstrap.servers': 'localhost:9092'} # Replace with your Kafka brokers
    feeds_to_monitor = [
        ("https://export.arxiv.org/rss/cs.AI", "AI"),
        ("https://export.arxiv.org/rss/q-bio.GN", "Genomics"), # Example for genomics
        # Add more feeds for longevity, neuroscience, etc.
    ]
    
    tasks = [fetch_arxiv_feed(url, topic, producer_conf) for url, topic in feeds_to_monitor]
    # In a real system, this would run periodically or via a scheduler
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # This is a simplified run; in production, use a scheduler like APScheduler or Airflow
    asyncio.run(main_ingestion_loop())
