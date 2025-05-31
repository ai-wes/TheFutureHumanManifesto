
import requests
import redis
import json
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from ..utils.models import RawDataItem, DataSource
from ..utils.config import ConfigLoader
from ..utils.logging import get_logger

class GDELTFetcher:
    """Fetches events from GDELT database"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigLoader(config_path)
        self.redis_client = redis.Redis(**self.config.get_redis_config())
        self.logger = get_logger("gdelt_fetcher")
        self.stream_name = self.config.get('redis.streams.raw_gdelt', 'stream:raw_gdelt')

    def fetch_recent_events(self, hours_back: int = 6) -> List[RawDataItem]:
        """Fetch recent events from GDELT"""
        self.logger.info(f"Fetching GDELT events from last {hours_back} hours")

        themes = self.config.get('data_sources.gdelt.themes', [])
        events = []

        try:
            # Get the latest GDELT export files
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)

            # GDELT v2 uses 15-minute intervals
            time_slots = []
            current_time = start_time
            while current_time <= end_time:
                time_slots.append(current_time.strftime('%Y%m%d%H%M%S'))
                current_time += timedelta(minutes=15)

            for time_slot in time_slots[-10:]:  # Limit to last 10 time slots
                try:
                    events_data = self._fetch_gdelt_export(time_slot)
                    if events_data:
                        events.extend(events_data)
                except Exception as e:
                    self.logger.debug(f"No data for time slot {time_slot}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error fetching GDELT events: {e}")

        self.logger.info(f"Fetched {len(events)} GDELT events")
        return events

    def _fetch_gdelt_export(self, time_slot: str) -> List[RawDataItem]:
        """Fetch GDELT export for specific time slot"""
        base_url = self.config.get('data_sources.gdelt.base_url')

        # GDELT v2 Events format
        events_url = f"{base_url}{time_slot}.export.CSV.zip"

        try:
            # Download and parse CSV
            df = pd.read_csv(events_url, compression='zip', sep='\t', low_memory=False)

            events = []
            for _, row in df.iterrows():
                try:
                    # Filter by themes if specified
                    themes = self.config.get('data_sources.gdelt.themes', [])
                    if themes and not any(theme in str(row.get('V2Themes', '')) for theme in themes):
                        continue

                    event_id = self._generate_id(f"{row['GLOBALEVENTID']}_{time_slot}")

                    raw_item = RawDataItem(
                        id=event_id,
                        source=DataSource.GDELT,
                        title=f"GDELT Event {row['GLOBALEVENTID']}",
                        content=self._create_event_description(row),
                        url=row.get('SOURCEURL', ''),
                        published_date=self._parse_gdelt_date(row.get('DATEADDED')),
                        metadata={
                            'event_id': str(row['GLOBALEVENTID']),
                            'event_code': row.get('EventCode'),
                            'event_base_code': row.get('EventBaseCode'),
                            'goldstein_scale': row.get('GoldsteinScale'),
                            'num_mentions': row.get('NumMentions'),
                            'num_sources': row.get('NumSources'),
                            'num_articles': row.get('NumArticles'),
                            'avg_tone': row.get('AvgTone'),
                            'actor1_name': row.get('Actor1Name'),
                            'actor2_name': row.get('Actor2Name'),
                            'actor1_country': row.get('Actor1CountryCode'),
                            'actor2_country': row.get('Actor2CountryCode'),
                            'action_geo_country': row.get('ActionGeo_CountryCode'),
                            'themes': str(row.get('V2Themes', '')),
                            'time_slot': time_slot
                        }
                    )

                    events.append(raw_item)

                except Exception as e:
                    self.logger.debug(f"Error processing GDELT row: {e}")
                    continue

            return events

        except Exception as e:
            self.logger.debug(f"Error fetching GDELT export {time_slot}: {e}")
            return []

    def _create_event_description(self, row: pd.Series) -> str:
        """Create human-readable event description"""
        actor1 = row.get('Actor1Name', 'Unknown')
        actor2 = row.get('Actor2Name', 'Unknown') 
        event_code = row.get('EventCode', 'Unknown')

        description = f"Event involving {actor1}"
        if actor2 and actor2 != 'Unknown':
            description += f" and {actor2}"

        description += f" (Event Code: {event_code})"

        if row.get('ActionGeo_CountryCode'):
            description += f" in {row.get('ActionGeo_CountryCode')}"

        return description

    def _parse_gdelt_date(self, date_str: str) -> datetime:
        """Parse GDELT date format"""
        if not date_str:
            return datetime.now()
        try:
            return datetime.strptime(str(date_str), '%Y%m%d%H%M%S')
        except:
            return datetime.now()

    def publish_to_stream(self, events: List[RawDataItem]) -> int:
        """Publish events to Redis stream"""
        published_count = 0

        for event in events:
            try:
                # Convert to dict for Redis
                event_data = event.dict()
                event_data['published_date'] = event_data['published_date'].isoformat() if event_data['published_date'] else None
                event_data['extracted_date'] = event_data['extracted_date'].isoformat()

                # Add to Redis stream
                stream_id = self.redis_client.xadd(
                    self.stream_name,
                    event_data
                )

                published_count += 1
                self.logger.debug(f"Published event {event.id} to stream")

            except Exception as e:
                self.logger.error(f"Error publishing event {event.id}: {e}")
                continue

        self.logger.info(f"Published {published_count} events to Redis stream {self.stream_name}")
        return published_count

    def _generate_id(self, identifier: str) -> str:
        """Generate unique ID"""
        return hashlib.md5(identifier.encode()).hexdigest()

    def run(self) -> int:
        """Main execution method"""
        self.logger.info("Starting GDELT fetcher")
        events = self.fetch_recent_events()
        return self.publish_to_stream(events)

if __name__ == "__main__":
    fetcher = GDELTFetcher()
    fetcher.run()
