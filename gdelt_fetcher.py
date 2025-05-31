import requests
import redis
import json
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from ..utils.models import RawDataItem, DataSource
from ..utils.config_loader import ConfigLoader # Adjusted import
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

        themes_filter = self.config.get('data_sources.gdelt.themes', []) # Renamed for clarity
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

            # Limit to last N time slots as per original logic, or a config value
            max_time_slots = self.config.get('data_sources.gdelt.max_time_slots', 10)

            for time_slot in time_slots[-max_time_slots:]:
                try:
                    events_data = self._fetch_gdelt_export(time_slot, themes_filter)
                    if events_data:
                        events.extend(events_data)
                except Exception as e: # More specific exception handling could be useful
                    self.logger.debug(f"No data or error for time slot {time_slot}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error fetching GDELT events: {e}")

        self.logger.info(f"Fetched {len(events)} GDELT events")
        return events

    def _fetch_gdelt_export(self, time_slot: str, themes_filter: List[str]) -> List[RawDataItem]:
        """Fetch GDELT export for specific time slot"""
        base_url = self.config.get('data_sources.gdelt.base_url')
        if not base_url:
            self.logger.error("GDELT base_url not configured.")
            return []

        # GDELT v2 Events format
        events_url = f"{base_url}{time_slot}.export.CSV.zip"
        self.logger.debug(f"Fetching GDELT data from {events_url}")

        try:
            # Download and parse CSV
            # Added GDELT V2 column names for clarity, can be loaded from a config or constant
            gdelt_columns = [
                'GLOBALEVENTID', 'Day', 'MonthYear', 'Year', 'FractionDate',
                'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode', 'Actor1EthnicCode',
                'Actor1Religion1Code', 'Actor1Religion2Code', 'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
                'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode', 'Actor2EthnicCode',
                'Actor2Religion1Code', 'Actor2Religion2Code', 'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
                'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode', 'QuadClass',
                'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone',
                'Actor1Geo_Type', 'Actor1Geo_Fullname', 'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code',
                'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
                'Actor2Geo_Type', 'Actor2Geo_Fullname', 'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code',
                'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID',
                'ActionGeo_Type', 'ActionGeo_Fullname', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code', 'ActionGeo_ADM2Code',
                'ActionGeo_Lat', 'ActionGeo_Long', 'ActionGeo_FeatureID',
                'DATEADDED', 'SOURCEURL', 'V2Themes' # Added V2Themes explicitly
            ]
            df = pd.read_csv(events_url, compression='zip', sep='\t', header=None, names=gdelt_columns, low_memory=False)

            events = []
            for _, row in df.iterrows():
                try:
                    # Filter by themes if specified
                    if themes_filter and not any(theme in str(row.get('V2Themes', '')) for theme in themes_filter):
                        continue

                    event_id_str = f"{row.get('GLOBALEVENTID', 'UNKNOWN_ID')}_{time_slot}"
                    event_id = self._generate_id(event_id_str)

                    raw_item = RawDataItem(
                        id=event_id,
                        source=DataSource.GDELT,
                        title=f"GDELT Event {row.get('GLOBALEVENTID', 'N/A')}",
                        content=self._create_event_description(row),
                        url=row.get('SOURCEURL', ''),
                        published_date=self._parse_gdelt_date(row.get('DATEADDED')), # DATEADDED is processing time, not event time
                        metadata={
                            'event_id': str(row.get('GLOBALEVENTID', '')),
                            'event_code': str(row.get('EventCode', '')),
                            'event_base_code': str(row.get('EventBaseCode', '')),
                            'goldstein_scale': float(row.get('GoldsteinScale', 0.0)),
                            'num_mentions': int(row.get('NumMentions', 0)),
                            'num_sources': int(row.get('NumSources', 0)),
                            'num_articles': int(row.get('NumArticles', 0)),
                            'avg_tone': float(row.get('AvgTone', 0.0)),
                            'actor1_name': str(row.get('Actor1Name', '')),
                            'actor2_name': str(row.get('Actor2Name', '')),
                            'actor1_country': str(row.get('Actor1CountryCode', '')),
                            'actor2_country': str(row.get('Actor2CountryCode', '')),
                            'action_geo_country': str(row.get('ActionGeo_CountryCode', '')),
                            'themes': str(row.get('V2Themes', '')),
                            'time_slot': time_slot,
                            'event_date': str(row.get('Day')) # Actual event date from GDELT
                        }
                    )
                    events.append(raw_item)

                except Exception as e:
                    self.logger.debug(f"Error processing GDELT row for event {row.get('GLOBALEVENTID', 'UNKNOWN')}: {e}")
                    continue
            return events

        except requests.exceptions.RequestException as e: # Handle network errors
            self.logger.warning(f"Network error fetching GDELT export {events_url}: {e}")
            return []
        except pd.errors.EmptyDataError:
            self.logger.debug(f"No data in GDELT export {events_url}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching/processing GDELT export {events_url}: {e}")
            return []


    def _create_event_description(self, row: pd.Series) -> str:
        """Create human-readable event description"""
        actor1 = row.get('Actor1Name', 'Unknown Actor1')
        actor2 = row.get('Actor2Name', '') # Default to empty if no Actor2
        event_code = row.get('EventCode', 'Unknown EventCode')

        description = f"Event involving {actor1}"
        if actor2 and actor2.strip() and actor2.lower() != 'unknown':
            description += f" and {actor2}"
        description += f" (Event Code: {event_code})"
        if row.get('ActionGeo_Fullname'):
            description += f" in {row.get('ActionGeo_Fullname')}"
        elif row.get('ActionGeo_CountryCode'):
            description += f" in country {row.get('ActionGeo_CountryCode')}"
        return description

    def _parse_gdelt_date(self, date_int: Any) -> Optional[datetime]:
        """Parse GDELT date format (YYYYMMDDHHMMSS)"""
        if pd.isna(date_int) or not date_int:
            return None # Return None if date is missing, RawDataItem will handle default
        try:
            # GDELT DATEADDED is an integer like 20150331000000
            return datetime.strptime(str(int(date_int)), '%Y%m%d%H%M%S')
        except ValueError:
            self.logger.debug(f"Could not parse GDELT date: {date_int}")
            return None

    def publish_to_stream(self, events: List[RawDataItem]) -> int:
        """Publish events to Redis stream"""
        published_count = 0
        for event in events:
            try:
                event_data = event.dict()
                event_data['published_date'] = event_data['published_date'].isoformat() if event_data['published_date'] else None
                event_data['extracted_date'] = event_data['extracted_date'].isoformat()
                event_data['source'] = event_data['source'].value # Enum to string

                stream_id = self.redis_client.xadd(
                    self.stream_name,
                    event_data
                )
                published_count += 1
                self.logger.debug(f"Published event {event.id} to stream {self.stream_name} with ID {stream_id}")
            except Exception as e:
                self.logger.error(f"Error publishing event {event.id} to {self.stream_name}: {e}")
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
        if events: # Only publish if events were fetched
            return self.publish_to_stream(events)
        return 0

if __name__ == "__main__":
    fetcher = GDELTFetcher(config_path="../../config/config.yaml") # Adjust path
    fetcher.run()