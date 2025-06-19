import redis
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
import json
import time
from datetime import datetime
import signal
import os
from typing import Dict, List, Any, Optional

# Assuming your project structure allows these imports
from utils.config_loader import ConfigLoader
from utils.logging import get_logger # Assuming you have this
# from utils.models import DataSource # If needed for specific logic

# --- Configuration ---
CONFIG_PATH = os.getenv("GAPS_CONFIG_PATH", "config/config.yaml")
CONSUMER_NAME = f"kg_consumer_{os.getpid()}"
GROUP_NAME = "kg_processing_group"
BLOCK_MS = 5000  # How long to block on XREADGROUP before timing out
COUNT_PER_READ = 10 # How many messages to fetch per XREADGROUP call

logger = get_logger("kg_redis_consumer")
config_loader = ConfigLoader(config_path=CONFIG_PATH)

redis_config = config_loader.get_redis_config()
neo4j_config = config_loader.get_neo4j_config()

# Get stream names from config
RAW_ARXIV_STREAM = config_loader.get("redis.streams.raw_arxiv", "stream:raw_arxiv")
RAW_GDELT_STREAM = config_loader.get("redis.streams.raw_gdelt", "stream:raw_gdelt")
RAW_NEWS_STREAM = config_loader.get("redis.streams.raw_news", "stream:raw_news")
# Add other streams if fetchers for them exist (patents, markets, social)
# RAW_PATENTS_STREAM = config_loader.get("redis.streams.raw_patents", "stream:raw_patents")

STREAMS_TO_CONSUME = {
    RAW_ARXIV_STREAM: ">", # ">" means only new messages in the group
    RAW_GDELT_STREAM: ">",
    RAW_NEWS_STREAM: ">",
    # RAW_PATENTS_STREAM: ">",
}

# --- Neo4j Connection ---
neo4j_driver = None

def get_neo4j_driver():
    global neo4j_driver
    if neo4j_driver is None or neo4j_driver._closed: # Check if driver is closed
        try:
            uri = neo4j_config['uri']
            user = neo4j_config['username']
            password = neo4j_config['password']
            neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            neo4j_driver.verify_connectivity()
            logger.info("Neo4j connection established.")
        except neo4j_exceptions.ServiceUnavailable as e:
            logger.error(f"Neo4j connection failed: {e}")
            neo4j_driver = None # Reset driver on failure
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while connecting to Neo4j: {e}")
            neo4j_driver = None
            raise
    return neo4j_driver

# --- Redis Connection ---
redis_client = None

def get_redis_client():
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db'],
                decode_responses=False # Important: stream data is often binary
            )
            redis_client.ping()
            logger.info("Redis connection established.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            redis_client = None
            raise
    return redis_client

# --- Data Processing and Neo4j Loading ---
def process_raw_data_item(tx, item_data: Dict[str, Any], source_stream_name: str):
    """
    Processes a single RawDataItem dictionary and loads it into Neo4j.
    """
    # Determine the primary label based on the source stream or item_data['source']
    source_value = item_data.get('source', 'UnknownSource').upper() # e.g., ARXIV, GDELT, NEWS
    primary_label = source_value.capitalize() # Arxiv, Gdelt, News

    # Common properties
    props_to_set = {
        "title": item_data.get("title"),
        "content_summary": item_data.get("content", "")[:500], # Store a summary
        "url": item_data.get("url"),
        "author": item_data.get("author"),
        "source_name": item_data.get("source"), # From RawDataItem.source enum value
        "original_id": item_data.get("id"), # The MD5 hash ID from the fetcher
        "extracted_timestamp_str": item_data.get("extracted_date"), # Already ISO string
        "published_timestamp_str": item_data.get("published_date"), # Already ISO string or None
    }
    # Convert timestamps to Neo4j datetime if they exist and are valid
    if props_to_set["extracted_timestamp_str"]:
        try:
            props_to_set["extractedAt"] = datetime.fromisoformat(props_to_set["extracted_timestamp_str"])
        except (ValueError, TypeError):
            logger.warning(f"Could not parse extracted_date: {props_to_set['extracted_timestamp_str']}")
            props_to_set["extractedAt"] = None # Or handle as string
    if props_to_set["published_timestamp_str"]:
        try:
            props_to_set["publishedAt"] = datetime.fromisoformat(props_to_set["published_timestamp_str"])
        except (ValueError, TypeError):
            logger.warning(f"Could not parse published_date: {props_to_set['published_timestamp_str']}")
            props_to_set["publishedAt"] = None # Or handle as string

    # Remove string versions after conversion or if None
    props_to_set.pop("extracted_timestamp_str", None)
    props_to_set.pop("published_timestamp_str", None)

    # Store all other metadata as properties, prefixing to avoid conflicts
    metadata = item_data.get("metadata", {})
    for k, v in metadata.items():
        # Ensure value is Neo4j compatible (str, number, bool, list of these, datetime)
        if isinstance(v, (dict, list)):
            props_to_set[f"meta_{k}"] = json.dumps(v) # Serialize complex types
        elif v is not None:
            props_to_set[f"meta_{k}"] = v

    # Cypher query to MERGE the document
    # Using URL as a unique constraint for merging if available, otherwise original_id
    unique_prop = "url" if props_to_set.get("url") else "original_id"

    query = f"""
    MERGE (doc:{primary_label} {{ {unique_prop}: $unique_value }})
    ON CREATE SET doc += $props, doc.createdAt = datetime()
    ON MATCH SET doc += $props, doc.updatedAt = datetime()
    RETURN id(doc) as nodeId
    """
    params = {"unique_value": props_to_set[unique_prop], "props": props_to_set}

    try:
        result = tx.run(query, params)
        record = result.single()
        if record:
            logger.debug(f"Processed and merged item {item_data.get('id')} into Neo4j as node {record['nodeId']} with label {primary_label}.")
        else:
            logger.warning(f"No result from Neo4j for item {item_data.get('id')}, query: {query}, params: {params}")

    except neo4j_exceptions.Neo4jError as e:
        logger.error(f"Neo4jError processing item {item_data.get('id')}: {e}. Query: {query}, Params: {props_to_set}")
    except Exception as e:
        logger.error(f"Unexpected error processing item {item_data.get('id')} in Neo4j: {e}")


# --- Consumer Group Setup ---
def setup_consumer_groups(r_client, streams_dict, group_name):
    for stream_name in streams_dict.keys():
        try:
            r_client.xgroup_create(name=stream_name, groupname=group_name, id="0", mkstream=True)
            logger.info(f"Consumer group '{group_name}' created for stream '{stream_name}'.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{group_name}' already exists for stream '{stream_name}'.")
            else:
                logger.error(f"Error creating consumer group for stream '{stream_name}': {e}")
                raise

# --- Main Consumer Loop ---
running = True

def signal_handler(signum, frame):
    global running
    logger.info("Signal received, shutting down consumer...")
    running = False

def consume_messages():
    global running
    r_client = get_redis_client()
    driver = get_neo4j_driver()

    if not r_client or not driver:
        logger.error("Failed to initialize Redis or Neo4j client. Exiting.")
        return

    setup_consumer_groups(r_client, STREAMS_TO_CONSUME, GROUP_NAME)

    stream_ids_map = {stream: ">" for stream in STREAMS_TO_CONSUME.keys()}

    logger.info(f"Starting consumer '{CONSUMER_NAME}' for group '{GROUP_NAME}' on streams: {list(STREAMS_TO_CONSUME.keys())}")

    while running:
        try:
            # Check connections
            if not neo4j_driver or neo4j_driver._closed:
                logger.warning("Neo4j connection lost. Attempting to reconnect...")
                time.sleep(5)
                driver = get_neo4j_driver() # Re-establish
                if not driver: continue # Skip iteration if reconnect fails

            if not r_client:
                logger.warning("Redis connection lost. Attempting to reconnect...")
                time.sleep(5)
                r_client = get_redis_client()
                if not r_client: continue

            # Read messages from multiple streams
            messages = r_client.xreadgroup(
                groupname=GROUP_NAME,
                consumername=CONSUMER_NAME,
                streams=stream_ids_map,
                count=COUNT_PER_READ,
                block=BLOCK_MS
            )

            if not messages:
                # logger.debug("No new messages. Polling again.")
                continue

            for stream_name_bytes, stream_messages in messages:
                stream_name = stream_name_bytes.decode('utf-8')
                message_ids_to_ack = []
                for message_id_bytes, message_data_bytes_dict in stream_messages:
                    message_id = message_id_bytes.decode('utf-8')
                    try:
                        # Convert message data from bytes to string, then parse JSON
                        # The fetchers publish dicts, redis-py xadd handles serialization.
                        # xreadgroup returns field names and values as bytes.
                        parsed_item_data = {
                            k.decode('utf-8'): v.decode('utf-8') if v is not None else None
                            for k, v in message_data_bytes_dict.items()
                        }
                        # Attempt to parse JSON for fields that might be JSON strings (like metadata)
                        # This depends on how fetchers structure the data.
                        # RawDataItem is simple enough that direct key access should work.

                        logger.info(f"Received message ID {message_id} from stream {stream_name}. Title: {parsed_item_data.get('title', 'N/A')[:50]}")

                        with driver.session(database=neo4j_config.get('database', 'neo4j')) as session:
                            session.execute_write(process_raw_data_item, parsed_item_data, stream_name)

                        message_ids_to_ack.append(message_id)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON for message ID {message_id} from {stream_name}: {e}. Data: {message_data_bytes_dict}")
                        # Potentially move to dead-letter queue or XACK to avoid reprocessing bad message
                    except Exception as e:
                        logger.error(f"Error processing message ID {message_id} from {stream_name}: {e}")
                        # Decide on retry logic or moving to DLQ

                if message_ids_to_ack:
                    ack_result = r_client.xack(stream_name, GROUP_NAME, *message_ids_to_ack)
                    logger.debug(f"Acknowledged {ack_result} messages for stream {stream_name}.")

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection error in main loop: {e}. Attempting to reconnect...")
            redis_client = None # Force reconnect in next iteration
            time.sleep(5)
        except neo4j_exceptions.ServiceUnavailable as e:
            logger.error(f"Neo4j ServiceUnavailable in main loop: {e}. Attempting to reconnect...")
            neo4j_driver = None # Force reconnect
            time.sleep(10)
        except KeyboardInterrupt: # Already handled by signal_handler, but good practice
            logger.info("KeyboardInterrupt in loop, stopping.")
            running = False
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main consumer loop: {e}")
            time.sleep(5) # Wait before retrying

    logger.info("Consumer loop finished.")
    if neo4j_driver:
        neo4j_driver.close()
    logger.info("KG Redis Consumer shut down.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        consume_messages()
    except Exception as e:
        logger.critical(f"Consumer crashed: {e}", exc_info=True)