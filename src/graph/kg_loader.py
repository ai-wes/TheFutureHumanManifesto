# src/graph/kg_loader.py
from neo4j import GraphDatabase
from confluent_kafka import Consumer, KafkaError, TopicPartition
import json
import time
import uuid # Ensure uuid is imported if used for generating IDs, though not in this snippet directly for consumption

# Configuration Constants
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password" # IMPORTANT: Change this in a real environment
KAFKA_SERVERS = 'localhost:9092'
KAFKA_TOPIC_KG = "documents_for_kg"
KAFKA_CONSUMER_GROUP = 'neo4j_loader_group'

def get_neo4j_driver():
    # Ensure the driver is robustly created and managed.
    # For long-running apps, consider connection pooling or retry mechanisms.
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity() # Check if connection is possible
        print("Successfully connected to Neo4j.")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        # Depending on the application, you might want to retry or exit.
        raise

def add_document_to_graph(tx, doc_data: dict):
    # This Cypher query is based on the README.
    # It's good practice to make queries idempotent (MERGE helps).
    # Ensure all expected fields in doc_data are handled or have defaults.
    
    # Default missing optional fields to avoid Cypher errors
    doc_data.setdefault('id', uuid.uuid4().hex) # Generate ID if not present
    doc_data.setdefault('title', 'N/A')
    doc_data.setdefault('link', 'N/A')
    doc_data.setdefault('summary', '')
    doc_data.setdefault('published_date', '1970-01-01T00:00:00Z') # Default date
    doc_data.setdefault('source', 'Unknown')
    doc_data.setdefault('fetch_timestamp', '1970-01-01T00:00:00Z') # Default date
    doc_data.setdefault('topic', 'General')
    doc_data.setdefault('entities', [])

    query = (
        "MERGE (p:ResearchPaper {id: $id}) "
        "ON CREATE SET p.title = $title, p.link = $link, p.summary = $summary, "
        "    p.publishedDate = datetime($published_date), p.source = $source, "
        "    p.fetchTimestamp = datetime($fetch_timestamp), p.topic = $topic, p.processedTimestamp = datetime() "
        "ON MATCH SET p.title = $title, p.summary = $summary, p.processedTimestamp = datetime() " # Example: update some fields on match
        "WITH p "
        "MERGE (t:Topic {name: $topic}) "
        "MERGE (p)-[:HAS_TOPIC]->(t) "
        "WITH p " # Ensure p is carried over for the next FOREACH
        "FOREACH (entity_name IN $entities | "
        "  MERGE (e:Entity {name: entity_name}) "
        "  MERGE (p)-[:MENTIONS_ENTITY]->(e)"
        ")"
    )
    tx.run(query, **doc_data)

def consume_and_load():
    driver = None
    try:
        driver = get_neo4j_driver()
    except Exception:
        # If connection fails at startup, exit or retry based on deployment strategy
        print("Exiting due to Neo4j connection failure on startup.")
        return

    consumer_conf = {
        'bootstrap.servers': KAFKA_SERVERS,
        'group.id': KAFKA_CONSUMER_GROUP,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False # Recommended for at-least-once processing
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([KAFKA_TOPIC_KG])

    print(f"Starting Kafka consumer for topic: {KAFKA_TOPIC_KG} with group: {KAFKA_CONSUMER_GROUP}")
    processed_count = 0
    error_count = 0
    
    try:
        while True:
            msg = consumer.poll(timeout=1.0) # Poll for messages
            if msg is None:
                continue # No message, continue polling
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event, not necessarily an error
                    print(f"Reached end of partition for {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
                elif msg.error():
                    print(f"Kafka error: {msg.error()}")
                    # Consider more robust error handling, e.g., backoff, retries for certain errors
                    error_count +=1
                continue

            try:
                doc_data = json.loads(msg.value().decode('utf-8'))
                # print(f"Processing document: {doc_data.get('title', 'N/A')[:50]}...") # Can be verbose
                
                with driver.session(database="neo4j") as session: # Specify database if not default
                    session.execute_write(add_document_to_graph, doc_data)
                
                consumer.commit(message=msg, asynchronous=False) # Commit offset after successful processing
                processed_count += 1
                if processed_count % 100 == 0: # Log progress periodically
                    print(f"Successfully processed and committed {processed_count} messages.")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}. Message value: {msg.value()}")
                error_count += 1
                # Decide how to handle malformed messages (e.g., dead-letter queue)
                # For now, we'll commit to skip it.
                consumer.commit(message=msg, asynchronous=False)
            except Exception as e:
                print(f"Error writing to Neo4j or other processing error: {e}")
                error_count += 1
                # For Neo4j errors, you might not want to commit offset and retry,
                # or use a dead-letter queue strategy.
                # For this example, we will not commit and let it be re-polled,
                # which could lead to repeated errors if the issue is persistent.
                # A robust solution would involve more sophisticated error handling.
                time.sleep(5) # Wait before potential retry

            # Basic rate limiting if needed, though Kafka polling timeout naturally spaces things out
            # time.sleep(0.1) 

    except KeyboardInterrupt:
        print("Stopping consumer...")
    except Exception as e:
        print(f"An unexpected error occurred in the consumer loop: {e}")
    finally:
        print(f"Shutting down. Processed: {processed_count}, Errors: {error_count}")
        consumer.close()
        if driver:
            driver.close()
        print("Consumer and Neo4j driver closed.")

if __name__ == "__main__":
    consume_and_load()
