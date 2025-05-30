# src/processing/stream_processor.py
from quixstreams import Application, State
import json
from pydantic import BaseModel # Re-use FeedItem or a processed version

# Define KAFKA_TOPIC_RAW_DOCUMENTS if it's not available through other means
# This constant should ideally be shared across modules, e.g. in a common config file
KAFKA_TOPIC_RAW_DOCUMENTS = "raw_documents"


# Placeholder for NLP processing (e.g., entity extraction, sentiment)
def process_raw_text(text_data: dict) -> dict:
    # Use spaCy, NLTK, or Hugging Face Transformers for NER, sentiment, summarization
    processed_data = text_data.copy()
    processed_data["entities"] = ["entity1", "entity2"] # Replace with actual NER
    processed_data["sentiment_score"] = 0.5 # Replace with actual sentiment
    return processed_data

def setup_quix_app():
    app = Application(
        broker_address="localhost:9092", # Your Kafka brokers
        consumer_group="gaps_document_processor",
        auto_offset_reset="earliest"
    )

    input_topic = app.topic(KAFKA_TOPIC_RAW_DOCUMENTS)
    output_topic_kg = app.topic("documents_for_kg") # To Neo4j loader
    output_topic_alerts = app.topic("critical_event_alerts") # For immediate attention

    sdf = app.dataframe(input_topic)

    # 1. Deserialize JSON
    sdf = sdf.apply(lambda value: json.loads(value))

    # 2. Basic NLP Processing (can be more sophisticated)
    sdf = sdf.apply(process_raw_text)

    # 3. Filter for specific keywords or high sentiment for alerts (example)
    def alert_filter(data_row):
        critical_keywords = ["breakthrough", "revolution", "major advance"]
        # Ensure 'summary' is present and is a string
        summary = data_row.get("summary", "")
        if not isinstance(summary, str):
            summary = "" # Default to empty string if not a string
            
        if any(keyword in summary.lower() for keyword in critical_keywords) or \
           data_row.get("sentiment_score", 0) > 0.8:
            return True
        return False

    alert_stream = sdf[sdf.apply(alert_filter)]
    # The to_topic method was called on a DataFrame, it should be called on a Stream.
    # This is a conceptual representation; actual Quix Streams API might differ slightly
    # or this might be a simplified representation from the README.
    # Assuming sdf.to_topic() is the correct way to produce for all items in sdf,
    # and alert_stream.to_topic() for the filtered items.
    # For clarity, let's assume separate production steps if using older Quix versions or if a more explicit send is needed.
    # However, modern Quix Streams allows chaining like this.

    alert_stream.to_topic(output_topic_alerts) # Produce filtered alerts
    
    # 4. Send all processed data to be loaded into Knowledge Graph
    sdf.to_topic(output_topic_kg) # Produce all processed documents
    
    return app

if __name__ == "__main__":
    quix_app = setup_quix_app()
    # In production, this would be managed by Quix Cloud or a deployment script
    print("Starting Quix Streams application (conceptual). Press Ctrl+C to exit.")
    # quix_app.run() #.run() is blocking, so for a script that might do more, consider how it's managed.
                    # For this file, if it's the main entry point for this processor, .run() is appropriate.
    
    # To make it runnable for testing without Kafka/Quix Platform, we might comment out run
    # and add a small piece of code to show it would start.
    # For the purpose of creating the codebase, the provided code is fine.
    # If running this script directly:
    try:
        quix_app.run()
    except KeyboardInterrupt:
        print("Quix Streams application stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure Kafka is running and Quix Streams is configured correctly.")
