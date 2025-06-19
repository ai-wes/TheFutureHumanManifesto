# GAPS (Generative Assistive Prediction System) - Operational README

## 1. Overview

This project, GAPS, is a sophisticated system designed to assist in generating future scenarios, particularly focused on transhumanism and related technological advancements. It ingests data from various sources, processes it into a knowledge graph, uses probabilistic forecasting models to predict trends and event likelihoods, and leverages Large Language Models (LLMs) to evolve scenarios and synthesize narratives.

The system comprises several key parts:

- **Data Fetchers:** Collect raw data from sources like ArXiv, GDELT, News APIs.
- **Redis Streams:** Act as a message broker for ingested data.
- **Knowledge Graph Consumer:** Processes data from Redis and populates a Neo4j graph database.
- **Forecasting Engines:**
  - A TFP-based forecaster (`src/forecasting/hybrid_forecaster.py`) for time-series predictions, potentially used by the Pipeline Orchestrator.
  - A PyMC-based Bayesian forecaster (`src/forecasting/pymc_forecaster.py`) for time-series predictions.
  - A GAPS-E specific forecaster (`src/gapse_subsystem/hybrid_probabilistic_forecaster.py`) using a PyTorch NN and a Bayesian Ridge model to assess scenario probabilities.
- **Scenario Generation & Evolution (GAPS-E):**
  - `EvolutionaryScenarioGenerator` uses LLMs to create and evolve diverse `ScenarioGenome` objects.
  - `ContradictionAnalysisEngine` assesses the internal consistency of scenarios.
- **Orchestration:**
  - `PipelineOrchestrator` (`src/pipeline/orchestrator.py`) for scheduling data ingestion and forecasting tasks (TFP-based).
  - `GAPSESystem` (`src/gapse_subsystem/gapse_system.py`) manages the evolutionary scenario generation and evaluation loop.
- **Utilities:** Configuration loading, logging, data models.

**Current State:** This is a complex system under development. Many components have been drafted, but some require further implementation (e.g., actual LLM calls in all places, robust model training data, full implementation of the `PipelineOrchestrator`'s phases).

## 2. Prerequisites

### 2.1. Software & Services:

- **Python:** Version 3.9+ recommended.
- **Redis:** A running Redis instance.
- **Neo4j:** A running Neo4j instance (version 4.x or 5.x).
- **LLM Access:**
  - **OpenAI API Key:** If using OpenAI models.
  - **Local LLM (Optional):** If using local models (e.g., via Ollama for Gemma), ensure the Ollama service is running and the desired model is pulled.
- **Git:** For version control.

### 2.2. API Keys & Credentials:

These should be stored in a `.env` file in the project root or set as environment variables.

- `OPENAI_API_KEY`: For OpenAI.
- `NEWS_API_KEY`: For NewsAPI.org.
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: If different from defaults or `config.yaml`.
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`: If different from defaults or `config.yaml`.
- `GAPS_CONFIG_PATH` (Optional): To specify a non-default path for `config.yaml`.

## 3. Project Structure

The project should be organized as follows (key directories):

IGNORE_WHEN_COPYING_START
Use code with caution. Markdown
IGNORE_WHEN_COPYING_END

.
├── config/
│ └── config.yaml
├── models/ <-- For saving trained ML models, scalers, vectorizers
├── src/
│ ├── data_fetchers/
│ ├── forecasting/
│ ├── gapse_subsystem/
│ ├── pipeline/
│ ├── processing/
│ ├── scenarios/
│ └── utils/
├── .env
└── README.md (this file)

Ensure all subdirectories within `src/` contain an `__init__.py` file to be recognized as Python packages.

## 4. Configuration

### 4.1. `config/config.yaml`

This is the primary configuration file. It defines:

- Data source parameters (URLs, categories, keywords).
- Redis and Neo4j connection details and stream names.
- OpenAI (or other LLM) model preferences and API key (can be overridden by `.env`).
- Scheduling parameters for the `PipelineOrchestrator` (cron strings).
- Settings for the GAPS-E system (`gapse_settings`), including:
  - LLM model for scenario generation.
  - Evolutionary algorithm parameters.
  - Paths for saving/loading GAPS-E forecaster models, scalers, and vectorizers.
  - Training parameters for GAPS-E models.
- Settings for the TFP/PyMC forecasters.

**CRITICAL:** Review and update placeholder values in `config.yaml`, especially API keys (though `.env` is preferred for these), Neo4j/Redis connection details if not default, and paths.

### 4.2. `.env` File

Create a `.env` file in the project root for sensitive credentials and environment-specific overrides:

```env
OPENAI_API_KEY="sk-yourActualOpenAIKey"
NEWS_API_KEY="yourActualNewsApiKey"

# Optional: Override Neo4j/Redis from config.yaml if needed for this environment
# NEO4J_URI="bolt://custom_neo4j_host:7687"
# NEO4J_USERNAME="myuser"
# NEO4J_PASSWORD="mypassword"
# REDIS_HOST="custom_redis_host"

# Optional: Specify config path if not 'config/config.yaml'
# GAPS_CONFIG_PATH="path/to/your/custom_config.yaml"



IGNORE_WHEN_COPYING_START
Use code with caution.
IGNORE_WHEN_COPYING_END

The ConfigLoader (src/utils/config_loader.py) will automatically load this file.
5. Setup Steps

    Clone the Repository:


    git clone <repository_url>
    cd <repository_name>



    IGNORE_WHEN_COPYING_START

Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Create and Activate Python Virtual Environment:


python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows



IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Install Python Dependencies:
A requirements.txt file needs to be created based on all imports in the project.
Action: Manually inspect all .py files for import statements and create requirements.txt.
Key libraries include:
arxiv, redis, requests, pandas, newspaper3k, apscheduler, tensorflow, tensorflow-probability, torch, scikit-learn, PyYAML, python-dotenv, langchain, openai, pymc, arviz, joblib, confluent-kafka, feedparser, aiohttp, numpy, pymoo
Once requirements.txt is created:


pip install -r requirements.txt



IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Set Up External Services:

    Redis: Start your Redis server. Ensure it's accessible based on config.yaml or .env (default: localhost:6379).

    Neo4j: Start your Neo4j server. Ensure it's accessible (default: neo4j://localhost:7687, user neo4j, pass password). Create the database specified in config.yaml if it's not the default 'neo4j'.

    LLM:

        If using OpenAI, ensure your API key is set in .env.

        If using a local LLM (e.g., Ollama), ensure the service is running and the model specified in config.yaml (e.g., gemma:2b-instruct) is downloaded/available.

Configure the Application:

    Carefully review and edit config/config.yaml for your specific setup, data source preferences, API keys (if not using .env), and paths.

    Create/populate the .env file in the project root with your API keys and any necessary overrides.

Create Model Storage Directory:
The GAPS-E training script and forecaster save/load models from a models/ directory by default.


mkdir models



IGNORE_WHEN_COPYING_START

    Use code with caution. Bash
    IGNORE_WHEN_COPYING_END

6. Running the Application

IMPORTANT: Run all Python module commands from the project root directory. This ensures correct relative imports and path resolutions for config.yaml.
6.1. Data Ingestion Pipeline
6.1.1. Running Individual Data Fetchers (for testing/manual ingestion)

Modify the config_path in the if __name__ == "__main__": block of each fetcher script (e.g., src/data_fetchers/arxiv_fetcher.py) from ../../config/config.yaml to config/config.yaml.


python -m src.data_fetchers.arxiv_fetcher
python -m src.data_fetchers.gdelt_fetcher
python -m src.data_fetchers.news_fetcher
# Add commands for patent_fetcher, market_fetcher, social_fetcher when implemented



IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

These will publish data to the respective Redis streams defined in config.yaml.
6.1.2. Running the Knowledge Graph Redis Consumer

This script consumes data from Redis streams and loads it into Neo4j.


python -m src.processing.kg_redis_consumer



IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Keep this running in a separate terminal to continuously process incoming data. It will create consumer groups if they don't exist.
6.2. Training GAPS-E Models

The ProbabilisticNN and BayesianRidge models used by the GAPS-E HybridProbabilisticForecaster need to be trained.


python -m src.gapse_subsystem.train_probabilistic_nn



IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

    Note: This script currently uses dummy data for training. For meaningful results, replace generate_dummy_scenario_data with a function that loads your actual labeled ScenarioGenome data and corresponding target probabilities.

    This script will save the trained PyTorch model (.pth), Scikit-learn scaler (.joblib), TfidfVectorizer (.joblib), and BayesianRidge model (.joblib) to the models/ directory (or paths specified in config.yaml).

6.3. Running the GAPS-E System

This system performs evolutionary scenario generation and uses its specific forecaster.

    Ensure the LLM client in src/gapse_subsystem/evolutionary_scenario_generator.py is correctly configured and pointing to your desired LLM.

    Ensure the models trained in step 6.2 are available in the models/ directory.

    Modify the __main__ block of src/gapse_subsystem/gapse_system.py to load the main config/config.yaml instead of its inline sample_config for consistency.


python -m src.gapse_subsystem.gapse_system



IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END
6.4. Running the Pipeline Orchestrator

This orchestrator schedules data ingestion (via fetchers) and forecasting (using TFP/PyMC forecasters).

    Implementation Needed:

        The _ingestion_phase in src/pipeline/orchestrator.py needs to be fully implemented to call the run() methods of the actual data fetchers.

        The _forecasting_phase uses src/forecasting/hybrid_forecaster.py (TFP-based) and src/scenarios/evolutionary_generator.py (Pymoo-based). These components might need their own training/setup if used.

        The __main__ block of orchestrator.py uses mock Neo4j/LLM clients. Replace these with actual client initializations using ConfigLoader.

        Ensure the scheduling cron string parsing is correctly handled (as discussed previously, using CronTrigger.from_crontab).


python -m src.pipeline.orchestrator



IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

This will start a background scheduler.
6.5. Running Other Forecasters (TFP / PyMC)

You can train and test these individually if needed.

    TFP Forecaster (src/forecasting/hybrid_forecaster.py):

        Its __main__ block is for demonstration. Adapt it to use your ConfigLoader and an actual Neo4j driver.

        CRITICAL: The Cypher queries within this file must be adapted to your Neo4j graph schema to fetch appropriate time series and feature data.


    # Example (after adapting __main__)
    # python -m src.forecasting.hybrid_forecaster



    IGNORE_WHEN_COPYING_START

Use code with caution. Bash
IGNORE_WHEN_COPYING_END

PyMC Forecaster (src/forecasting/pymc_forecaster.py):

    Its __main__ block is also for demonstration.

    CRITICAL: The Cypher query in _get_time_series_data must be adapted to your Neo4j schema.


# Example (after adapting __main__)
# python -m src.forecasting.pymc_forecaster



IGNORE_WHEN_COPYING_START

    Use code with caution. Bash
    IGNORE_WHEN_COPYING_END

7. Key Components and Roles

    src/utils/config_loader.py: Loads config.yaml and .env.

    src/utils/logging.py: Provides a standardized logger.

    src/utils/models.py: Defines Pydantic data models like RawDataItem, ScenarioGenome.

    src/data_fetchers/: Modules for acquiring data (ArXiv, GDELT, News). They publish to Redis.

    src/processing/kg_redis_consumer.py: Consumes from Redis streams, processes data, and loads it into Neo4j.

    src/gapse_subsystem/: Contains the core GAPS-E components:

        evolutionary_scenario_generator.py: LLM-driven scenario evolution.

        hybrid_probabilistic_forecaster.py: PyTorch NN + Bayesian Ridge for GAPS-E scenario probability.

        train_probabilistic_nn.py: Training script for the GAPS-E forecaster models.

        contradiction_analysis_engine.py: Assesses scenario consistency.

        scenario_database.py: In-memory store for scenarios.

        gapse_system.py: Orchestrates the GAPS-E evolutionary loop.

    src/forecasting/: Contains forecasters potentially used by the PipelineOrchestrator:

        hybrid_forecaster.py: TFP-STS and GBM models, reads from Neo4j.

        pymc_forecaster.py: PyMC Bayesian time series model, reads from Neo4j.

    src/scenarios/: Contains components for the PipelineOrchestrator's scenario generation:

        evolutionary_generator.py: Pymoo-based evolutionary algorithm (distinct from GAPS-E's).

        contradiction_analyzer.py: Langchain-based contradiction analysis.

    src/pipeline/orchestrator.py: APScheduler-based task scheduler for data ingestion and (TFP/Pymoo) forecasting/scenario generation.

8. Data Flow (Simplified)

    Data Fetchers (arxiv_fetcher, etc.) -> Collect data.

    Data published to Redis Streams (stream:raw_arxiv, etc.).

    kg_redis_consumer.py -> Reads from Redis Streams.

    Consumer processes data -> Loads into Neo4j Database.

    Forecasting Models (tfp_hybrid_forecaster, pymc_forecaster, GAPS-E hybrid_probabilistic_forecaster):

        TFP/PyMC forecasters can read historical/feature data from Neo4j to train and predict.

        GAPS-E forecaster uses trained models (NN, BayesianRidge) to evaluate ScenarioGenome objects.

    Scenario Generators:

        GAPS-E EvolutionaryScenarioGenerator -> Uses LLM to generate/evolve ScenarioGenomes. These are evaluated by the GAPS-E forecaster and ContradictionAnalysisEngine.

        PipelineOrchestrator's evolutionary_generator.py (Pymoo) -> Generates scenarios, potentially evaluated by its contradiction_analyzer.py and TFP forecaster.

    Outputs are scenarios, narratives, and probability assessments.

9. Maintenance and Monitoring

    Log Files: Check log files (default: logs/gaps_system.log) for errors and operational messages from components using src/utils/logging.py.

    Redis Monitoring:

        Use redis-cli to check stream lengths (XLEN stream_name), consumer group info (XINFO GROUPS stream_name), and pending messages.

        Monitor Redis memory usage.

    Neo4j Monitoring:

        Use Neo4j Browser or cypher-shell to inspect graph data.

        Monitor Neo4j logs and performance metrics.

    Model Retraining:

        The GAPS-E forecaster models (ProbabilisticNN, BayesianRidge) will need periodic retraining as new (real) scenario data becomes available. Run train_probabilistic_nn.py with updated data.

        The TFP and PyMC forecasters in src/forecasting/ should also be retrained periodically by calling their train_sts_model / train_gbm_model / train_model methods. This can be scheduled or done manually.

    API Key Rotation: Update API keys in .env as needed.

    Dependency Updates: Periodically update Python libraries: pip list --outdated and pip install -U <package>. Test thoroughly after updates.

    Configuration Backups: Keep config.yaml and .env (excluding actual secrets if committed to a private repo) under version control.

10. Troubleshooting Common Issues

    ImportErrors:

        Ensure your virtual environment is activated.

        Ensure you are running Python commands from the project root directory.

        Verify PYTHONPATH if necessary, especially for IDEs.

        Check that all dependencies in requirements.txt are installed.

    Connection Errors (Redis, Neo4j, APIs):

        Verify the service is running.

        Check host, port, username, password, api_key in config.yaml and .env.

        Check network connectivity and firewalls.

    Configuration Errors:

        Validate config.yaml syntax (YAML linters can help).

        Ensure keys referenced in Python code match exactly those in config.yaml.

    API Key Issues:

        "Invalid API Key" errors: Double-check the key in .env. Ensure the correct environment variable name is used by the code (e.g., OPENAI_API_KEY, NEWS_API_KEY).

        Rate limits: You might hit API rate limits. Implement backoff/retry logic in fetchers if this is frequent.

    Model Files Not Found (GAPS-E):

        Ensure train_probabilistic_nn.py has been run successfully and saved models to the correct paths (default: models/).

        Verify model_save_path, scaler_save_path, etc., in config.yaml under gapse_settings.forecaster match where files are saved/expected.

    Neo4j Query Errors:

        If the kg_redis_consumer or forecasters fail with Neo4j errors, the Cypher queries likely don't match your graph schema. Adapt them carefully. Use Neo4j Browser to test queries.

    LLM Issues:

        Ensure the LLM client in EvolutionaryScenarioGenerator is correctly initialized.

        Check prompt structures and expected JSON output formats. LLMs can sometimes deviate.

        Monitor token usage and costs.

11. Future Development & TODOs

    Implement Missing Fetchers: Create fetchers for patents, markets, and social data sources as defined in config.yaml.

    Real Training Data for GAPS-E Forecaster: Replace dummy data generation in train_probabilistic_nn.py with a robust pipeline for loading/creating labeled ScenarioGenome data.

    Robust LLM Interaction: Enhance error handling, retries, and JSON parsing for LLM calls in EvolutionaryScenarioGenerator. Make LLM choice more flexible (e.g., support local models via Ollama more directly if not OpenAI).

    Complete Pipeline Orchestrator: Fully implement _ingestion_phase and _forecasting_phase in src/pipeline/orchestrator.py. Ensure its components (src/forecasting/hybrid_forecaster.py, src/scenarios/evolutionary_generator.py) are fully functional and potentially trained.

    Refine Neo4j Schema and Queries: Develop a more detailed and optimized Neo4j graph model and ensure all Cypher queries are efficient and correct.

    User Interface/Dashboard: Expand the conceptual Streamlit dashboard (docs/backend.md mentions one) for better interaction, scenario review, and visualization.

    Advanced Contradiction Analysis: Enhance ContradictionAnalysisEngine with more sophisticated NLP techniques beyond simple pattern matching.

    Bayesian Model in GAPS-E: Implement a more complex Bayesian model (e.g., PyMC) in src/gapse_subsystem/hybrid_probabilistic_forecaster.py to replace or augment the BayesianRidge placeholder.

    Comprehensive Testing: Add unit tests and integration tests for various components.

    Scalability and Performance Optimization: For larger-scale deployment, consider optimizations for data processing, model inference, and database interactions.

    Documentation: Keep this README and other documentation (docs/) updated as the system evolves.




```
