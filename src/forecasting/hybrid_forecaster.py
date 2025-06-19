# src/forecasting/hybrid_forecaster.py
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List # Added List
from datetime import datetime # For time bucketing

from utils.config_loader import ConfigLoader # Assuming ConfigLoader is passed
from utils.logging import get_logger

logger = get_logger("tfp_hybrid_forecaster")

class HybridForecaster: # TFP-based
    def __init__(self, neo4j_driver, config_loader_instance: Optional[ConfigLoader] = None):
        self.driver = neo4j_driver
        self.config_loader = config_loader_instance
        self.sts_models = {}
        self.gbm_models = {}

        if self.config_loader:
            self.sts_variational_steps = self.config_loader.get('forecasting.sts.variational_steps', 200)
            self.sts_learning_rate = self.config_loader.get('forecasting.sts.learning_rate', 0.1)
            self.gbm_estimators = self.config_loader.get('forecasting.gbm.n_estimators', 100)
            self.neo4j_db_name = self.config_loader.get('neo4j.database', 'neo4j')
        else:
            self.sts_variational_steps = 200
            self.sts_learning_rate = 0.1
            self.gbm_estimators = 100
            self.neo4j_db_name = 'neo4j'


    def _run_cypher_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        if not self.driver:
            logger.error("Neo4j driver not initialized.")
            return []
        try:
            with self.driver.session(database=self.neo4j_db_name) as session:
                logger.debug(f"Running Cypher: {query[:150]}... with params: {params}")
                result = session.run(query, params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error running Cypher query '{query[:100]}...': {e}")
            return []

    def train_sts_model(self, domain_identifier: str): # domain_identifier could be 'arxiv', 'news_ai', etc.
        logger.info(f"Training TFP STS model for domain/source: {domain_identifier}")

        # Map domain_identifier to a source property or label in Neo4j
        # Example: if domain_identifier is "ARXIV_CS.AI", query for source='arxiv' and category='cs.AI'
        # For simplicity, let's assume domain_identifier directly maps to a 'source_name' property
        # on nodes like ArxivPaper, NewsArticle, GdeltEvent.
        # We will count items per day.
        query = """
        MATCH (item)
        WHERE item.source_name = $source_identifier AND item.publishedAt IS NOT NULL
        WITH date(item.publishedAt) AS publication_date, count(item) AS daily_count
        WHERE publication_date IS NOT NULL // Ensure date conversion was successful
        RETURN publication_date AS observation_time, daily_count AS value
        ORDER BY observation_time
        LIMIT 2000 // Limit for training stability
        """
        params = {"source_identifier": domain_identifier.lower()} # Assuming source_name is stored lowercase
        
        data_records = self._run_cypher_query(query, params)

        if not data_records or len(data_records) < 20: # Increased minimum for better seasonality
            logger.warning(f"Not enough data (found {len(data_records)}) for TFP STS model for source: {domain_identifier}")
            return

        df = pd.DataFrame(data_records)
        
        # Convert Neo4j Date/Datetime to pandas Datetime if not already
        df['observation_time'] = pd.to_datetime(df['observation_time'].apply(lambda x: x.to_native() if hasattr(x, 'to_native') else x))
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['value', 'observation_time'], inplace=True)
        df.sort_values('observation_time', inplace=True)

        # Resample to ensure regular time series (e.g., daily) and fill missing values
        # This is crucial for many STS models.
        if not df.empty:
            df = df.set_index('observation_time').resample('D').sum().fillna(0) # Resample daily, sum counts, fill NaN with 0
            observed_time_series = df['value'].values.astype(np.float32)
        else:
            logger.warning(f"No valid data after processing for TFP STS model for source: {domain_identifier}")
            return
            
        if len(observed_time_series) < 20:
            logger.warning(f"Not enough data points ({len(observed_time_series)}) after resampling for TFP STS for source: {domain_identifier}")
            return

        trend = tfp.sts.LocalLinearTrend(observed_time_series=observed_time_series)
        model_components = [trend]

        # Improved Seasonality: Add weekly and yearly if enough data
        if len(observed_time_series) > 30: # Enough for weekly
            try:
                weekly_seasonality = tfp.sts.Seasonal(
                    num_seasons=7, 
                    num_steps_per_season=1, # If data is daily
                    observed_time_series=observed_time_series,
                    name='weekly_seasonality'
                )
                model_components.append(weekly_seasonality)
            except Exception as e:
                logger.warning(f"Could not add weekly seasonality for {domain_identifier}: {e}")

        if len(observed_time_series) > 400: # Enough for yearly (approx)
            try:
                yearly_seasonality = tfp.sts.Seasonal(
                    num_seasons=365, # Or use Fourier series for more flexibility: tfp.sts.Seasonal(num_seasons=52, num_fourier_terms=10) for weekly data
                    num_steps_per_season=1, # If data is daily
                    observed_time_series=observed_time_series,
                    name='yearly_seasonality'
                )
                model_components.append(yearly_seasonality)
            except Exception as e:
                logger.warning(f"Could not add yearly seasonality for {domain_identifier}: {e}")
        
        logger.info(f"STS model for {domain_identifier} will include components: {[c.name for c in model_components]}")
        model = tfp.sts.Sum(model_components, observed_time_series=observed_time_series)
        
        variational_posterior = tfp.sts.build_factored_surrogate_posterior(model=model)
        optimizer = tf.optimizers.Adam(learning_rate=self.sts_learning_rate)

        @tf.function(experimental_compile=True)
        def train_loop():
            elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=model.joint_log_prob(observed_time_series=observed_time_series),
                surrogate_posterior=variational_posterior,
                optimizer=optimizer,
                num_steps=self.sts_variational_steps,
                jit_compile=True)
            return elbo_loss_curve

        logger.info(f"Fitting TFP STS surrogate posterior for {domain_identifier} ({self.sts_variational_steps} steps)...")
        losses = train_loop()
        final_loss = losses[-1].numpy() if hasattr(losses, 'numpy') and losses.shape[0] > 0 else 'N/A'
        logger.info(f"TFP STS model training for {domain_identifier} complete. Final ELBO loss: {final_loss}")
        self.sts_models[domain_identifier] = (model, variational_posterior, observed_time_series)

    def train_gbm_model(self, domain_identifier: str):
        from sklearn.ensemble import GradientBoostingRegressor
        logger.info(f"Training GBM model for domain/source: {domain_identifier}")
        
        # THIS QUERY IS HIGHLY SPECULATIVE AND NEEDS TO MATCH YOUR GRAPH.
        # Assumes you have nodes (e.g., 'EnrichedArticle') that have a 'featureVector'
        # (list of numbers) and a 'targetImpactScore' (a number to predict).
        # The 'domain_tag' property would link it to the domain_identifier.
        query = f"""
        MATCH (item)
        WHERE item.domain_tag = $domain_tag AND item.featureVector IS NOT NULL AND item.targetImpactScore IS NOT NULL
        RETURN item.featureVector as X, item.targetImpactScore as y
        LIMIT 5000 
        """
        # Example: domain_identifier could be "AI_Impact"
        # and nodes have item.domain_tag = "AI_Impact"
        params = {"domain_tag": domain_identifier}
        data_records = self._run_cypher_query(query, params)

        if not data_records or len(data_records) < 20:
            logger.warning(f"Not enough data to train GBM model for domain_tag: {domain_identifier}")
            return

        try:
            # Ensure X is a list of lists (or list of np.arrays) and y is a list of numbers
            X_train_list = [rec['X'] for rec in data_records if isinstance(rec.get('X'), list) and all(isinstance(x, (int, float)) for x in rec.get('X'))]
            y_train_list = [rec['y'] for rec in data_records if isinstance(rec.get('y'), (int, float))]

            if not X_train_list or len(X_train_list) != len(y_train_list):
                logger.warning(f"Data format error or mismatch for GBM training for {domain_identifier}.")
                return

            X_train = np.array(X_train_list, dtype=np.float32)
            y_train = np.array(y_train_list, dtype=np.float32)

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error processing data for GBM training for domain {domain_identifier}: {e}")
            return
        
        if X_train.shape[0] < 10:
            logger.warning(f"Insufficient valid samples ({X_train.shape[0]}) for GBM training for domain {domain_identifier}.")
            return

        model = GradientBoostingRegressor(n_estimators=self.gbm_estimators, random_state=42)
        model.fit(X_train, y_train)
        self.gbm_models[domain_identifier] = model
        logger.info(f"GBM model training for {domain_identifier} complete.")

    def forecast(self, domain: str, steps: int, future_features_gbm: Optional[np.ndarray] = None):
        logger.info(f"Generating TFP-based forecast for domain: {domain} for {steps} steps.")
        sts_forecast_dist = None
        if domain in self.sts_models:
            sts_model_struct, posterior, observed_ts = self.sts_models[domain]
            q_samples = posterior.sample(500) # Number of samples can be configured
            
            # For tfp.sts.forecast, we need to ensure the model structure is correctly passed
            # if it includes regressors, future_regressors must be provided.
            # Assuming no regressors in this simplified STS for now.
            sts_forecast_dist = tfp.sts.forecast(
                model=sts_model_struct, 
                observed_time_series=observed_ts,
                parameter_samples=q_samples,
                num_steps_forecast=steps)
            logger.info(f"TFP STS forecast distribution generated for {domain}.")
        else:
            logger.warning(f"No TFP STS model trained for domain: {domain}.")

        gbm_forecast_points = None
        if domain in self.gbm_models:
            if future_features_gbm is not None:
                if future_features_gbm.shape[0] == steps:
                    gbm_model = self.gbm_models[domain]
                    try:
                        # Ensure future_features_gbm has the correct number of features expected by the GBM
                        # This depends on how X_train was structured during train_gbm_model
                        # Example: if X_train had N features, future_features_gbm should be (steps, N)
                        if future_features_gbm.ndim == 1 and steps == 1: # Single step, single feature array
                             future_features_gbm = future_features_gbm.reshape(1, -1)
                        elif future_features_gbm.ndim == 1 and steps > 1: # Multiple steps, but 1D array implies one feature per step
                             future_features_gbm = future_features_gbm.reshape(steps, 1) # Or error if more features expected

                        # Add check for number of features if possible
                        # n_features_expected = gbm_model.n_features_in_ (if scikit-learn >= 0.24)
                        # if future_features_gbm.shape[1] != n_features_expected:
                        #    logger.error(f"GBM future_features shape mismatch. Expected {n_features_expected} features, got {future_features_gbm.shape[1]}")
                        #    gbm_forecast_points = None
                        # else:
                        gbm_forecast_points = gbm_model.predict(future_features_gbm)
                        logger.info(f"GBM point estimate generated for {domain}.")
                    except Exception as e:
                        logger.error(f"Error during GBM prediction for {domain}: {e}. Features shape: {future_features_gbm.shape}")
                else:
                    logger.warning(f"GBM future_features_gbm shape mismatch. Expected {steps} steps, got {future_features_gbm.shape[0]}.")
            else:
                logger.warning(f"Cannot generate GBM forecast for {domain}: future_features_gbm not provided.")
        else:
            logger.info(f"No GBM model trained for domain: {domain}.")

        return self._combine_forecasts(sts_forecast_dist, gbm_forecast_points)

    def _combine_forecasts(self, sts_forecast_dist, gbm_forecast_points):
        sts_mean, sts_stddev, sts_samples = None, None, None
        if sts_forecast_dist is not None:
            sts_mean = sts_forecast_dist.mean().numpy().tolist()
            sts_stddev = sts_forecast_dist.stddev().numpy().tolist()
            sts_samples = sts_forecast_dist.sample(100).numpy().tolist() # Include some samples

        gbm_fp_list = gbm_forecast_points.tolist() if gbm_forecast_points is not None else None
        
        if sts_mean is None and gbm_fp_list is None:
            logger.warning("No forecasts could be generated.")
            return {"error": "No forecast data available."}
            
        return {
            "sts_mean": sts_mean,
            "sts_stddev": sts_stddev,
            "sts_samples": sts_samples,
            "gbm_point_forecast": gbm_fp_list,
        }
        
        
        
# Example Usage (conceptual, needs actual Neo4j driver and data)
if __name__ == '__main__':
    class MockNeo4jDriver: # Mock driver for testing
        def session(self): return self
        def __enter__(self): return self
        def __exit__(self, type, value, traceback): pass
        def run(self, query, params=None):
            class MockResult:
                def data(self): return []
            return MockResult()

    mock_driver = MockNeo4jDriver()

    # Example config (replace with actual ConfigLoader if needed)
    class MockConfig:
        def get(self, key, default=None):
            if key == 'forecasting.sts.variational_steps': return 50 # Faster for demo
            if key == 'forecasting.sts.learning_rate': return 0.1
            return default

    mock_config = MockConfig()

    forecaster = HybridForecaster(neo4j_driver=mock_driver, config=mock_config)

    domain_to_forecast = "Technology" # Example domain

    print(f"Training models for domain: {domain_to_forecast}")
    forecaster.train_sts_model(domain_to_forecast)
    forecaster.train_gbm_model(domain_to_forecast) # Assumes features are available

    print(f"\nGenerating forecast for domain: {domain_to_forecast}")
    forecast_result = forecaster.forecast(domain_to_forecast, steps=12) # Forecast 12 steps ahead

    print("\nForecast Result:")
    import json
    print(json.dumps(forecast_result, indent=2))

    if forecast_result.get("sts_mean"):
        print(f"\nSTS Mean Forecast for first 3 steps: {forecast_result['sts_mean'][:3]}")
    if forecast_result.get("gbm_point_forecast"):
        print(f"GBM Point Forecast for first 3 steps: {forecast_result['gbm_point_forecast'][:3]}")
