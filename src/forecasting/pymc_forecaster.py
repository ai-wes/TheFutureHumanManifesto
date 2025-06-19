import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os

# Assuming these utilities are in place
from utils.config_loader import ConfigLoader
from utils.logging import get_logger

logger = get_logger("pymc_forecaster")

class PyMCBayesianForecaster:
    def __init__(self, neo4j_driver, config_loader_instance: ConfigLoader):
        self.driver = neo4j_driver # Neo4j driver instance
        self.config_loader = config_loader_instance
        self.models_trace = {} # To store trained models (idata - InferenceData)

        # PyMC specific settings from config (if any)
        self.pm_draws = self.config_loader.get("forecasting.pymc.draws", 2000)
        self.pm_tune = self.config_loader.get("forecasting.pymc.tune", 1000)
        self.pm_chains = self.config_loader.get("forecasting.pymc.chains", 2)
        self.pm_cores = self.config_loader.get("forecasting.pymc.cores", 1)
        self.neo4j_db_name = self.config_loader.get('neo4j.database', 'neo4j')


    def _run_cypher_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        if not self.driver:
            logger.error("Neo4j driver not initialized for PyMCForecaster.")
            return []
        try:
            with self.driver.session(database=self.neo4j_db_name) as session:
                logger.debug(f"PyMCForecaster running Cypher: {query[:150]}... with params: {params}")
                result = session.run(query, params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"PyMCForecaster error running Cypher query '{query[:100]}...': {e}")
            return []

    def _get_time_series_data(self, domain_identifier: str, days_history: int = 365*2) -> Optional[pd.DataFrame]:
        """
        Fetches and prepares time series data for a given domain identifier.
        The domain_identifier is expected to map to a 'source_name' in Neo4j.
        Returns a DataFrame with 'time' (numeric, days since start) and 'value'.
        """
        logger.info(f"Fetching time series data for PyMC model for source: {domain_identifier}")
        
        # Query to count items per day for the given source
        # Similar to the TFP HybridForecaster's STS query
        query = """
        MATCH (item)
        WHERE item.source_name = $source_identifier AND item.publishedAt IS NOT NULL
        WITH date(item.publishedAt) AS publication_date, count(item) AS daily_count
        WHERE publication_date IS NOT NULL AND publication_date >= date() - duration({days: $days_limit})
        RETURN publication_date AS observation_time, daily_count AS value
        ORDER BY observation_time
        """
        params = {"source_identifier": domain_identifier.lower(), "days_limit": days_history}
        data_records = self._run_cypher_query(query, params)

        if not data_records or len(data_records) < 30: # Need a reasonable amount of data
            logger.warning(f"Not enough data (found {len(data_records)}) for PyMC model for source: {domain_identifier}")
            return None

        df = pd.DataFrame(data_records)
        df['observation_time'] = pd.to_datetime(df['observation_time'].apply(lambda x: x.to_native() if hasattr(x, 'to_native') else x))
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['value', 'observation_time'], inplace=True)
        df.sort_values('observation_time', inplace=True)

        if df.empty or len(df) < 30:
            logger.warning(f"No valid data after processing for PyMC model for source: {domain_identifier}")
            return None

        # Resample to daily frequency, sum counts, and fill missing days with 0
        df = df.set_index('observation_time').resample('D').sum().fillna(0).reset_index()
        
        # Create a numeric time index (days since the start of the series)
        df['time'] = (df['observation_time'] - df['observation_time'].min()).dt.days.astype(float)
        
        logger.info(f"Prepared time series for {domain_identifier} with {len(df)} daily data points.")
        return df[['time', 'value', 'observation_time']]


    def train_model(self, domain_identifier: str, days_history: int = 365*2):
        """
        Trains a Bayesian time series model for the given domain_identifier.
        A simple Bayesian linear regression with time as predictor.
        """
        df_ts = self._get_time_series_data(domain_identifier, days_history)
        if df_ts is None or df_ts.empty:
            logger.error(f"Cannot train PyMC model for {domain_identifier} due to lack of data.")
            return

        time_idx = df_ts['time'].values
        values = df_ts['value'].values

        logger.info(f"Defining PyMC model for {domain_identifier}...")
        with pm.Model() as bayesian_linear_model:
            # Priors
            intercept = pm.Normal("intercept", mu=np.mean(values), sigma=np.std(values) * 2) # Centered around mean
            slope = pm.Normal("slope", mu=0, sigma=1) # Weakly informative prior for trend
            sigma = pm.HalfCauchy("sigma", beta=np.std(values)) # Noise level

            # Likelihood
            mu = intercept + slope * time_idx
            likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=values)

            logger.info(f"Starting MCMC sampling for {domain_identifier} (draws={self.pm_draws}, tune={self.pm_tune})...")
            try:
                idata = pm.sample(draws=self.pm_draws, tune=self.pm_tune, chains=self.pm_chains, cores=self.pm_cores, progressbar=True)
                self.models_trace[domain_identifier] = idata
                logger.info(f"PyMC model training for {domain_identifier} complete.")
                
                # Log summary (optional)
                # summary = az.summary(idata, var_names=["intercept", "slope", "sigma"])
                # logger.debug(f"Model summary for {domain_identifier}:\n{summary}")

            except Exception as e:
                logger.error(f"Error during PyMC sampling for {domain_identifier}: {e}")


    def forecast(self, domain_identifier: str, steps: int) -> Optional[Dict[str, Any]]:
        """
        Generates forecasts using the trained PyMC model.
        'steps' is the number of future days to forecast.
        """
        if domain_identifier not in self.models_trace:
            logger.warning(f"No trained PyMC model found for {domain_identifier}. Please train first.")
            # Attempt to train if not found
            logger.info(f"Attempting to train PyMC model for {domain_identifier} before forecasting...")
            self.train_model(domain_identifier)
            if domain_identifier not in self.models_trace: # Check again
                logger.error(f"Training failed or model still not available for {domain_identifier}.")
                return None
        
        idata = self.models_trace[domain_identifier]
        
        # Get the last time index from the original data used for training
        # This requires storing the original data or its last time index.
        # For simplicity, let's re-fetch the last observation time to determine future time indices.
        # A more robust way would be to store the original df_ts or its properties.
        
        original_df_ts = self._get_time_series_data(domain_identifier) # Re-fetch to get context
        if original_df_ts is None or original_df_ts.empty:
            logger.error(f"Could not get original data context for forecasting {domain_identifier}.")
            return None
            
        last_time_idx = original_df_ts['time'].max()
        last_observation_date = original_df_ts['observation_time'].max()

        # Create future time indices
        future_time_indices = np.arange(last_time_idx + 1, last_time_idx + 1 + steps)
        future_dates = [last_observation_date + timedelta(days=i) for i in range(1, steps + 1)]

        logger.info(f"Generating PyMC forecast for {domain_identifier} for {steps} future steps...")
        with pm.Model() as forecast_model: # Rebuild model structure for prediction
            intercept = pm.Normal("intercept", mu=0, sigma=10) # Use broad priors or load from idata
            slope = pm.Normal("slope", mu=0, sigma=1)
            sigma = pm.HalfCauchy("sigma", beta=5)
            
            # Define mu using future time indices
            mu_forecast = intercept + slope * future_time_indices
            
            # Sample from posterior predictive distribution
            # This uses the trace (idata) from the trained model
            ppc = pm.sample_posterior_predictive(
                idata, 
                var_names=["likelihood", "intercept", "slope"], # Include likelihood to get predictions
                model=forecast_model, # Provide the model context
                random_seed=42
            )
        
        # Extract forecast samples for 'likelihood'
        forecast_samples = ppc.posterior_predictive["likelihood"].values
        # Reshape: (chains * draws, steps)
        num_chains, num_draws, num_steps_pred = forecast_samples.shape
        forecast_samples_flat = forecast_samples.reshape(num_chains * num_draws, num_steps_pred)

        # Calculate mean, stddev, and credible intervals
        forecast_mean = np.mean(forecast_samples_flat, axis=0)
        forecast_stddev = np.std(forecast_samples_flat, axis=0)
        hdi_3 = az.hdi(forecast_samples_flat, hdi_prob=0.03).T # For 97% interval (approx 3 sigma)
        hdi_97 = az.hdi(forecast_samples_flat, hdi_prob=0.97).T # For 97% interval

        # Ensure hdi arrays are correctly indexed if they come out as (2, steps)
        lower_ci = np.percentile(forecast_samples_flat, 2.5, axis=0)
        upper_ci = np.percentile(forecast_samples_flat, 97.5, axis=0)


        return {
            "domain": domain_identifier,
            "forecast_dates": [d.isoformat() for d in future_dates],
            "mean_forecast": forecast_mean.tolist(),
            "stddev_forecast": forecast_stddev.tolist(),
            "credible_interval_lower": lower_ci.tolist(),
            "credible_interval_upper": upper_ci.tolist(),
            "raw_samples": forecast_samples_flat[:, ::max(1, steps//10)].tolist() # Subsample raw samples for size
        }

if __name__ == "__main__":
    # This __main__ block requires a running Neo4j instance and a valid config.
    # Create a dummy config for testing if needed
    if not os.path.exists("config/config.yaml"):
        os.makedirs("config", exist_ok=True)
        with open("config/config.yaml", "w") as f:
            f.write("neo4j:\n  uri: neo4j://localhost:7687\n  username: neo4j\n  password: password\n  database: neo4j\n")
            f.write("forecasting:\n  pymc:\n    draws: 500\n    tune: 500\n    chains: 1\n    cores: 1\n") # Faster for demo
        logger.info("Created dummy config/config.yaml for PyMCForecaster demo.")

    try:
        config = ConfigLoader(config_path="config/config.yaml")
        neo4j_conf = config.get_neo4j_config()
        
        from neo4j import GraphDatabase # Import here for example
        driver = GraphDatabase.driver(neo4j_conf['uri'], auth=(neo4j_conf['username'], neo4j_conf['password']))
        driver.verify_connectivity() # Check connection
        logger.info("Neo4j connection verified for PyMC demo.")

        forecaster = PyMCBayesianForecaster(neo4j_driver=driver, config_loader_instance=config)
        
        # Example: Assuming you have data for 'arxiv' source in Neo4j
        # To make this runnable without real data, you might need to mock _get_time_series_data
        # or ensure some data exists.
        # For now, let's try to run it. If no data, it should log warnings.
        
        test_domain = "arxiv" # This should match a 'source_name' in your Neo4j data
        logger.info(f"--- Training PyMC model for domain: {test_domain} ---")
        forecaster.train_model(test_domain, days_history=180) # Use shorter history for demo

        if test_domain in forecaster.models_trace:
            logger.info(f"--- Generating PyMC forecast for domain: {test_domain} ---")
            forecast_result = forecaster.forecast(test_domain, steps=30)
            if forecast_result:
                logger.info(f"Forecast for {test_domain}:")
                logger.info(f"  Dates (first 3): {forecast_result['forecast_dates'][:3]}")
                logger.info(f"  Mean (first 3): {[round(x,2) for x in forecast_result['mean_forecast'][:3]]}")
                logger.info(f"  StdDev (first 3): {[round(x,2) for x in forecast_result['stddev_forecast'][:3]]}")
            else:
                logger.error(f"Failed to generate forecast for {test_domain}.")
        else:
            logger.warning(f"Model for {test_domain} was not trained successfully, skipping forecast.")

        driver.close()

    except Exception as e:
        logger.critical(f"Error in PyMCForecaster demo: {e}", exc_info=True)