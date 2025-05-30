# src/forecasting/sts_forecaster.py
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd # For data handling

# It's good practice to clearly define dependencies, even if they are mocked here.
# from neo4j import GraphDatabase # Example if you were to connect to Neo4j directly

# Placeholder for fetching historical data and features
# In a real application, this would connect to your data source (e.g., Neo4j, a data warehouse)
def get_historical_milestone_data_and_features(milestone_name: str, neo4j_driver=None):
    """
    Fetches historical data for a given milestone and relevant features.
    This is a placeholder and should be implemented to query your actual data source.
    """
    print(f"Fetching historical data for milestone: {milestone_name} (using placeholder data)")
    
    # Dummy data for demonstration:
    # Let's simulate data for "AGI Achieved Year"
    # Observed values could be expert predictions collected over years, or a proxy metric.
    # Timesteps (e.g., years)
    years_observed = np.arange(2010, 2025) 
    
    # Target variable: e.g., consensus predicted year for AGI, or a progress score towards it.
    # For this example, let's assume it's a decreasing predicted year (getting closer).
    observed_target_values = np.array([
        2045, 2043, 2042, 2040, 2038, 2039, 2037, 2035, 2033, 2032, 2030, 2031, 2029, 2028, 2027
    ], dtype=np.float32) 
    
    # Example features (regressors) that might come from a Knowledge Graph or other sources:
    # These should be normalized or scaled appropriately before use.
    # Feature 1: Normalized growth rate of AI research papers
    num_ai_papers_growth_rate = np.linspace(0.1, 0.8, len(years_observed)).astype(np.float32).reshape(-1, 1)
    # Feature 2: Index of available compute power (e.g., log-scaled)
    compute_power_idx = np.logspace(1, 2, len(years_observed)).astype(np.float32).reshape(-1, 1)
    
    # Combine features into a design matrix
    # Shape: (num_timesteps, num_features)
    feature_matrix = np.concatenate([num_ai_papers_growth_rate, compute_power_idx], axis=1)
    
    # It's crucial that observed_target_values and feature_matrix are aligned in time.
    # The first dimension of feature_matrix must match the length of observed_target_values.
    
    return years_observed, observed_target_values, feature_matrix

# TensorFlow Probability distributions and bijectors
tfd = tfp.distributions
tfb = tfp.bijectors

class STSModel:
    def __init__(self, milestone_name: str, neo4j_driver=None):
        self.milestone_name = milestone_name
        self.neo4j_driver = neo4j_driver # Pass driver if used in data fetching
        
        # Fetch and prepare data
        self.years, self.observed_values, self.features = get_historical_milestone_data_and_features(
            self.milestone_name, 
            self.neo4j_driver
        )
        
        # Ensure observed_values is a 1D array for STS
        if self.observed_values.ndim > 1:
            self.observed_values = self.observed_values.squeeze()
            
        self.model = self._build_model()

    def _build_model(self):
        # Define structural components for the time series
        # LocalLinearTrend is a common choice for series with evolving trends.
        trend = tfp.sts.LocalLinearTrend(observed_time_series=self.observed_values)
        
        # Define regression component for external features (covariates)
        # The design_matrix should have shape (num_timesteps, num_features)
        # and match the `observed_time_series`.
        design_matrix = tf.convert_to_tensor(self.features, dtype=tf.float32)
        linear_regression = tfp.sts.LinearRegression(design_matrix=design_matrix)

        # Combine components into a Sum model
        # Add seasonality or other components as needed for your specific data.
        model = tfp.sts.Sum(
            components=[trend, linear_regression], 
            observed_time_series=self.observed_values
        )
        return model

    def fit(self, num_variational_steps=200, learning_rate=0.1, verbose=True):
        """Fits the STS model using variational inference."""
        if verbose:
            print(f"Fitting STS model for {self.milestone_name}...")
            
        # Build the surrogate posterior
        variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=self.model)
        
        # Define the optimizer
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Define the training step with tf.function for performance
        @tf.function(jit_compile=True)
        def train_step():
            with tf.GradientTape() as tape:
                loss = tfp.vi.monte_carlo_variational_loss(
                    target_log_prob_fn=self.model.joint_log_prob(
                        observed_time_series=self.observed_values),
                    surrogate_posterior=variational_posteriors,
                    sample_size=1 # Can increase for better gradient estimates, at cost of speed
                )
            gradients = tape.gradient(loss, variational_posteriors.trainable_variables)
            optimizer.apply_gradients(zip(gradients, variational_posteriors.trainable_variables))
            return loss

        for i in range(num_variational_steps):
            loss = train_step()
            if verbose and (i % 50 == 0 or i == num_variational_steps - 1):
                print(f"Step {i+1}/{num_variational_steps}, Loss: {loss.numpy():.4f}")
        
        self.surrogate_posterior = variational_posteriors
        if verbose:
            print("Fitting complete.")
        return variational_posteriors

    def forecast(self, num_steps_forecast=10, num_samples=1000, future_features=None):
        """Generates forecasts from the fitted model."""
        if not hasattr(self, 'surrogate_posterior'):
            raise ValueError("Model has not been fit yet. Call .fit() first.")

        if future_features is None:
            # Simple extrapolation for features: repeat the last observed feature values.
            # WARNING: This is a naive approach and should be replaced with proper feature forecasting.
            print("Warning: No future_features provided. Using naive extrapolation (repeating last known values).")
            last_feature_values = self.features[-1, :] # Shape (num_features,)
            future_features_dummy = np.tile(last_feature_values, (num_steps_forecast, 1)) # Shape (num_steps_forecast, num_features)
            future_features_tensor = tf.convert_to_tensor(future_features_dummy, dtype=tf.float32)
        else:
            future_features_tensor = tf.convert_to_tensor(future_features, dtype=tf.float32)
            if future_features_tensor.shape[0] != num_steps_forecast:
                 raise ValueError(f"Mismatch in forecast steps ({num_steps_forecast}) and future_features length ({future_features_tensor.shape[0]}).")
            if future_features_tensor.shape[1] != self.features.shape[1]:
                 raise ValueError(f"Mismatch in number of features in future_features ({future_features_tensor.shape[1]}) and original features ({self.features.shape[1]}).")


        # Sample from the surrogate posterior to get parameter samples
        parameter_samples = self.surrogate_posterior.sample(num_samples)
        
        # Create a forecast distribution using the original model structure, observed series, 
        # parameter samples, and the design matrix for future steps.
        forecast_distribution = tfp.sts.forecast(
            model=self.model, 
            observed_time_series=self.observed_values, # Needed for context
            parameter_samples=parameter_samples,
            num_steps_forecast=num_steps_forecast,
            design_matrix_forecast=future_features_tensor # Key for feature-based forecasting
        )
        
        # Extract mean, standard deviation, and samples from the forecast distribution
        forecast_mean = forecast_distribution.mean().numpy()
        forecast_stddev = forecast_distribution.stddev().numpy()
        # Samples shape: (num_samples, num_steps_forecast) or (num_samples, num_steps_forecast, 1)
        # We usually want (num_samples, num_steps_forecast)
        forecast_samples_raw = forecast_distribution.sample().numpy() 
        if forecast_samples_raw.ndim == 3 and forecast_samples_raw.shape[-1] == 1:
            forecast_samples = forecast_samples_raw.squeeze(-1)
        else:
            forecast_samples = forecast_samples_raw


        # Determine the years for the forecast period
        last_observed_year = self.years[-1] if len(self.years) > 0 else 2024 # Fallback if years is empty
        forecast_years = last_observed_year + np.arange(1, num_steps_forecast + 1)

        return {
            "mean": forecast_mean.flatten(), # Ensure 1D array
            "stddev": forecast_stddev.flatten(), # Ensure 1D array
            "samples": forecast_samples, # Shape (num_samples, num_steps_forecast)
            "forecast_years": forecast_years
        }

if __name__ == "__main__":
    print(f"Using TensorFlow version: {tf.__version__}")
    print(f"Using TensorFlow Probability version: {tfp.__version__}")

    # Example Usage:
    agi_forecaster = STSModel(milestone_name="AGI_Achievement_Year")
    agi_forecaster.fit(num_variational_steps=200, learning_rate=0.05) # Reduced LR for potentially more stable convergence
    
    # Dummy future features for the next 10 years
    # In a real application, these would come from other predictive models or defined scenarios.
    num_future_steps = 10
    # Example: Assume AI paper growth slows down, compute power growth continues moderately.
    future_num_ai_papers_growth_dummy = np.linspace(0.8, 1.2, num_future_steps).astype(np.float32).reshape(-1,1)
    future_compute_power_index_dummy = np.logspace(2, 2.3, num_future_steps).astype(np.float32).reshape(-1,1)
    dummy_future_features_for_forecast = np.concatenate(
        [future_num_ai_papers_growth_dummy, future_compute_power_index_dummy], axis=1
    )

    forecast_results = agi_forecaster.forecast(
        num_steps_forecast=num_future_steps, 
        future_features=dummy_future_features_for_forecast,
        num_samples=500 # Number of sample paths for the forecast
    )
    
    print("\n--- Forecast Results ---")
    df_forecast_results = pd.DataFrame({
        'Year': forecast_results["forecast_years"],
        'Mean_Forecast': forecast_results["mean"],
        'StdDev_Forecast': forecast_results["stddev"]
    })
    print(df_forecast_results)
    
    # The forecast_results["samples"] (shape: num_samples, num_steps_forecast)
    # can be used for Monte Carlo simulations in the scenario generation layer.
    # For example, to get the median forecast path (50th percentile):
    median_forecast_path = np.percentile(forecast_results["samples"], 50, axis=0)
    print(f"\nMedian forecast path for the next {num_future_steps} years: {median_forecast_path}")

    # Example of accessing a few sample paths:
    print(f"\nFirst few sample paths (first 3 samples, up to 5 years):")
    for i in range(min(3, forecast_results["samples"].shape[0])):
        print(f"Sample {i+1}: {forecast_results['samples'][i, :min(5, num_future_steps)]}")
