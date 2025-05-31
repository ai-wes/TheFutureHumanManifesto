# src/forecasting/hybrid_forecaster.py
import tensorflow_probability as tfp
import pymc as pm
from sklearn.ensemble import GradientBoostingRegressor

class HybridForecaster:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.sts_models = {}
        self.gbm_models = {}
        
    def train_sts_model(self, domain):
        # Structural Time Series with TensorFlow Probability
        query = f"""
        MATCH (n:{domain})-[r]->(m)
        RETURN n.timestamp as date, r.influence as value
        """
        data = self._run_cypher(query)
        
        model = tfp.sts.Sum([
            tfp.sts.LocalLinearTrend(observed_time_series=data['value']),
            tfp.sts.Seasonal(num_seasons=4, observed_time_series=data['value'])
        ])
        
        variational_posterior = tfp.sts.build_factored_surrogate_posterior(model)
        losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=model.joint_distribution(
                observed_time_series=data['value']).log_prob,
            surrogate_posterior=variational_posterior,
            optimizer=tf.optimizers.Adam(0.1),
            num_steps=200)
        
        self.sts_models[domain] = (model, variational_posterior)

    def train_gbm_model(self, domain):
        # Gradient Boosting Machine for point estimates
        query = f"""
        MATCH (n:{domain})
        RETURN n.features as X, n.value as y
        """
        data = self._run_cypher(query)
        
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(data['X'], data['y'])
        self.gbm_models[domain] = model

    def forecast(self, domain, steps):
        sts_model, posterior = self.sts_models[domain]
        gbm_model = self.gbm_models[domain]
        
        # Bayesian uncertainty
        samples = posterior.sample(1000)
        sts_forecast = tfp.sts.forecast(
            sts_model,
            observed_time_series=data['value'],
            parameter_samples=samples,
            num_steps=steps)
        
        # Point estimate blending
        gbm_forecast = gbm_model.predict(sts_forecast.mean())
        
        return self._combine_forecasts(sts_forecast, gbm_forecast)
