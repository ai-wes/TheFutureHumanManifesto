import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
# Assuming ScenarioGenome is defined in evolutionary_scenario_generator
from .evolutionary_scenario_generator import ScenarioGenome

# A more sophisticated feature extractor might be needed.
# This is a simplified version.
def extract_features_from_genome(scenario: ScenarioGenome, max_features=30) -> np.ndarray:
    """
    Extracts numerical features from a ScenarioGenome for ML models.
    This is a placeholder and would need to be significantly more robust.
    """
    # Combine textual components
    text_parts = scenario.technological_factors + \
                 scenario.social_factors + \
                 scenario.economic_factors + \
                 scenario.key_events + \
                 [scenario.timeline]
    full_text = " ".join(text_parts)

    # TF-IDF on text components (very basic text featurization)
    # In a real system, pre-trained embeddings (e.g., Sentence-BERT) would be better.
    # Also, the vectorizer should be fit on a larger corpus, not on single scenarios.
    # For now, this is just illustrative.
    try:
        # This is problematic: TfidfVectorizer should be fit on a corpus, not instance by instance.
        # For a real system, fit it once on a representative dataset of scenario texts.
        # Here, we simulate this by fitting on the current text, which is not ideal for consistency.
        vectorizer = TfidfVectorizer(max_features=max_features-5 if max_features > 5 else max_features) # Reserve some for other features
        text_features = vectorizer.fit_transform([full_text]).toarray().flatten()
        if len(text_features) < (max_features - 5): # Pad if fewer features extracted
             text_features = np.pad(text_features, (0, (max_features - 5) - len(text_features)), 'constant')
        text_features = text_features[:max_features-5] # Ensure fixed size

    except ValueError: # Handles empty vocabulary
        text_features = np.zeros(max_features-5 if max_features > 5 else max_features)


    # Timeline encoding (e.g., duration or start/end year normalized)
    try:
        start_year, end_year = map(int, scenario.timeline.split('-'))
        timeline_duration = end_year - start_year
        timeline_start_norm = (start_year - 2020) / 50 # Normalize start year (assuming 2020 base, 50yr range)
    except:
        timeline_duration = 10 # Default
        timeline_start_norm = 0.5 # Default

    # Event complexity score (e.g., number of events, average length)
    num_key_events = len(scenario.key_events)
    avg_event_length = np.mean([len(event.split()) for event in scenario.key_events]) if scenario.key_events else 0

    # Domain diversity score
    domain_diversity = len(set(scenario.domains_focused)) if scenario.domains_focused else 0

    numerical_features = np.array([
        timeline_duration, timeline_start_norm, num_key_events, avg_event_length, domain_diversity
    ], dtype=float)

    # Combine all features
    combined_features = np.concatenate([text_features, numerical_features]).astype(float)

    # Ensure fixed length (e.g., for NN input)
    # This should match the input_dim of the neural network
    # If max_features is the total desired dimension:
    if len(combined_features) < max_features:
        combined_features = np.pad(combined_features, (0, max_features - len(combined_features)), 'constant')

    return combined_features[:max_features]


class ProbabilisticNN(nn.Module):
    """Neural network for probabilistic forecasting, predicting mean and variance."""
    def __init__(self, input_dim=50, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Output head for the mean of the probability
        self.mean_head = nn.Linear(hidden_dim2, 1)
        # Output head for the variance (or log-variance for stability)
        # Using Softplus to ensure variance is positive
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim2, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        mean = torch.sigmoid(self.mean_head(encoded)) # Sigmoid to keep probability between 0 and 1
        variance = self.var_head(encoded) + 1e-6 # Add small epsilon for numerical stability
        return mean, variance


class HybridProbabilisticForecaster:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}

        # Neural Network Predictor
        self.nn_input_dim = self.config.get("forecaster_nn_input_dim", 50) # Must match feature extraction output
        self.neural_predictor = ProbabilisticNN(
            input_dim=self.nn_input_dim,
            hidden_dim1=self.config.get("forecaster_nn_hidden1", 128),
            hidden_dim2=self.config.get("forecaster_nn_hidden2", 64),
            dropout_rate=self.config.get("forecaster_nn_dropout", 0.2)
        )
        # In a real system, you would load pre-trained weights for the neural_predictor
        # self.neural_predictor.load_state_dict(torch.load("path/to/model_weights.pth"))
        self.neural_predictor.eval() # Set to evaluation mode if pre-trained

        # Bayesian Model (Placeholder - PyMC3/4 or similar would be used here)
        # self.bayesian_model = self._load_bayesian_model() # Placeholder

        self.ensemble_weights = self.config.get("forecaster_ensemble_weights", {'bayesian': 0.4, 'neural': 0.6})

    def _train_neural_predictor(self, training_data: List[Tuple[ScenarioGenome, float]]):
        """
        Placeholder for training the neural network.
        Requires a dataset of (scenario_features, actual_outcome_probability).
        This is a complex task involving data collection and labeling.
        """
        # optimizer = torch.optim.Adam(self.neural_predictor.parameters(), lr=0.001)
        # criterion = NLLLoss_Gaussian() # Custom loss for mean and variance
        # self.neural_predictor.train()
        # for epoch in range(num_epochs):
        #     for scenario_genome, target_prob in training_data:
        #         features = extract_features_from_genome(scenario_genome, max_features=self.nn_input_dim)
        #         features_tensor = torch.FloatTensor(features).unsqueeze(0) # Add batch dim
        #
        #         optimizer.zero_grad()
        #         mean_pred, var_pred = self.neural_predictor(features_tensor)
        #
        #         # Loss calculation needs to be defined based on target_prob
        #         # If target_prob is just a point, and we predict a distribution,
        #         # we might use Negative Log Likelihood of a Gaussian.
        #         # loss = criterion(mean_pred, var_pred, torch.FloatTensor([target_prob]))
        #         # loss.backward()
        #         # optimizer.step()
        print("Neural predictor training is a placeholder. Load pre-trained model in a real system.")
        pass


    def _bayesian_predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Placeholder for Bayesian model prediction.
        This would involve running inference with a trained Bayesian model (e.g., PyMC).
        """
        # Simulate Bayesian prediction (e.g., from a Gaussian Process or Bayesian Linear Regression)
        # These are dummy values.
        simulated_mean = np.random.uniform(0.2, 0.8)
        simulated_variance = np.random.uniform(0.01, 0.05)
        return {'mean': simulated_mean, 'variance': simulated_variance}

    def _neural_predict(self, features: np.ndarray) -> Dict[str, float]:
        """Predicts mean and variance using the neural network."""
        features_tensor = torch.FloatTensor(features).unsqueeze(0) # Add batch dimension
        with torch.no_grad(): # No need to track gradients during inference
            mean, variance = self.neural_predictor(features_tensor)
        return {'mean': mean.item(), 'variance': variance.item()}

    def predict_scenario_probability(self, scenario: ScenarioGenome) -> Dict[str, float]:
        """
        Generates probabilistic forecasts for a scenario using a hybrid approach.
        """
        # 1. Extract features from the scenario
        # The number of features must match self.nn_input_dim
        features = extract_features_from_genome(scenario, max_features=self.nn_input_dim)

        # 2. Get predictions from individual models
        bayesian_pred = self._bayesian_predict(features) # Placeholder
        neural_pred = self._neural_predict(features)

        # 3. Ensemble prediction (weighted average for mean and variance)
        # Simple weighted average for mean:
        ensemble_mean = (self.ensemble_weights['bayesian'] * bayesian_pred['mean'] +
                         self.ensemble_weights['neural'] * neural_pred['mean'])

        # For variance of a weighted sum of random variables (assuming independence for simplicity, which may not hold):
        # Var(aX + bY) = a^2 Var(X) + b^2 Var(Y) + 2ab Cov(X,Y)
        # Assuming Cov(X,Y) = 0 for simplicity (this is a strong assumption)
        ensemble_variance = (self.ensemble_weights['bayesian']**2 * bayesian_pred['variance'] +
                             self.ensemble_weights['neural']**2 * neural_pred['variance'])

        # Ensure probability is within [0, 1] and variance is non-negative
        ensemble_mean = np.clip(ensemble_mean, 0.0, 1.0)
        ensemble_variance = max(0.0, ensemble_variance) # Should be positive due to Softplus and epsilon

        # Calculate confidence interval (e.g., 95% CI assuming Gaussian distribution)
        # Z-score for 95% CI is approx 1.96
        z_score = 1.96
        std_dev = np.sqrt(ensemble_variance)
        conf_interval_lower = np.clip(ensemble_mean - z_score * std_dev, 0.0, 1.0)
        conf_interval_upper = np.clip(ensemble_mean + z_score * std_dev, 0.0, 1.0)

        return {
            'probability': float(ensemble_mean),
            'variance': float(ensemble_variance),
            'std_dev': float(std_dev),
            'confidence_interval_lower': float(conf_interval_lower),
            'confidence_interval_upper': float(conf_interval_upper),
            'uncertainty_metric': float(std_dev), # Using std_dev as a simple uncertainty metric
            'model_contributions': { # Optional: to see individual model outputs
                'bayesian_mean': bayesian_pred['mean'], 'bayesian_variance': bayesian_pred['variance'],
                'neural_mean': neural_pred['mean'], 'neural_variance': neural_pred['variance']
            }
        }

# Example Usage:
if __name__ == '__main__':
    forecaster = HybridProbabilisticForecaster(config={"forecaster_nn_input_dim": 30}) # Match max_features in dummy extractor

    # Example ScenarioGenome
    @dataclass
    class ScenarioGenome: # Minimal mock for testing
        technological_factors: List[str]
        social_factors: List[str]
        economic_factors: List[str]
        timeline: str
        key_events: List[str]
        domains_focused: List[str]

    test_scenario = ScenarioGenome(
        technological_factors=["AGI breakthrough", "Fusion power online"],
        social_factors=["Universal Basic Income adopted", "AI rights debates"],
        economic_factors=["Mass job displacement in transport", "New markets in virtual experiences"],
        timeline="2035-2045",
        key_events=["True AGI demonstrated (2038)", "First city powered by fusion (2042)"],
        domains_focused=["artificial_general_intelligence", "energy"]
    )

    prediction = forecaster.predict_scenario_probability(test_scenario)
    print("Hybrid Probabilistic Forecast:")
    for key, value in prediction.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value:.4f}")
        else:
            print(f"  {key}: {value:.4f}")