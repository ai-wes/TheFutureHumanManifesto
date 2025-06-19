import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler # For loading scaler
from joblib import load as joblib_load # For loading scaler
from joblib import dump, load as joblib_load
import os 




from models import ScenarioGenome
from config_loader import ConfigLoader
from logging import get_logger

logger = get_logger("hybrid_probabilistic_forecaster")


# --- ProbabilisticNN (remains the same) ---
class ProbabilisticNN(nn.Module):
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
        self.mean_head = nn.Linear(hidden_dim2, 1)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim2, 1),
            nn.Softplus()
        )
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        mean = torch.sigmoid(self.mean_head(encoded))
        variance = self.var_head(encoded) + 1e-6
        return mean, variance

def extract_features_from_genome_for_inference(scenario: ScenarioGenome, vectorizer: Optional[TfidfVectorizer], max_total_features: int, tfidf_max_features: int):
    text_parts = scenario.technological_factors + \
                 scenario.social_factors + \
                 scenario.economic_factors + \
                 scenario.key_events + \
                 [scenario.timeline]
    full_text = " ".join(text_parts)

    if vectorizer:
        try:
            text_features_sparse = vectorizer.transform([full_text]) # Use transform
            text_features = text_features_sparse.toarray().flatten()
        except Exception as e:
            logger.error(f"Error transforming text with TfidfVectorizer: {e}. Using zeros.")
            text_features = np.zeros(tfidf_max_features) # Use the expected number of features
    else:
        logger.warning("TfidfVectorizer not loaded. Using zeros for text features.")
        text_features = np.zeros(tfidf_max_features) # Use the expected number of features

    # Ensure text_features has the correct length (tfidf_max_features)
    if len(text_features) < tfidf_max_features:
        text_features = np.pad(text_features, (0, tfidf_max_features - len(text_features)), 'constant')
    elif len(text_features) > tfidf_max_features:
        text_features = text_features[:tfidf_max_features]
        
    # ... (numerical feature extraction remains the same as in train_probabilistic_nn.py's extract_features_for_dataset)
    try:
        timeline_parts = scenario.timeline.split('-')
        start_year = int(timeline_parts[0])
        end_year = int(timeline_parts[-1])
        timeline_duration = end_year - start_year if end_year > start_year else 5
        timeline_start_norm = (start_year - 2020) / 50
    except:
        timeline_duration = 10
        timeline_start_norm = 0.5
    num_key_events = len(scenario.key_events)
    avg_event_length = np.mean([len(event.split()) for event in scenario.key_events]) if scenario.key_events else 0
    domain_diversity = len(set(scenario.domains_focused)) if scenario.domains_focused else 0
    numerical_features = np.array([
        timeline_duration, timeline_start_norm, num_key_events, avg_event_length, domain_diversity
    ], dtype=np.float32)

    combined_features = np.concatenate([text_features, numerical_features]).astype(np.float32)
    if len(combined_features) < max_total_features:
        combined_features = np.pad(combined_features, (0, max_total_features - len(combined_features)), 'constant', constant_values=0.0)
    return combined_features[:max_total_features]


class HybridProbabilisticForecaster:
    def __init__(self, config_loader_instance: Optional[ConfigLoader] = None):
        # ... (config loading remains the same) ...
        if config_loader_instance:
            self.config_loader = config_loader_instance
        else:
            config_path = os.getenv("GAPS_CONFIG_PATH", "config/config.yaml")
            self.config_loader = ConfigLoader(config_path=config_path)

        gapse_config = self.config_loader.get("gapse_settings", {})
        forecaster_config = gapse_config.get("forecaster", {}) # Use gapse_settings.forecaster

        self.nn_input_dim = forecaster_config.get("nn_input_dim", 50)
        self.model_path = forecaster_config.get("model_save_path", "models/probabilistic_nn.pth")
        self.scaler_path = forecaster_config.get("scaler_save_path", "models/feature_scaler.joblib")
        self.vectorizer_path = forecaster_config.get("vectorizer_save_path", "models/tfidf_vectorizer.joblib") # New
        self.tfidf_max_features = forecaster_config.get("tfidf_max_features", self.nn_input_dim - 5) # New

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_predictor = ProbabilisticNN(
            input_dim=self.nn_input_dim,
            hidden_dim1=forecaster_config.get("nn_hidden1", 128),
            hidden_dim2=forecaster_config.get("nn_hidden2", 64),
            dropout_rate=forecaster_config.get("nn_dropout", 0.2)
        ).to(self.device)

        self.feature_scaler: Optional[StandardScaler] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None # New

        self._load_model_scaler_vectorizer() # Modified method name

        self.ensemble_weights = forecaster_config.get("ensemble_weights", {'bayesian': 0.4, 'neural': 0.6})
        # Initialize Bayesian model (placeholder for now, will be replaced)
        self.bayesian_model = None # Will be loaded/initialized in _load_model_scaler_vectorizer
        self.bayesian_model_path = forecaster_config.get("bayesian_model_save_path", "models/bayesian_ridge_model.joblib") # New


    def _load_model_scaler_vectorizer(self): # Renamed
        # Load NN Model
        try:
            if os.path.exists(self.model_path):
                self.neural_predictor.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.neural_predictor.eval()
                logger.info(f"Neural predictor model loaded from {self.model_path}")
            else:
                logger.warning(f"NN Model weights not found at {self.model_path}.")
        except Exception as e:
            logger.error(f"Error loading NN model weights from {self.model_path}: {e}.")

        # Load Scaler
        try:
            if os.path.exists(self.scaler_path):
                self.feature_scaler = joblib_load(self.scaler_path)
                logger.info(f"Feature scaler loaded from {self.scaler_path}")
            else:
                logger.warning(f"Feature scaler not found at {self.scaler_path}.")
        except Exception as e:
            logger.error(f"Error loading feature scaler from {self.scaler_path}: {e}.")

        # Load TfidfVectorizer
        try:
            if os.path.exists(self.vectorizer_path):
                self.tfidf_vectorizer = joblib_load(self.vectorizer_path)
                logger.info(f"TfidfVectorizer loaded from {self.vectorizer_path}")
            else:
                logger.warning(f"TfidfVectorizer not found at {self.vectorizer_path}.")
        except Exception as e:
            logger.error(f"Error loading TfidfVectorizer from {self.vectorizer_path}: {e}.")
        
        # Load Bayesian Model
        try:
            if os.path.exists(self.bayesian_model_path):
                self.bayesian_model = joblib_load(self.bayesian_model_path)
                logger.info(f"Bayesian model loaded from {self.bayesian_model_path}")
            else:
                logger.warning(f"Bayesian model not found at {self.bayesian_model_path}. Bayesian predictions will be dummied.")
        except Exception as e:
            logger.error(f"Error loading Bayesian model: {e}")


    def _bayesian_predict(self, features_raw: np.ndarray) -> Dict[str, float]:
        if self.bayesian_model:
            # Assume features_raw might need scaling if the Bayesian model was trained on scaled features
            # For BayesianRidge, it's often robust to unscaled features, but consistency is key.
            # If scaler was used for Bayesian model training, apply it here too.
            # For simplicity, let's assume BayesianRidge was trained on features similar to NN input.
            
            # If the Bayesian model was trained on scaled features:
            features_for_bayesian = features_raw.reshape(1, -1)
            if self.feature_scaler: # Use the same scaler as NN if applicable
                try:
                    features_for_bayesian = self.feature_scaler.transform(features_for_bayesian)
                except Exception as e:
                    logger.error(f"Error scaling features for Bayesian model: {e}. Using raw.")

            try:
                mean_pred, std_pred = self.bayesian_model.predict(features_for_bayesian, return_std=True)
                # BayesianRidge predict returns arrays, get the single value
                mean = float(mean_pred[0])
                variance = float(std_pred[0]**2)
                logger.debug(f"Bayesian prediction: mean={mean:.4f}, var={variance:.4f}")
                return {'mean': np.clip(mean, 0.0, 1.0), 'variance': max(1e-6, variance)}
            except Exception as e:
                logger.error(f"Error during Bayesian model prediction: {e}")
        
        # Fallback dummy prediction
        logger.warning("Bayesian model not loaded or prediction failed. Using dummy Bayesian prediction.")
        simulated_mean = np.random.uniform(0.2, 0.8)
        simulated_variance = np.random.uniform(0.01, 0.05)
        return {'mean': simulated_mean, 'variance': simulated_variance}


    def _neural_predict(self, features_raw: np.ndarray) -> Dict[str, float]:
        # ... (scaling logic inside here remains the same) ...
        features_for_nn = features_raw.reshape(1, -1)
        if self.feature_scaler:
            try:
                features_for_nn = self.feature_scaler.transform(features_for_nn)
            except Exception as e:
                logger.error(f"Error applying feature scaler for NN: {e}. Using raw features.")
        
        features_tensor = torch.FloatTensor(features_for_nn).to(self.device)
        with torch.no_grad():
            mean, variance = self.neural_predictor(features_tensor)
        logger.debug(f"Neural prediction: mean={mean.item():.4f}, var={variance.item():.4f}")
        return {'mean': mean.item(), 'variance': variance.item()}

    def predict_scenario_probability(self, scenario: ScenarioGenome) -> Dict[str, float]:
        # Use the inference-specific feature extractor
        features_raw = extract_features_from_genome_for_inference(
            scenario, 
            self.tfidf_vectorizer, 
            max_total_features=self.nn_input_dim,
            tfidf_max_features=self.tfidf_max_features
        )
        # ... (rest of the ensembling logic remains the same) ...
        bayesian_pred = self._bayesian_predict(features_raw)
        neural_pred = self._neural_predict(features_raw)

        ensemble_mean = (self.ensemble_weights['bayesian'] * bayesian_pred['mean'] +
                         self.ensemble_weights['neural'] * neural_pred['mean'])
        # Var(aX + bY) = a^2 Var(X) + b^2 Var(Y) assuming independence
        ensemble_variance = (self.ensemble_weights['bayesian']**2 * bayesian_pred['variance'] +
                             self.ensemble_weights['neural']**2 * neural_pred['variance'])
        ensemble_mean = np.clip(ensemble_mean, 0.0, 1.0)
        ensemble_variance = max(1e-6, ensemble_variance) # Ensure positive variance
        std_dev = np.sqrt(ensemble_variance)
        z_score = 1.96
        conf_interval_lower = np.clip(ensemble_mean - z_score * std_dev, 0.0, 1.0)
        conf_interval_upper = np.clip(ensemble_mean + z_score * std_dev, 0.0, 1.0)

        return {
            'probability': float(ensemble_mean),
            'variance': float(ensemble_variance),
            'std_dev': float(std_dev),
            'confidence_interval_lower': float(conf_interval_lower),
            'confidence_interval_upper': float(conf_interval_upper),
            'uncertainty_metric': float(std_dev),
            'model_contributions': {
                'bayesian_mean': bayesian_pred['mean'], 'bayesian_variance': bayesian_pred['variance'],
                'neural_mean': neural_pred['mean'], 'neural_variance': neural_pred['variance']
            }
        }
    # ... (example usage if __name__ == '__main__':)