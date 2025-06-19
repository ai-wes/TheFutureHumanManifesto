import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Any
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # For feature scaling
from joblib import dump, load as joblib_load # For saving/loading scaler
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load as joblib_load
import os # For path joining and checking
from sklearn.linear_model import BayesianRidge
from hybrid_probabilistic_forecaster import ProbabilisticNN
from evolutionary_scenario_generator import ScenarioGenome # Or from utils.models if moved

from config_loader import ConfigLoader
from custom_logging import Logger

logger = Logger("nn_trainer")
config_loader = ConfigLoader(config_path=os.getenv("GAPS_CONFIG_PATH", "config/config.yaml"))

# --- Configuration ---
MODEL_SAVE_PATH = config_loader.get("gapse_settings.forecaster.model_save_path", "models/probabilistic_nn.pth")
SCALER_SAVE_PATH = config_loader.get("gapse_settings.forecaster.scaler_save_path", "models/feature_scaler.joblib")
NN_INPUT_DIM = config_loader.get("gapse_settings.forecaster.nn_input_dim", 50) # Must match feature extraction
HIDDEN_DIM1 = config_loader.get("gapse_settings.forecaster.nn_hidden1", 128)
HIDDEN_DIM2 = config_loader.get("gapse_settings.forecaster.nn_hidden2", 64)
DROPOUT_RATE = config_loader.get("gapse_settings.forecaster.nn_dropout", 0.2)
LEARNING_RATE = config_loader.get("gapse_settings.training.learning_rate", 0.001)
NUM_EPOCHS = config_loader.get("gapse_settings.training.num_epochs", 100)
BATCH_SIZE = config_loader.get("gapse_settings.training.batch_size", 32)
DUMMY_DATA_SIZE = config_loader.get("gapse_settings.training.dummy_data_size", 1000) # For generating dummy data
VECTORIZER_SAVE_PATH = config_loader.get("gapse_settings.forecaster.vectorizer_save_path", "models/tfidf_vectorizer.joblib")
TFIDF_MAX_FEATURES = config_loader.get("gapse_settings.forecaster.tfidf_max_features", NN_INPUT_DIM - 5) # Reserve 5 for other numerical features
BAYESIAN_MODEL_SAVE_PATH = config_loader.get("gapse_settings.forecaster.bayesian_model_save_path", "models/bayesian_ridge_model.joblib")
# --- Dataset Class ---
# --- Modified Feature Extractor (to be used by Dataset, but vectorizer passed in) ---
def extract_features_for_dataset(scenario: ScenarioGenome, vectorizer: TfidfVectorizer, max_total_features: int) -> np.ndarray:
    text_parts = scenario.technological_factors + \
                 scenario.social_factors + \
                 scenario.economic_factors + \
                 scenario.key_events + \
                 [scenario.timeline]
    full_text = " ".join(text_parts)
    
    text_features_sparse = vectorizer.transform([full_text]) # Use transform, not fit_transform
    text_features = text_features_sparse.toarray().flatten()
    
    # The vectorizer should already be fitted with the correct number of max_features (TFIDF_MAX_FEATURES)
    # So, text_features should have length TFIDF_MAX_FEATURES

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
    domain_diversity = len(set(scenario.domains)) if scenario.domains else 0
    
    numerical_features = np.array([
        timeline_duration, timeline_start_norm, num_key_events, avg_event_length, domain_diversity
    ], dtype=np.float32) # Ensure float32 for PyTorch

    combined_features = np.concatenate([text_features, numerical_features]).astype(np.float32)
    
    # Ensure final length matches max_total_features (NN_INPUT_DIM)
    if len(combined_features) < max_total_features:
        combined_features = np.pad(combined_features, (0, max_total_features - len(combined_features)), 'constant', constant_values=0.0)
    return combined_features[:max_total_features]


class ScenarioDataset(Dataset):
    def __init__(self, genomes: List[ScenarioGenome], targets: List[float], 
                 vectorizer: TfidfVectorizer, scaler: Optional[StandardScaler], 
                 nn_input_dim: int):
        self.genomes = genomes
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.nn_input_dim = nn_input_dim

        raw_features_list = [extract_features_for_dataset(g, self.vectorizer, self.nn_input_dim) for g in self.genomes]
        self.features_np = np.array(raw_features_list, dtype=np.float32)

        if self.scaler:
            self.features_np = self.scaler.transform(self.features_np)
        self.features = torch.FloatTensor(self.features_np)

    def __len__(self):
        return len(self.genomes)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]



# --- Loss Function ---
class NLLLossGaussian(nn.Module):
    """ Negative Log-Likelihood for a Gaussian distribution. """
    def __init__(self):
        super().__init__()

    def forward(self, mean_pred: torch.Tensor, var_pred: torch.Tensor, target: torch.Tensor):
        # Ensure variance is positive and add epsilon for stability
        var_pred_stable = torch.clamp(var_pred, min=1e-6)
        log_var = torch.log(var_pred_stable)
        sq_error = (target - mean_pred).pow(2)
        
        # NLL = 0.5 * (log(2*pi*sigma^2) + (y - mu)^2 / sigma^2)
        # We can ignore constant terms like log(2*pi) for optimization
        loss = 0.5 * (log_var + sq_error / var_pred_stable)
        return loss.mean() # Average loss over the batch

# --- Training Function ---
def train_model(
    model: ProbabilisticNN,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device
):
    model.to(device)
    logger.info(f"Starting training on {device} for {num_epochs} epochs.")

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0.0
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            mean_pred, var_pred = model(features)
            loss = criterion(mean_pred, var_pred, targets)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)
        log_msg = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}"

        if val_loader:
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    mean_pred, var_pred = model(features)
                    loss = criterion(mean_pred, var_pred, targets)
                    val_loss_accum += loss.item()
            avg_val_loss = val_loss_accum / len(val_loader)
            log_msg += f", Val Loss: {avg_val_loss:.6f}"

        logger.info(log_msg)

    logger.info("Training finished.")

def outcome_to_target_probability(outcome_str: Optional[str], prediction_date_str: str, actual_outcome_date_str: Optional[str], timeline_str: str) -> float:
    """
    Converts a textual outcome to a target probability, considering timeliness.
    This is a heuristic and can be refined.
    """
    if not outcome_str:
        return 0.5  # Neutral if outcome unknown

    outcome_str = outcome_str.lower()
    base_prob = 0.5 # Start with neutral

    # Determine predicted end year (approximate)
    predicted_end_year = None
    try:
        if "by " in timeline_str.lower():
            year_part = timeline_str.lower().split("by ")[-1].strip()
            if year_part.isdigit() and len(year_part) == 4:
                predicted_end_year = int(year_part)
        elif "-" in timeline_str:
            predicted_end_year = int(timeline_str.split('-')[-1].strip())
    except:
        pass # Could not parse predicted_end_year

    # Determine actual outcome year
    actual_year = None
    if actual_outcome_date_str and actual_outcome_date_str.lower() not in ["tbd", "n/a", "never"]:
        try:
            actual_year = int(actual_outcome_date_str.split('-')[0])
        except:
            pass
    
    timing_penalty = 0.0
    if predicted_end_year and actual_year:
        diff = abs(actual_year - predicted_end_year)
        if diff <= 2: timing_penalty = 0.0       # Very close
        elif diff <= 5: timing_penalty = 0.05    # Close
        elif diff <= 10: timing_penalty = 0.15   # A bit late/early
        else: timing_penalty = 0.25             # Significantly off

    if "realized as predicted" in outcome_str:
        base_prob = np.random.uniform(0.9, 0.98)
    elif "realized (close)" in outcome_str or "realized (early)" in outcome_str:
        base_prob = np.random.uniform(0.8, 0.9)
    elif "realized for 50+ years" in outcome_str: # For Moore's Law type
         base_prob = np.random.uniform(0.95, 1.0)
    elif "realized in different form" in outcome_str:
        base_prob = np.random.uniform(0.6, 0.75)
    elif "partially realized" in outcome_str:
        base_prob = np.random.uniform(0.4, 0.6)
    elif "nearly realized" in outcome_str: # For China economy example
        base_prob = np.random.uniform(0.75, 0.85)
    elif "not realized (timeline)" in outcome_str or "realized (slightly late)" in outcome_str or "realized much later" in outcome_str:
        base_prob = np.random.uniform(0.3, 0.5) # It happened, but timing was off
    elif "not realized" in outcome_str or "never" in outcome_str or "failed" in outcome_str:
        base_prob = np.random.uniform(0.02, 0.1)
    elif "completely wrong" in outcome_str or "massively underestimated" in outcome_str or "spectacularly false" in outcome_str:
        base_prob = np.random.uniform(0.01, 0.05)
    elif "pending" in outcome_str:
        # For pending predictions, we can't assign a definite outcome probability.
        # Option 1: Skip them for training.
        # Option 2: Assign a neutral probability (0.5) or one based on current expert sentiment.
        # Option 3: Try to estimate a "current likelihood" based on notes (very advanced).
        # For now, let's skip them or assign neutral.
        logger.debug(f"Skipping pending prediction or assigning neutral: {outcome_str}")
        return 0.5 # Or you could return None and filter these out later

    final_prob = np.clip(base_prob - timing_penalty, 0.01, 0.99)
    return final_prob


def load_historical_predictions_data(json_file_path: str) -> Tuple[List[ScenarioGenome], List[float]]:
    genomes = []
    targets = []
    if not os.path.exists(json_file_path):
        logger.error(f"Historical data file not found: {json_file_path}. Cannot load real data.")
        return genomes, targets # Return empty lists

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        for item_dict in data_list:
            metadata = item_dict.get('metadata', {})
            actual_outcome = metadata.get('actual_outcome')
            
            # Skip pending predictions for now, or assign them a special target if your model can handle it
            if actual_outcome and "pending" in actual_outcome.lower():
                logger.debug(f"Skipping pending prediction ID {item_dict.get('id')} for training.")
                continue

            # Reconstruct ScenarioGenome from dict
            genome_id = item_dict.get('id', str(uuid.uuid4())) # Ensure ID
            if isinstance(genome_id, int): genome_id = str(genome_id) # Ensure ID is string

            genome = ScenarioGenome(
                id=genome_id,
                technological_factors=item_dict.get('technological_factors', []),
                social_factors=item_dict.get('social_factors', []),
                economic_factors=item_dict.get('economic_factors', []),
                timeline=item_dict.get('timeline', "Unknown Timeline"),
                key_events=item_dict.get('key_events', []),
                domains_focused=item_dict.get('domains_focused', []), # Ensure this key matches ScenarioGenome
                # These might not be in historical data, so provide defaults
                probability_weights=item_dict.get('probability_weights', {}),
                fitness_score=item_dict.get('fitness_score'), # Will be None
                generation=item_dict.get('generation', -1), # Mark as historical
                parent_ids=item_dict.get('parent_ids', [])
            )
            genomes.append(genome)
            
            target_prob = outcome_to_target_probability(
                actual_outcome,
                metadata.get("prediction_date"),
                metadata.get("actual_outcome_date"),
                genome.timeline
            )
            targets.append(target_prob)
            
        logger.info(f"Loaded {len(genomes)} historical (non-pending) prediction data samples from {json_file_path}")
    except Exception as e:
        logger.error(f"Failed to load/process historical training data from {json_file_path}: {e}", exc_info=True)
        # Return empty lists on error to prevent fallback to dummy if real data is intended
        return [], [] 
    return genomes, targets



# --- Main Execution ---
if __name__ == "__main__":
    
    
    config_loader = ConfigLoader(config_path=os.getenv("GAPS_CONFIG_PATH", r"F:\TheFutureHumanManifesto\config\config.yaml"))
    # (This line is already present in your script, so we'll use that instance)

    print(f"--- EFFECTIVE CONFIGURATION VALUES USED BY TRAINING SCRIPT (Path: {config_loader.config_path}) ---")
    print(f"MODEL_SAVE_PATH: {MODEL_SAVE_PATH} (from config: {config_loader.get('gapse_settings.forecaster.model_save_path')})")
    print(f"SCALER_SAVE_PATH: {SCALER_SAVE_PATH} (from config: {config_loader.get('gapse_settings.forecaster.scaler_save_path')})")
    print(f"VECTORIZER_SAVE_PATH: {VECTORIZER_SAVE_PATH} (from config: {config_loader.get('gapse_settings.forecaster.vectorizer_save_path')})")
    print(f"BAYESIAN_MODEL_SAVE_PATH: {BAYESIAN_MODEL_SAVE_PATH} (from config: {config_loader.get('gapse_settings.forecaster.bayesian_model_save_path')})")
    print(f"NN_INPUT_DIM: {NN_INPUT_DIM} (from config: {config_loader.get('gapse_settings.forecaster.nn_input_dim')})")
    print(f"TFIDF_MAX_FEATURES: {TFIDF_MAX_FEATURES} (from config: {config_loader.get('gapse_settings.forecaster.tfidf_max_features')})")
    print(f"HIDDEN_DIM1: {HIDDEN_DIM1} (from config: {config_loader.get('gapse_settings.forecaster.nn_hidden1')})")
    print(f"HIDDEN_DIM2: {HIDDEN_DIM2} (from config: {config_loader.get('gapse_settings.forecaster.nn_hidden2')})")
    print(f"DROPOUT_RATE: {DROPOUT_RATE} (from config: {config_loader.get('gapse_settings.forecaster.nn_dropout')})")
    print(f"LEARNING_RATE: {LEARNING_RATE} (from config: {config_loader.get('gapse_settings.training.learning_rate')})")
    print(f"NUM_EPOCHS: {NUM_EPOCHS} (from config: {config_loader.get('gapse_settings.training.num_epochs')})")
    print(f"BATCH_SIZE: {BATCH_SIZE} (from config: {config_loader.get('gapse_settings.training.batch_size')})")
    print(f"DUMMY_DATA_SIZE: {DUMMY_DATA_SIZE} (from config: {config_loader.get('gapse_settings.training.dummy_data_size')})")
    print("----------------------------------------------------------------------")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VECTORIZER_SAVE_PATH), exist_ok=True) # For TfidfVectorizer
    os.makedirs(os.path.dirname(BAYESIAN_MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Replace dummy data generation with:
    historical_data_file = "data/historical_predictions_labeled.json" # Path to your JSON file
    logger.info(f"Loading historical prediction data from {historical_data_file}...")

    all_genomes, all_targets = load_historical_predictions_data(historical_data_file)

    if not all_genomes: # If loading failed or file was empty/only pending
        logger.warning(f"No usable historical data loaded from {historical_data_file}. Falling back to dummy data.")
        train_genomes, val_genomes, train_targets, val_targets = train_test_split(
                all_genomes, all_targets, test_size=0.2, random_state=42
            )

    # 2a. Fit TfidfVectorizer on training texts
    # --- Train and Save TfidfVectorizer ---
    train_texts = [" ".join(g.technological_factors + g.social_factors + g.economic_factors + g.key_events + [g.timeline]) for g in train_genomes]
    tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    tfidf_vectorizer.fit(train_texts)
    logger.info(f"TfidfVectorizer fitted on training texts with {len(tfidf_vectorizer.vocabulary_)} features.")
    try:
        dump(tfidf_vectorizer, VECTORIZER_SAVE_PATH)
        logger.info(f"TfidfVectorizer saved to {VECTORIZER_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not save TfidfVectorizer: {e}")

    # --- Extract features for training set to fit scaler and Bayesian model ---
    # Use the extract_features_for_dataset which now takes the fitted vectorizer
    train_features_np = np.array(
        [extract_features_for_dataset(g, tfidf_vectorizer, NN_INPUT_DIM) for g in train_genomes],
        dtype=np.float32
    )
    # Also extract for validation set (for potential Bayesian model evaluation, though not implemented here)
    # val_features_np = np.array(
    #     [extract_features_for_dataset(g, tfidf_vectorizer, NN_INPUT_DIM) for g in val_genomes],
    #     dtype=np.float32
    # )

    # --- Train and Save Scaler ---
    scaler = StandardScaler()
    scaler.fit(train_features_np) # Fit scaler on the combined (text+numerical) features
    logger.info("Feature scaler fitted on training data.")
    try:
        dump(scaler, SCALER_SAVE_PATH)
        logger.info(f"Feature scaler saved to {SCALER_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not save feature scaler: {e}")

    # --- Train and Save BayesianRidge Model ---
    logger.info("Training BayesianRidge model...")
    bayesian_ridge_model = BayesianRidge(max_iter=300, tol=1e-3, fit_intercept=True, compute_score=True)
    # Scale features before training Bayesian model
    train_features_scaled_for_bayes = scaler.transform(train_features_np)
    train_targets_np = np.array(train_targets, dtype=np.float32) # Ensure targets are numpy array

    bayesian_ridge_model.fit(train_features_scaled_for_bayes, train_targets_np.ravel()) # ravel targets
    logger.info(f"BayesianRidge model trained. Coefs: {bayesian_ridge_model.coef_[:5]}..., Intercept: {bayesian_ridge_model.intercept_}")
    try:
        dump(bayesian_ridge_model, BAYESIAN_MODEL_SAVE_PATH)
        logger.info(f"BayesianRidge model saved to {BAYESIAN_MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not save BayesianRidge model: {e}")

    # --- Create Datasets and DataLoaders for NN (uses fitted vectorizer and scaler) ---
    train_dataset = ScenarioDataset(train_genomes, train_targets, tfidf_vectorizer, scaler, NN_INPUT_DIM)
    val_dataset = ScenarioDataset(val_genomes, val_targets, tfidf_vectorizer, scaler, NN_INPUT_DIM)
    # ... (rest of NN training: DataLoaders, Model init, Optimizer, Criterion, train_model call, Save NN model) ...
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.info(f"Created DataLoaders. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    model = ProbabilisticNN(
        input_dim=NN_INPUT_DIM,
        hidden_dim1=HIDDEN_DIM1,
        hidden_dim2=HIDDEN_DIM2,
        dropout_rate=DROPOUT_RATE
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = NLLLossGaussian()
    logger.info("NN Model, optimizer, and criterion initialized.")

    train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device)

    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logger.info(f"Trained NN model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not save trained NN model: {e}")