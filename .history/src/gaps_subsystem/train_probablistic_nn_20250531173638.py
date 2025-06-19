# src/gapse_subsystem/train_probabilistic_nn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Any
import os
import uuid # For generating IDs if historical data misses them

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import BayesianRidge
from joblib import dump, load as joblib_load

# Corrected imports based on project structure
from config_loader import ConfigLoader
from custom_logging import get_logger
logger = get_logger("nn_trainer")
try:
    from src.utils.models import ScenarioGenome # Preferred location
except ImportError:
    # This fallback might be problematic if ScenarioGenome in evolutionary_scenario_generator
    # is a dataclass and the one in models.py is Pydantic, or vice-versa.
    # Ensure consistency or handle the difference.
    logger.warning("Could not import ScenarioGenome from src.utils.models, trying from src.gapse_subsystem.evolutionary_scenario_generator")
    from evolutionary_scenario_generator import ScenarioGenome

# ProbabilisticNN should be defined in hybrid_probabilistic_forecaster
from hybrid_probabilistic_forecaster import ProbabilisticNN

logger = get_logger("nn_trainer")
config_loader = ConfigLoader(config_path=os.getenv("GAPS_CONFIG_PATH", r"F:\TheFutureHumanManifesto\config\config.yaml"))
json_file_path = r"F:\TheFutureHumanManifesto\data\historical_predictions.json"
# --- Configuration ---
MODEL_SAVE_PATH = config_loader.get("gapse_settings.forecaster.model_save_path", "models/probabilistic_nn.pth")
SCALER_SAVE_PATH = config_loader.get("gapse_settings.forecaster.scaler_save_path", "models/feature_scaler.joblib")
VECTORIZER_SAVE_PATH = config_loader.get("gapse_settings.forecaster.vectorizer_save_path", "models/tfidf_vectorizer.joblib")
BAYESIAN_MODEL_SAVE_PATH = config_loader.get("gapse_settings.forecaster.bayesian_model_save_path", "models/bayesian_ridge_model.joblib")

NN_INPUT_DIM = int(config_loader.get("gapse_settings.forecaster.nn_input_dim", 50))
TFIDF_MAX_FEATURES = int(config_loader.get("gapse_settings.forecaster.tfidf_max_features", NN_INPUT_DIM - 5)) # Assuming 5 numerical features
HIDDEN_DIM1 = int(config_loader.get("gapse_settings.forecaster.nn_hidden1", 128))
HIDDEN_DIM2 = int(config_loader.get("gapse_settings.forecaster.nn_hidden2", 64))
DROPOUT_RATE = float(config_loader.get("gapse_settings.forecaster.nn_dropout", 0.2))

LEARNING_RATE = float(config_loader.get("gapse_settings.training.learning_rate", 0.001))
NUM_EPOCHS = int(config_loader.get("gapse_settings.training.num_epochs", 100))
BATCH_SIZE = int(config_loader.get("gapse_settings.training.batch_size", 32))
DUMMY_DATA_SIZE = int(config_loader.get("gapse_settings.training.dummy_data_size", 1000))
HISTORICAL_DATA_PATH = r"F:\TheFutureHumanManifesto\data\historical_predictions.json"


# --- Feature Extractor ---
def extract_features_for_dataset(scenario: ScenarioGenome, vectorizer: TfidfVectorizer, max_total_features: int) -> np.ndarray:
    text_parts = (scenario.technological_factors or []) + \
                 (scenario.social_factors or []) + \
                 (scenario.economic_factors or []) + \
                 (scenario.key_events or []) + \
                 ([scenario.timeline] if scenario.timeline else [])
    full_text = " ".join(filter(None, text_parts)) # Filter None and join
    
    text_features_sparse = vectorizer.transform([full_text if full_text else ""]) # Use transform, handle empty text
    text_features = text_features_sparse.toarray().flatten()

    # Ensure text_features has the expected length (TFIDF_MAX_FEATURES)
    if len(text_features) < TFIDF_MAX_FEATURES:
        text_features = np.pad(text_features, (0, TFIDF_MAX_FEATURES - len(text_features)), 'constant', constant_values=0.0)
    elif len(text_features) > TFIDF_MAX_FEATURES:
        text_features = text_features[:TFIDF_MAX_FEATURES]

    timeline_duration = 10.0  # Default
    timeline_start_norm = 0.5 # Default
    if scenario.timeline:
        try:
            timeline_parts = scenario.timeline.split('-')
            start_year_str = timeline_parts[0].strip()
            end_year_str = timeline_parts[-1].strip()
            
            start_year_num_part = ''.join(filter(str.isdigit, start_year_str))
            end_year_num_part = ''.join(filter(str.isdigit, end_year_str))

            start_year = int(start_year_num_part) if start_year_num_part else 2025
            end_year = int(end_year_num_part) if end_year_num_part else start_year + 10

            timeline_duration = float(end_year - start_year if end_year > start_year else 5)
            timeline_start_norm = float((start_year - 2020) / 50.0)
        except Exception as e:
            logger.debug(f"Could not parse timeline '{scenario.timeline}': {e}. Using defaults.")
            # Defaults are already set

    num_key_events = float(len(scenario.key_events or []))
    avg_event_length = np.mean([len(event.split()) for event in (scenario.key_events or [])]) if scenario.key_events else 0.0
    
    domain_diversity = float(len(set(scenario.domains_focused or [])))
    
    numerical_features = np.array([
        timeline_duration, timeline_start_norm, num_key_events, avg_event_length, domain_diversity
    ], dtype=np.float32)

    combined_features = np.concatenate([text_features, numerical_features]).astype(np.float32)
    
    if len(combined_features) < max_total_features:
        combined_features = np.pad(combined_features, (0, max_total_features - len(combined_features)), 'constant', constant_values=0.0)
    return combined_features[:max_total_features]


# --- Dataset Class ---
class ScenarioDataset(Dataset):
    def __init__(self, genomes: List[ScenarioGenome], targets: List[float], 
                 vectorizer: TfidfVectorizer, scaler: Optional[StandardScaler], 
                 nn_input_dim: int):
        self.genomes = genomes
        self.targets = torch.FloatTensor(targets).unsqueeze(1) # Target is probability
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.nn_input_dim = nn_input_dim

        # Extract all features once
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
        var_pred_stable = torch.clamp(var_pred, min=1e-6) # Ensure variance is positive
        log_var = torch.log(var_pred_stable)
        sq_error = (target - mean_pred).pow(2)
        loss = 0.5 * (log_var + sq_error / var_pred_stable + np.log(2 * np.pi)) # Full NLL
        return loss.mean()


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

        if val_loader and len(val_loader) > 0: # Check if val_loader is not empty
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
        else:
            log_msg += ", No validation data."


        logger.info(log_msg)
    logger.info("Training finished.")


# --- Data Loading and Preprocessing Functions ---
def outcome_to_target_probability(outcome_str: Optional[str], prediction_date_str: Optional[str], 
                                  actual_outcome_date_str: Optional[str], timeline_str: Optional[str]) -> float:
    if not outcome_str: return 0.5
    outcome_str = outcome_str.lower()
    base_prob = 0.5
    predicted_end_year, actual_year = None, None
    
    if timeline_str:
        try:
            year_parts = [int(''.join(filter(str.isdigit, part))) for part in timeline_str.split('-') if ''.join(filter(str.isdigit, part))]
            if year_parts: predicted_end_year = year_parts[-1]
            if "by " in timeline_str.lower():
                year_part_by = timeline_str.lower().split("by ")[-1].strip()
                if year_part_by.isdigit() and len(year_part_by) == 4:
                    predicted_end_year = int(year_part_by)
        except: pass

    if actual_outcome_date_str and actual_outcome_date_str.lower() not in ["tbd", "n/a", "never"]:
        try: actual_year = int(actual_outcome_date_str.split('-')[0])
        except: pass
    
    timing_penalty = 0.0
    if predicted_end_year and actual_year:
        diff = abs(actual_year - predicted_end_year)
        if diff <= 1: timing_penalty = 0.0
        elif diff <= 3: timing_penalty = 0.05
        elif diff <= 7: timing_penalty = 0.15
        else: timing_penalty = 0.25

    if "realized as predicted" in outcome_str: base_prob = np.random.uniform(0.9, 0.98)
    elif "realized (close)" in outcome_str or "realized (early)" in outcome_str: base_prob = np.random.uniform(0.8, 0.9)
    elif "realized for 50+ years" in outcome_str: base_prob = np.random.uniform(0.95, 1.0)
    elif "realized in different form" in outcome_str: base_prob = np.random.uniform(0.6, 0.75)
    elif "partially realized" in outcome_str: base_prob = np.random.uniform(0.4, 0.6)
    elif "nearly realized" in outcome_str: base_prob = np.random.uniform(0.75, 0.85)
    elif "not realized (timeline)" in outcome_str or "realized (slightly late)" in outcome_str or "realized much later" in outcome_str: base_prob = np.random.uniform(0.3, 0.5)
    elif "not realized" in outcome_str or "never" in outcome_str or "failed" in outcome_str: base_prob = np.random.uniform(0.02, 0.1)
    elif "completely wrong" in outcome_str or "massively underestimated" in outcome_str or "spectacularly false" in outcome_str: base_prob = np.random.uniform(0.01, 0.05)
    elif "pending" in outcome_str: return 0.5 # Neutral for pending, or filter out

    return np.clip(base_prob - timing_penalty, 0.01, 0.99)


def load_historical_predictions_data(json_file_path: str) -> Tuple[List[ScenarioGenome], List[float]]:
    genomes: List[ScenarioGenome] = []
    targets: List[float] = []
    if not os.path.exists(json_file_path):
        logger.error(f"Historical data file not found: {json_file_path}. Cannot load real data.")
        return genomes, targets

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        current_year = datetime.now().year

        for item_dict in data_list:
            metadata = item_dict.get('metadata', {})
            actual_outcome = metadata.get('actual_outcome')
            
            if actual_outcome and "pending" in actual_outcome.lower():
                # logger.debug(f"Skipping pending prediction ID {item_dict.get('id')} for training.")
                continue

            genome_id_val = item_dict.get('id', str(uuid.uuid4()))
            if isinstance(genome_id_val, int): genome_id_val = str(genome_id_val)

            tsp_years: Optional[float] = None
            prediction_date_str = metadata.get("prediction_date")
            if prediction_date_str:
                try:
                    pred_year_str = prediction_date_str.split('-')[0]
                    if pred_year_str.isdigit():
                        pred_year = int(pred_year_str)
                        tsp_years = float(current_year - pred_year)
                except Exception as e:
                    logger.warning(f"Could not parse prediction_date '{prediction_date_str}' for tsp_years for ID {genome_id_val}: {e}")
            
            # Assuming your Pydantic ScenarioGenome now has 'domains_focused: List[str]'
            # and 'domains: Optional[List[Domain]] = None'
            # and 'time_since_prediction_years: Optional[float] = None'
            genome_data_for_pydantic = {
                'id': genome_id_val,
                'technological_factors': item_dict.get('technological_factors', []),
                'social_factors': item_dict.get('social_factors', []),
                'economic_factors': item_dict.get('economic_factors', []),
                'timeline': item_dict.get('timeline', "Unknown Timeline"),
                'key_events': item_dict.get('key_events', []),
                'domains_focused': item_dict.get('domains_focused', []), # From your JSON
                'time_since_prediction_years': tsp_years, # Calculated
                # 'domains' (enum list) would be None unless you convert domains_focused
                'probability_weights': item_dict.get('probability_weights', {}),
                'fitness_score': item_dict.get('fitness_score'),
                'generation': item_dict.get('generation', -1),
                'parent_ids': item_dict.get('parent_ids', [])
            }
            
            try:
                genome = ScenarioGenome(**genome_data_for_pydantic)
                genomes.append(genome)
                target_prob = outcome_to_target_probability(
                    actual_outcome,
                    metadata.get("prediction_date"),
                    metadata.get("actual_outcome_date"),
                    genome.timeline
                )
                targets.append(target_prob)
            except Exception as pydantic_e:
                logger.error(f"Pydantic validation failed for item ID {genome_id_val}: {pydantic_e}. Data being passed: {genome_data_for_pydantic}", exc_info=True) # Add exc_info here too

        logger.info(f"Loaded {len(genomes)} historical (non-pending) prediction data samples from {json_file_path}")
    except Exception as e:
        logger.error(f"Failed to load/process historical training data from {json_file_path}: {e}", exc_info=True)
        return [], [] 
    return genomes, targets


def generate_dummy_scenario_data(num_samples: int) -> Tuple[List[ScenarioGenome], List[float]]:
    genomes = []
    targets = []
    available_domains_for_dummy = config_loader.get("gapse_settings.scenario_generator.available_domains", [
        "artificial_general_intelligence", "biotechnology_longevity", "brain_computer_interfaces"
    ])
    for i in range(num_samples):
        tech_factors = [f"Dummy Tech {j} for scn {i}" for j in range(np.random.randint(2, 5))]
        key_events = [f"Dummy Event {j} at year {2030+j*np.random.randint(1,5)}" for j in range(np.random.randint(1, 4))]
        genome = ScenarioGenome(
            id=f"dummy_scn_{i}_{str(uuid.uuid4())[:8]}",
            technological_factors=tech_factors,
            social_factors=[f"Dummy Social {j}" for j in range(1,3)],
            economic_factors=[f"Dummy Econ {j}" for j in range(1,3)],
            timeline=f"{2025+i%10}-{2040+i%10+np.random.randint(0,5)}",
            key_events=key_events,
            domains_focused=np.random.choice(available_domains_for_dummy, np.random.randint(1,min(3, len(available_domains_for_dummy)+1 )), replace=False).tolist()
        )
        genomes.append(genome)
        target_prob = np.clip(0.1 + 0.15 * len(tech_factors) - 0.05 * len(key_events) + np.random.normal(0, 0.1), 0.05, 0.95)
        targets.append(target_prob)
    return genomes, targets


# --- Main Execution ---
if __name__ == "__main__":
    # Debug prints for effective configuration values
    print(f"--- EFFECTIVE CONFIGURATION VALUES USED BY TRAINING SCRIPT (Path: {config_loader.config_path}) ---")
    print(f"MODEL_SAVE_PATH: {MODEL_SAVE_PATH} (from config: {config_loader.get('gapse_settings.forecaster.model_save_path')})")
    print(f"SCALER_SAVE_PATH: {SCALER_SAVE_PATH} (from config: {config_loader.get('gapse_settings.forecaster.scaler_save_path')})")
    print(f"VECTORIZER_SAVE_PATH: {VECTORIZER_SAVE_PATH} (from config: {config_loader.get('gapse_settings.forecaster.vectorizer_save_path')})")
    print(f"BAYESIAN_MODEL_SAVE_PATH: {BAYESIAN_MODEL_SAVE_PATH} (from config: {config_loader.get('gapse_settings.forecaster.bayesian_model_save_path')})")
    print(f"NN_INPUT_DIM: {NN_INPUT_DIM} (from config: {config_loader.get('gapse_settings.forecaster.nn_input_dim')})")
    print(f"TFIDF_MAX_FEATURES: {TFIDF_MAX_FEATURES} (from config: {config_loader.get('gapse_settings.forecaster.tfidf_max_features')})")
    print(f"LEARNING_RATE: {LEARNING_RATE} (from config: {config_loader.get('gapse_settings.training.learning_rate')})")
    print(f"NUM_EPOCHS: {NUM_EPOCHS} (from config: {config_loader.get('gapse_settings.training.num_epochs')})")
    print(f"BATCH_SIZE: {BATCH_SIZE} (from config: {config_loader.get('gapse_settings.training.batch_size')})")
    print(f"DUMMY_DATA_SIZE: {DUMMY_DATA_SIZE} (from config: {config_loader.get('gapse_settings.training.dummy_data_size')})")
    print(f"HISTORICAL_DATA_PATH: {HISTORICAL_DATA_PATH} (from config: {config_loader.get('gapse_settings.training.historical_data_path')})")
    print("----------------------------------------------------------------------")

    # Ensure model/data directories exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VECTORIZER_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BAYESIAN_MODEL_SAVE_PATH), exist_ok=True)
    if HISTORICAL_DATA_PATH:
        os.makedirs(os.path.dirname(HISTORICAL_DATA_PATH), exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Attempting to load historical prediction data from {HISTORICAL_DATA_PATH}...")
    all_genomes, all_targets = load_historical_predictions_data(HISTORICAL_DATA_PATH)
    
    if not all_genomes or len(all_genomes) < BATCH_SIZE * 2 : # Ensure enough data for train/val split
        logger.warning(f"Not enough usable historical data loaded (found {len(all_genomes)}). Falling back to dummy data.")
        all_genomes, all_targets = generate_dummy_scenario_data(DUMMY_DATA_SIZE)
        if not all_genomes:
             logger.critical("Failed to generate dummy data as well. Exiting.")
             exit() # Exit if no data at all
    
    logger.info(f"Total samples for training/validation: {len(all_genomes)}")

    # Split data: ensure val_genomes is not empty if BATCH_SIZE is large relative to data
    test_size = 0.2
    if len(all_genomes) * test_size < 1: # Ensure at least one sample in validation if possible
        test_size = 1.0 / len(all_genomes) if len(all_genomes) > 1 else 0.0
    
    if test_size > 0:
        train_genomes, val_genomes, train_targets, val_targets = train_test_split(
            all_genomes, all_targets, test_size=test_size, random_state=42, stratify=None # Stratify might fail with small float targets
        )
    else: # Not enough data for a validation set
        train_genomes, train_targets = all_genomes, all_targets
        val_genomes, val_targets = [], []
        logger.warning("Too few samples for a validation set. Training on all available data.")


    # Fit TfidfVectorizer on training texts
    train_texts = [" ".join((g.technological_factors or []) + (g.social_factors or []) + (g.economic_factors or []) + (g.key_events or []) + ([g.timeline] if g.timeline else [])) for g in train_genomes]
    if not train_texts: # Handle case where train_genomes might be empty after split
        logger.critical("No text data available for training TfidfVectorizer. Exiting.")
        exit()

    tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    tfidf_vectorizer.fit(train_texts)
    logger.info(f"TfidfVectorizer fitted on training texts with {len(tfidf_vectorizer.vocabulary_)} features.")
    try:
        dump(tfidf_vectorizer, VECTORIZER_SAVE_PATH)
        logger.info(f"TfidfVectorizer saved to {VECTORIZER_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not save TfidfVectorizer: {e}")

    # Extract features for training set to fit scaler and Bayesian model
    train_features_np = np.array(
        [extract_features_for_dataset(g, tfidf_vectorizer, NN_INPUT_DIM) for g in train_genomes],
        dtype=np.float32
    )

    # Fit Scaler
    scaler = StandardScaler()
    scaler.fit(train_features_np)
    logger.info("Feature scaler fitted on training data.")
    try:
        dump(scaler, SCALER_SAVE_PATH)
        logger.info(f"Feature scaler saved to {SCALER_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not save feature scaler: {e}")

    # Train BayesianRidge Model
    logger.info("Training BayesianRidge model...")
    bayesian_ridge_model = BayesianRidge(max_iter=500, tol=1e-3, fit_intercept=True, compute_score=True)
    train_features_scaled_for_bayes = scaler.transform(train_features_np) # Use the same scaler
    train_targets_np = np.array(train_targets, dtype=np.float32)
    bayesian_ridge_model.fit(train_features_scaled_for_bayes, train_targets_np.ravel())
    logger.info(f"BayesianRidge model trained. Coefs (first 5): {bayesian_ridge_model.coef_[:5]}..., Intercept: {bayesian_ridge_model.intercept_}")
    try:
        dump(bayesian_ridge_model, BAYESIAN_MODEL_SAVE_PATH)
        logger.info(f"BayesianRidge model saved to {BAYESIAN_MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not save BayesianRidge model: {e}")

    # Create DataLoaders for NN
    train_dataset = ScenarioDataset(train_genomes, train_targets, tfidf_vectorizer, scaler, NN_INPUT_DIM)
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True) # Adjust batch size if dataset is small
    
    val_loader = None
    if val_genomes and val_targets:
        val_dataset = ScenarioDataset(val_genomes, val_targets, tfidf_vectorizer, scaler, NN_INPUT_DIM)
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, len(val_dataset)), shuffle=False)
    
    logger.info(f"Created DataLoaders. Train size: {len(train_dataset)}, Val size: {len(val_dataset) if val_loader else 0}")

    # Initialize and Train NN Model
    model = ProbabilisticNN(
        input_dim=NN_INPUT_DIM,
        hidden_dim1=HIDDEN_DIM1,
        hidden_dim2=HIDDEN_DIM2,
        dropout_rate=DROPOUT_RATE
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = NLLLossGaussian()
    logger.info("NN Model, optimizer, and criterion initialized.")

    if len(train_loader) > 0: # Only train if there's training data
        train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device)
        try:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Trained NN model saved to {MODEL_SAVE_PATH}")
        except Exception as e:
            logger.error(f"Could not save trained NN model: {e}")
    else:
        logger.warning("No training data for NN model. Skipping NN training.")