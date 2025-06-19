# src/gapse_subsystem/train_probabilistic_nn.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Assuming src is parent of gaps_subsystem
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Any
# import os # os is already imported
import uuid # For generating IDs if historical data misses them
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import BayesianRidge
from joblib import dump, load as joblib_load

# Corrected imports based on project structure
from config_loader import ConfigLoader
from custom_logging import get_logger
logger = get_logger("nn_trainer")

# --- ScenarioGenome Import ---
from pydantic import BaseModel, Field # Using Pydantic as per your structure
class ScenarioGenome(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    technological_factors: List[str] = Field(default_factory=list)
    social_factors: List[str] = Field(default_factory=list)
    economic_factors: List[str] = Field(default_factory=list)
    timeline: Optional[str] = "2025-2050"
    key_events: List[str] = Field(default_factory=list)
    domains_focused: List[str] = Field(default_factory=list)
    probability_weights: Dict[str, float] = Field(default_factory=dict)
    fitness_score: Optional[float] = None
    generation: int = 0
    parent_ids: List[str] = Field(default_factory=list)
    time_since_prediction_years: Optional[float] = None

# ProbabilisticNN should be defined in hybrid_probabilistic_forecaster
try:
    from hybrid_probabilistic_forecaster import ProbabilisticNN
except ImportError:
    logger.error("Failed to import ProbabilisticNN from hybrid_probabilistic_forecaster. Ensure it's defined there.")
    class ProbabilisticNN(nn.Module): # Placeholder
        def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)
        def forward(self, x): return torch.sigmoid(self.fc(x)), torch.ones_like(self.fc(x)) * 0.1


config_loader = ConfigLoader(config_path=os.getenv("GAPS_CONFIG_PATH", r"F:\TheFutureHumanManifesto\config\config.yaml"))
gapse_settings = config_loader.get("gapse_settings", {})

# --- Configuration from YAML or defaults ---
forecaster_config = gapse_settings.get("forecaster", {})
training_config = gapse_settings.get("training", {})

MODEL_SAVE_PATH = forecaster_config.get("model_save_path", "models/probabilistic_nn.pth")
SCALER_SAVE_PATH = forecaster_config.get("scaler_save_path", "models/feature_scaler.joblib")
VECTORIZER_SAVE_PATH = forecaster_config.get("vectorizer_save_path", "models/tfidf_vectorizer.joblib")
BAYESIAN_MODEL_SAVE_PATH = forecaster_config.get("bayesian_model_save_path", "models/bayesian_ridge_model.joblib")

NN_INPUT_DIM = forecaster_config.get("nn_input_dim", 50)
TFIDF_MAX_FEATURES = forecaster_config.get("tfidf_max_features", NN_INPUT_DIM - 5) # Default if not in config
HIDDEN_DIM1 = forecaster_config.get("nn_hidden1", 128)
HIDDEN_DIM2 = forecaster_config.get("nn_hidden2", 64)
DROPOUT_RATE = forecaster_config.get("nn_dropout", 0.2)

LEARNING_RATE = training_config.get("learning_rate", 0.001)
NUM_EPOCHS = training_config.get("num_epochs", 100)
BATCH_SIZE = training_config.get("batch_size", 32)
DUMMY_DATA_SIZE = training_config.get("dummy_data_size", 0) # Default to 0 if not specified

HISTORICAL_DATA_PATH = training_config.get("historical_data_path", r"F:\TheFutureHumanManifesto\src\gaps_subsystem\historical_predictions.json")
REFINED_DATA_PATH = training_config.get("refined_data_with_plausibility_path", r"F:\TheFutureHumanManifesto\src\gaps_subsystem\refined_briefs_with_plausibility.json")
ORIGINAL_SYNTHETIC_DATA_PATH = training_config.get("original_synthetic_scenarios_path", r"F:\TheFutureHumanManifesto\src\gaps_subsystem\synthetic_scenarios_generated.json")


# --- Feature Extractor ---
def extract_features_for_dataset(scenario: ScenarioGenome, vectorizer: TfidfVectorizer, max_total_features: int) -> np.ndarray:
    # ... (Keep implementation as is) ...
    text_parts = (scenario.technological_factors or []) + \
                 (scenario.social_factors or []) + \
                 (scenario.economic_factors or []) + \
                 (scenario.key_events or []) + \
                 ([scenario.timeline] if scenario.timeline else [])
    full_text = " ".join(filter(None, text_parts))
    
    text_features_sparse = vectorizer.transform([full_text if full_text else ""])
    text_features = text_features_sparse.toarray().flatten()

    current_tfidf_max_features = TFIDF_MAX_FEATURES # Use the global config value
    if len(text_features) < current_tfidf_max_features:
        text_features = np.pad(text_features, (0, current_tfidf_max_features - len(text_features)), 'constant', constant_values=0.0)
    elif len(text_features) > current_tfidf_max_features:
        text_features = text_features[:current_tfidf_max_features]

    timeline_duration = 10.0
    timeline_start_norm = 0.5
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
    # ... (Keep implementation as is) ...
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
    def __len__(self): return len(self.genomes)
    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

# --- Loss Function ---
class NLLLossGaussian(nn.Module):
    # ... (Keep implementation as is) ...
    def __init__(self): super().__init__()
    def forward(self, mean_pred: torch.Tensor, var_pred: torch.Tensor, target: torch.Tensor):
        var_pred_stable = torch.clamp(var_pred, min=1e-6)
        log_var = torch.log(var_pred_stable)
        sq_error = (target - mean_pred).pow(2)
        loss = 0.5 * (log_var + sq_error / var_pred_stable + np.log(2 * np.pi))
        return loss.mean()

# --- Training Function (MODIFIED to return stats) ---
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, patience=15, min_delta=1e-4) -> Dict[str, Any]:
    model.to(device)
    logger.info(f"Starting training on {device} for {num_epochs} epochs (early stopping patience={patience}, min_delta={min_delta}).")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(2, patience//4), min_lr=1e-7)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    stopped_early_at_epoch = -1 
    final_train_loss = float('nan')
    final_val_loss = float('nan')


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
        final_train_loss = avg_train_loss # Keep track of the last training loss
        log_msg = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}"
        
        current_val_loss_epoch = float('nan')
        if val_loader and len(val_loader) > 0:
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    mean_pred, var_pred = model(features)
                    loss = criterion(mean_pred, var_pred, targets)
                    val_loss_accum += loss.item()
            avg_val_loss = val_loss_accum / len(val_loader)
            current_val_loss_epoch = avg_val_loss
            final_val_loss = avg_val_loss # Keep track of last validation loss
            log_msg += f", Val Loss: {avg_val_loss:.6f}"

            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                logger.info(f"New best val loss: {best_val_loss:.6f} at epoch {epoch+1}. Model checkpointed.")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement in val loss for {epochs_no_improve} epoch(s). (Patience={patience})")
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                    stopped_early_at_epoch = epoch + 1
                    break
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"Learning rate reduced from {old_lr} to {new_lr}")
            logger.info(f"Current learning rate: {new_lr}")
        else:
            log_msg += ", No validation data."
            # If no validation, we can't do early stopping based on val_loss,
            # so we might just train for all epochs or stop if train_loss plateaus (more complex)
            # For now, if no val_loader, it runs all epochs.
            # We can still use the scheduler with training loss if desired.
            scheduler.step(avg_train_loss) # Step scheduler on training loss if no val_loader
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr: # Assuming old_lr was defined earlier or initialized
                 logger.info(f"Learning rate reduced from {old_lr} to {new_lr} based on training loss.")

        logger.info(log_msg)

    logger.info("Training finished.")
    
    actual_epochs_run = epoch + 1 # Number of epochs actually run

    if stopped_early_at_epoch != -1 and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Model weights restored to best validation loss epoch.")
    elif best_model_state is not None: # No early stopping, but best_model_state was set (e.g. last epoch was best)
        model.load_state_dict(best_model_state)
        logger.info("Model weights set to best validation loss epoch (or last epoch if no improvement).")
    else: # No validation loader, or training stopped before best_model_state was ever set (e.g. 1 epoch run)
        logger.info("Using model weights from the last trained epoch (no validation improvements or no validation set).")


    training_stats = {
        "actual_epochs_run": actual_epochs_run,
        "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
        "stopped_early_at_epoch": stopped_early_at_epoch if stopped_early_at_epoch != -1 else None,
        "final_train_loss_at_stop": final_train_loss,
        "final_val_loss_at_stop": final_val_loss if not np.isnan(final_val_loss) else None,
        "final_learning_rate": optimizer.param_groups[0]['lr']
    }
    return training_stats

# --- Data Loading and Preprocessing Functions ---
def outcome_to_target_probability(outcome_str: Optional[str], prediction_date_str: Optional[str], 
                                  actual_outcome_date_str: Optional[str], timeline_str: Optional[str]) -> float:
    # ... (Keep implementation as is) ...
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
                if year_part_by.isdigit() and len(year_part_by) == 4: predicted_end_year = int(year_part_by)
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
    elif "pending" in outcome_str: return 0.5
    return np.clip(base_prob - timing_penalty, 0.01, 0.99)

def ensure_list(val):
    # ... (Keep implementation as is) ...
    if isinstance(val, list): return val
    elif isinstance(val, str): return [val]
    elif val is None: return []
    else: return list(val) if hasattr(val, '__iter__') else [val]

def load_historical_predictions_data(json_file_path: str) -> Tuple[List[ScenarioGenome], List[float]]:
    # ... (Keep implementation as is) ...
    logger.info(f"Loading historical predictions data from {json_file_path}")
    genomes: List[ScenarioGenome] = []
    targets: List[float] = []
    if not os.path.exists(json_file_path):
        logger.error(f"Historical data file not found: {json_file_path}. Cannot load real data.")
        return genomes, targets
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f: data_list = json.load(f)
        current_year = datetime.now().year
        for item_dict in data_list:
            metadata = item_dict.get('metadata', {})
            actual_outcome = metadata.get('actual_outcome')
            if actual_outcome and "pending" in actual_outcome.lower(): continue
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
                except Exception as e: logger.warning(f"Could not parse prediction_date '{prediction_date_str}' for ID {genome_id_val}: {e}")
            genome_data_for_pydantic = {
                'id': genome_id_val,
                'technological_factors': ensure_list(item_dict.get('technological_factors', [])),
                'social_factors': ensure_list(item_dict.get('social_factors', [])),
                'economic_factors': ensure_list(item_dict.get('economic_factors', [])),
                'timeline': item_dict.get('timeline', "Unknown Timeline"),
                'key_events': ensure_list(item_dict.get('key_events', [])),
                'domains_focused': ensure_list(item_dict.get('domains_focused', [])),
                'time_since_prediction_years': tsp_years,
                'probability_weights': item_dict.get('probability_weights', {}),
                'fitness_score': item_dict.get('fitness_score'),
                'generation': item_dict.get('generation', -1),
                'parent_ids': ensure_list(item_dict.get('parent_ids', []))
            }
            try:
                genome = ScenarioGenome(**genome_data_for_pydantic)
                genomes.append(genome)
                target_prob = outcome_to_target_probability(
                    actual_outcome, metadata.get("prediction_date"),
                    metadata.get("actual_outcome_date"), genome.timeline
                )
                targets.append(target_prob)
            except Exception as pydantic_e: logger.error(f"Pydantic validation failed for item ID {genome_id_val}: {pydantic_e}. Data: {genome_data_for_pydantic}")
        logger.info(f"Loaded {len(genomes)} historical (non-pending) samples from {json_file_path}")
    except Exception as e: logger.error(f"Failed to load/process historical data from {json_file_path}: {e}")
    return genomes, targets

def brief_to_genome(refined_brief_dict: Dict[str, Any], original_id: str, original_timeline: Optional[str], original_domains: Optional[List[str]]) -> ScenarioGenome:
    # ... (Keep implementation as is) ...
    tech_factors = [f"{item.get('driver', '')}: {item.get('implication', '')}".strip(": ") for item in refined_brief_dict.get("core_technological_drivers", [])]
    social_factors = [f"{item.get('dynamic', '')}: {item.get('implication', '')}".strip(": ") for item in refined_brief_dict.get("defining_social_dynamics", [])]
    economic_factors = [f"{item.get('transformation', '')}: {item.get('implication', '')}".strip(": ") for item in refined_brief_dict.get("key_economic_transformations", [])]
    key_events = refined_brief_dict.get("core_narrative_turning_points", [])
    strategic_overview = refined_brief_dict.get("strategic_coherence_overview", "")
    defining_challenge = refined_brief_dict.get("defining_strategic_challenge", "")
    if strategic_overview: social_factors.append(f"Strategic Overview: {strategic_overview}")
    if defining_challenge: social_factors.append(f"Defining Challenge: {defining_challenge}")
    timeline_to_use = original_timeline
    if not timeline_to_use and key_events:
        first_event_year_match = re.search(r'\b(20\d{2})\b', key_events[0])
        last_event_year_match = re.search(r'\b(20\d{2})\b', key_events[-1])
        if first_event_year_match and last_event_year_match: timeline_to_use = f"{first_event_year_match.group(1)}-{last_event_year_match.group(1)}"
        elif first_event_year_match:
             start_y = int(first_event_year_match.group(1))
             timeline_to_use = f"{start_y}-{start_y+20}"
        else: timeline_to_use = "2030-2050"
    elif not timeline_to_use: timeline_to_use = "2030-2050"
    return ScenarioGenome(id=f"refined_{original_id}", technological_factors=tech_factors, social_factors=social_factors, economic_factors=economic_factors, timeline=timeline_to_use, key_events=key_events, domains_focused=original_domains or [], generation=-10)

def load_llm_refined_scenarios_with_plausibility(plausibility_json_path: str, original_synthetic_scenarios_path: str) -> Tuple[List[ScenarioGenome], List[float]]:
    # ... (Keep implementation as is) ...
    genomes: List[ScenarioGenome] = []
    targets: List[float] = []
    if not os.path.exists(plausibility_json_path):
        logger.error(f"LLM-refined plausibility data file not found: {plausibility_json_path}")
        return genomes, targets
    original_scenarios_map = {}
    if os.path.exists(original_synthetic_scenarios_path):
        try:
            with open(original_synthetic_scenarios_path, 'r', encoding='utf-8') as f_orig:
                original_data_list = json.load(f_orig)
            original_scenarios_map = {item["id"]: {"timeline": item.get("timeline", "2025-2050"), "domains_focused": item.get("domains_focused", [])} for item in original_data_list}
        except Exception as e: logger.error(f"Failed to load original synthetic data from {original_synthetic_scenarios_path}: {e}")
    else: logger.warning(f"Original synthetic scenarios file not found at {original_synthetic_scenarios_path}. Timelines/domains may be default.")
    try:
        with open(plausibility_json_path, "r", encoding="utf-8") as f:
            refined_briefs_list = json.load(f)
        for item_wp in refined_briefs_list:
            original_id = item_wp.get("original_scenario_id")
            refined_brief = item_wp.get("refined_executive_brief")
            target_prob = item_wp.get("llm_assigned_plausibility")
            if original_id is None or refined_brief is None or target_prob is None:
                logger.warning(f"Skipping item due to missing data: {str(item_wp)[:100]}")
                continue
            original_data = original_scenarios_map.get(original_id, {})
            genome = brief_to_genome(refined_brief, original_id, original_data.get("timeline"), original_data.get("domains_focused"))
            genomes.append(genome)
            targets.append(float(target_prob))
        logger.info(f"Loaded {len(genomes)} LLM-refined scenarios with plausibility from {plausibility_json_path}")
    except Exception as e: logger.error(f"Error loading LLM-refined data from {plausibility_json_path}: {e}")
    return genomes, targets

def generate_dummy_scenario_data(num_samples: int) -> Tuple[List[ScenarioGenome], List[float]]:
    # ... (Keep implementation as is) ...
    genomes = []
    targets = []
    available_domains_for_dummy = config_loader.get("gapse_settings.scenario_generator.available_domains", ["artificial_general_intelligence", "biotechnology_longevity"]) # Simplified
    for i in range(num_samples):
        tech_factors = [f"Dummy Tech {j} for scn {i}" for j in range(np.random.randint(2, 5))]
        key_events = [f"Dummy Event {j} at year {2030+j*np.random.randint(1,5)}" for j in range(np.random.randint(1, 4))]
        genome = ScenarioGenome(
            id=f"dummy_scn_{i}_{str(uuid.uuid4())[:8]}",
            technological_factors=tech_factors, social_factors=[f"Dummy Social {j}" for j in range(1,3)],
            economic_factors=[f"Dummy Econ {j}" for j in range(1,3)], timeline=f"{2025+i%10}-{2040+i%10+np.random.randint(0,5)}",
            key_events=key_events, domains_focused=np.random.choice(available_domains_for_dummy, np.random.randint(1,min(2, len(available_domains_for_dummy)+1 )), replace=False).tolist()
        )
        genomes.append(genome)
        targets.append(np.clip(0.1 + 0.15*len(tech_factors) - 0.05*len(key_events) + np.random.normal(0,0.1),0.05,0.95))
    return genomes, targets

# --- Main Execution ---
if __name__ == "__main__":
    run_statistics = {
        "config_path": os.getenv("GAPS_CONFIG_PATH", r"F:\TheFutureHumanManifesto\config\config.yaml"),
        "parameters": {
            "NN_INPUT_DIM": NN_INPUT_DIM, "TFIDF_MAX_FEATURES": TFIDF_MAX_FEATURES,
            "HIDDEN_DIM1": HIDDEN_DIM1, "HIDDEN_DIM2": HIDDEN_DIM2, "DROPOUT_RATE": DROPOUT_RATE,
            "LEARNING_RATE": LEARNING_RATE, "NUM_EPOCHS_CONFIG": NUM_EPOCHS, "BATCH_SIZE": BATCH_SIZE,
            "DUMMY_DATA_SIZE_CONFIG": DUMMY_DATA_SIZE
        },
        "data_sources_used": [],
        "data_counts": {},
        "train_val_split": {},
        "tfidf_vocab_size": None,
        "bayesian_model_stats": {},
        "nn_training_stats": {}
    }

    logger.info(f"--- EFFECTIVE CONFIGURATION VALUES USED BY TRAINING SCRIPT (Path: {run_statistics['config_path']}) ---")
    for key, value in run_statistics["parameters"].items():
        logger.info(f"{key}: {value}")
    logger.info(f"HISTORICAL_DATA_PATH: {HISTORICAL_DATA_PATH}")
    logger.info(f"REFINED_DATA_PATH: {REFINED_DATA_PATH}")
    logger.info(f"ORIGINAL_SYNTHETIC_DATA_PATH: {ORIGINAL_SYNTHETIC_DATA_PATH}")
    logger.info("----------------------------------------------------------------------")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VECTORIZER_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BAYESIAN_MODEL_SAVE_PATH), exist_ok=True)
    if REFINED_DATA_PATH and not os.path.exists(os.path.dirname(REFINED_DATA_PATH)): os.makedirs(os.path.dirname(REFINED_DATA_PATH), exist_ok=True)
    if ORIGINAL_SYNTHETIC_DATA_PATH and not os.path.exists(os.path.dirname(ORIGINAL_SYNTHETIC_DATA_PATH)): os.makedirs(os.path.dirname(ORIGINAL_SYNTHETIC_DATA_PATH), exist_ok=True)
    if HISTORICAL_DATA_PATH and not os.path.exists(os.path.dirname(HISTORICAL_DATA_PATH)): os.makedirs(os.path.dirname(HISTORICAL_DATA_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    run_statistics["device"] = str(device)

    all_genomes: List[ScenarioGenome] = []
    all_targets: List[float] = []

    # 1. Load LLM-refined scenarios
    llm_refined_loaded_count = 0
    if os.path.exists(REFINED_DATA_PATH) and os.path.exists(ORIGINAL_SYNTHETIC_DATA_PATH):
        logger.info(f"Attempting to load LLM-refined scenarios from: {REFINED_DATA_PATH} (using {ORIGINAL_SYNTHETIC_DATA_PATH} for lookup)")
        llm_refined_genomes, llm_refined_targets = load_llm_refined_scenarios_with_plausibility(REFINED_DATA_PATH, ORIGINAL_SYNTHETIC_DATA_PATH)
        if llm_refined_genomes:
            llm_refined_loaded_count = len(llm_refined_genomes)
            logger.info(f"Successfully loaded {llm_refined_loaded_count} LLM-refined scenarios.")
            all_genomes.extend(llm_refined_genomes)
            all_targets.extend(llm_refined_targets)
            run_statistics["data_sources_used"].append("LLM-Refined")
        else: logger.warning(f"Could not load LLM-refined scenarios.")
    else: logger.warning(f"Skipping LLM-refined data: one or both files not found (Refined: {REFINED_DATA_PATH}, OriginalSynthetic: {ORIGINAL_SYNTHETIC_DATA_PATH}).")
    run_statistics["data_counts"]["llm_refined_loaded"] = llm_refined_loaded_count

    # 2. Load historical data
    historical_loaded_count = 0
    if not all_genomes or len(all_genomes) < BATCH_SIZE * 2:
        if os.path.exists(HISTORICAL_DATA_PATH):
            logger.info(f"Attempting to load historical data from {HISTORICAL_DATA_PATH}...")
            historical_genomes, historical_targets = load_historical_predictions_data(HISTORICAL_DATA_PATH)
            if historical_genomes:
                historical_loaded_count = len(historical_genomes)
                logger.info(f"Successfully loaded {historical_loaded_count} historical scenarios.")
                all_genomes.extend(historical_genomes)
                all_targets.extend(historical_targets)
                run_statistics["data_sources_used"].append("Historical")
            else: logger.warning(f"Could not load historical scenarios from {HISTORICAL_DATA_PATH}.")
        else: logger.warning(f"Historical data file not found at: {HISTORICAL_DATA_PATH}. Skipping.")
    run_statistics["data_counts"]["historical_loaded"] = historical_loaded_count
    
    # 3. Fallback to dummy data
    dummy_generated_count = 0
    if (not all_genomes or len(all_genomes) < BATCH_SIZE * 2) and DUMMY_DATA_SIZE > 0:
        logger.warning(f"Not enough LLM-refined or historical data (total {len(all_genomes)}). Generating {DUMMY_DATA_SIZE} dummy scenarios.")
        dummy_genomes, dummy_targets = generate_dummy_scenario_data(DUMMY_DATA_SIZE)
        if dummy_genomes:
            dummy_generated_count = len(dummy_genomes)
            all_genomes.extend(dummy_genomes)
            all_targets.extend(dummy_targets)
            run_statistics["data_sources_used"].append("DummyGenerated")
    run_statistics["data_counts"]["dummy_generated"] = dummy_generated_count

    if not all_genomes:
         logger.critical("CRITICAL: Failed to load any data. Exiting.")
         run_statistics["status"] = "Failed - No Data"
         print("\n--- RUN STATISTICS ---")
         print(json.dumps(run_statistics, indent=2))
         exit()
    
    run_statistics["data_counts"]["total_for_training_val"] = len(all_genomes)
    logger.info(f"Total samples available for training/validation: {len(all_genomes)}")
    if len(all_genomes) < BATCH_SIZE:
        logger.warning(f"Total available data ({len(all_genomes)}) is less than BATCH_SIZE ({BATCH_SIZE}).")

    # --- Train/Test Split ---
    # ... (Keep your existing robust train/test split logic) ...
    train_genomes: List[ScenarioGenome] = []
    val_genomes: List[ScenarioGenome] = []
    train_targets: List[float] = []
    val_targets: List[float] = []
    if len(all_genomes) == 1:
        logger.warning("Only one data sample available. Using it for training, no validation set.")
        train_genomes, train_targets = all_genomes, all_targets
    elif len(all_genomes) < BATCH_SIZE * 2 and len(all_genomes) > 1 :
        logger.warning(f"Data ({len(all_genomes)}) is limited for a robust train/val split. Adjusting test_size.")
        test_size_adjusted = 1.0 / len(all_genomes)
        if len(all_genomes) - int(len(all_genomes) * test_size_adjusted) > 0 :
             train_genomes, val_genomes, train_targets, val_targets = train_test_split(all_genomes, all_targets, test_size=test_size_adjusted, random_state=42, stratify=None)
        else:
            train_genomes, train_targets = all_genomes, all_targets
            logger.warning("Using all available data for training, no validation set due to small dataset size.")
    elif len(all_genomes) >= BATCH_SIZE * 2:
        test_size_standard = 0.2
        train_genomes, val_genomes, train_targets, val_targets = train_test_split(all_genomes, all_targets, test_size=test_size_standard, random_state=42, stratify=None)
    else: # Fallback for any edge case not covered, e.g. len(all_genomes) == 0 (though exited above)
        train_genomes, train_targets = all_genomes, all_targets
        logger.error("Unexpected condition in train/test split logic. Using all data for training.")
    run_statistics["train_val_split"]["train_size"] = len(train_genomes)
    run_statistics["train_val_split"]["val_size"] = len(val_genomes)


    # --- TF-IDF, Scaler, Bayesian Ridge Training ---
    if not train_genomes:
        logger.critical("No training genomes available after data loading and splitting. Exiting.")
        run_statistics["status"] = "Failed - No Training Data Post Split"
        print("\n--- RUN STATISTICS ---")
        print(json.dumps(run_statistics, indent=2))
        exit()
        
    train_texts = [" ".join(filter(None, (g.technological_factors or []) + (g.social_factors or []) + (g.economic_factors or []) + (g.key_events or []) + ([g.timeline] if g.timeline else []))) for g in train_genomes]
    if not any(train_texts): logger.warning("All training texts are empty. TF-IDF might not learn effectively.")
    
    tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    try:
        tfidf_vectorizer.fit(train_texts)
        vocab_size = len(tfidf_vectorizer.vocabulary_) if hasattr(tfidf_vectorizer, 'vocabulary_') and tfidf_vectorizer.vocabulary_ else 0
        run_statistics["tfidf_vocab_size"] = vocab_size
        logger.info(f"TfidfVectorizer fitted on training texts with {vocab_size} features.")
        if vocab_size == 0: logger.warning("TfidfVectorizer vocabulary is empty.")
        dump(tfidf_vectorizer, VECTORIZER_SAVE_PATH)
        logger.info(f"TfidfVectorizer saved to {VECTORIZER_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not fit or save TfidfVectorizer: {e}.")
        run_statistics["tfidf_vocab_size"] = "Error"
        # Decide if to exit or proceed with a non-functional vectorizer
        # For now, let's assume it's critical and exit if it fails badly
        if not hasattr(tfidf_vectorizer, 'vocabulary_') or not tfidf_vectorizer.vocabulary_:
            logger.critical("TF-IDF Vectorizer fitting failed critically. Exiting.")
            run_statistics["status"] = "Failed - TF-IDF Error"
            print("\n--- RUN STATISTICS ---")
            print(json.dumps(run_statistics, indent=2))
            exit()


    train_features_np_list = []
    valid_train_indices = []
    for idx, g in enumerate(train_genomes):
        try:
            features = extract_features_for_dataset(g, tfidf_vectorizer, NN_INPUT_DIM)
            train_features_np_list.append(features)
            valid_train_indices.append(idx)
        except Exception as e: logger.error(f"Error extracting features for training genome ID {getattr(g, 'id', 'N/A')}: {e}. Skipping.")
    
    if not train_features_np_list:
        logger.critical("No features extracted from training genomes. Exiting.")
        run_statistics["status"] = "Failed - Feature Extraction Error"
        print("\n--- RUN STATISTICS ---")
        print(json.dumps(run_statistics, indent=2))
        exit()
    
    train_targets_filtered = [train_targets[i] for i in valid_train_indices]
    train_features_np = np.array(train_features_np_list, dtype=np.float32)
    run_statistics["train_val_split"]["train_size_post_feature_extraction"] = len(train_features_np)


    scaler = StandardScaler()
    scaler.fit(train_features_np)
    logger.info("Feature scaler fitted on training data.")
    dump(scaler, SCALER_SAVE_PATH)
    logger.info(f"Feature scaler saved to {SCALER_SAVE_PATH}")

    if train_features_np.size > 0 and len(train_targets_filtered) > 0:
        logger.info("Training BayesianRidge model...")
        bayesian_ridge_model = BayesianRidge(max_iter=500, tol=1e-3, fit_intercept=True, compute_score=True, verbose=False) # Set verbose=False if not needed
        train_features_scaled_for_bayes = scaler.transform(train_features_np)
        train_targets_np = np.array(train_targets_filtered, dtype=np.float32)
        bayesian_ridge_model.fit(train_features_scaled_for_bayes, train_targets_np.ravel())
        logger.info(f"BayesianRidge model trained. Coefs (first 3): {bayesian_ridge_model.coef_[:3]}..., Intercept: {bayesian_ridge_model.intercept_}")
        dump(bayesian_ridge_model, BAYESIAN_MODEL_SAVE_PATH)
        logger.info(f"BayesianRidge model saved to {BAYESIAN_MODEL_SAVE_PATH}")
        run_statistics["bayesian_model_stats"] = {"intercept": bayesian_ridge_model.intercept_, "coef_preview": list(bayesian_ridge_model.coef_[:3])}
    else:
        logger.warning("Skipping BayesianRidge model training.")
        run_statistics["bayesian_model_stats"] = "Skipped"

    train_genomes_for_nn = [train_genomes[i] for i in valid_train_indices]
    if not train_genomes_for_nn:
        logger.critical("No training data for NN DataLoaders after filtering. Exiting.")
        run_statistics["status"] = "Failed - No NN Training Data"
        print("\n--- RUN STATISTICS ---")
        print(json.dumps(run_statistics, indent=2))
        exit()

    train_dataset = ScenarioDataset(train_genomes_for_nn, train_targets_filtered, tfidf_vectorizer, scaler, NN_INPUT_DIM)
    if len(train_dataset) == 0:
        logger.critical("Train dataset is empty. Exiting.")
        run_statistics["status"] = "Failed - Empty Train Dataset"
        print("\n--- RUN STATISTICS ---")
        print(json.dumps(run_statistics, indent=2))
        exit()
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
    
    val_loader = None
    val_dataset_size = 0
    if val_genomes and val_targets:
        val_genomes_for_nn = []
        val_targets_for_nn = []
        valid_val_indices = []
        for idx, g_val in enumerate(val_genomes):
            try:
                features_val = extract_features_for_dataset(g_val, tfidf_vectorizer, NN_INPUT_DIM)
                # No need to store features_val here, ScenarioDataset will handle it
                valid_val_indices.append(idx)
            except Exception as e: logger.error(f"Error extracting features for validation genome ID {getattr(g_val, 'id', 'N/A')}: {e}. Skipping.")
        
        val_genomes_for_nn = [val_genomes[i] for i in valid_val_indices]
        val_targets_for_nn = [val_targets[i] for i in valid_val_indices]

        if val_genomes_for_nn and val_targets_for_nn:
            val_dataset = ScenarioDataset(val_genomes_for_nn, val_targets_for_nn, tfidf_vectorizer, scaler, NN_INPUT_DIM)
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, len(val_dataset)), shuffle=False)
                val_dataset_size = len(val_dataset)
        else: logger.warning("No valid validation genomes/targets after feature extraction.")
    run_statistics["train_val_split"]["train_dataloader_size"] = len(train_dataset)
    run_statistics["train_val_split"]["val_dataloader_size"] = val_dataset_size
    logger.info(f"Created DataLoaders. Train size: {len(train_dataset)}, Val size: {val_dataset_size}")

    model = ProbabilisticNN(input_dim=NN_INPUT_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2, dropout_rate=DROPOUT_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = NLLLossGaussian()
    logger.info("NN Model, optimizer, and criterion initialized.")

    if len(train_loader) > 0:
        # MODIFIED: train_model now returns stats
        patience_val = training_config.get("early_stopping_patience", 15)
        min_delta_val = training_config.get("early_stopping_min_delta", 1e-4)
        run_statistics["parameters"]["EARLY_STOPPING_PATIENCE"] = patience_val
        run_statistics["parameters"]["EARLY_STOPPING_MIN_DELTA"] = min_delta_val

        nn_stats = train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device, patience=patience_val, min_delta=min_delta_val)
        run_statistics["nn_training_stats"] = nn_stats
        try:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Trained NN model saved to {MODEL_SAVE_PATH}")
            run_statistics["status"] = "Success - NN Model Saved"
        except Exception as e:
            logger.error(f"Could not save trained NN model: {e}")
            run_statistics["status"] = "Success - NN Training Completed (Save Failed)"
    else:
        logger.warning("No training data for NN model after all processing. Skipping NN training.")
        run_statistics["status"] = "Skipped NN Training - No Data"

    logger.info("--- Training script finished ---")
    
    print("\n--- FINAL RUN STATISTICS ---")
    print(json.dumps(run_statistics, indent=2, default=lambda o: str(o) if isinstance(o, (np.ndarray, np.generic)) else o))