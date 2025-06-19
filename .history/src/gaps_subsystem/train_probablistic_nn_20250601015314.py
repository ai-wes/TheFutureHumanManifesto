# src/gapse_subsystem/train_probabilistic_nn.py
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Any
import os
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
from pydantic import BaseModel, Field
class ScenarioGenome(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    technological_factors: List[str] = Field(default_factory=list)
    social_factors: List[str] = Field(default_factory=list)
    economic_factors: List[str] = Field(default_factory=list)
    timeline: Optional[str] = "2025-2050"
    key_events: List[str] = Field(default_factory=list)
    domains_focused: List[str] = Field(default_factory=list) # Changed from List[Domain] to List[str] for simplicity here
    probability_weights: Dict[str, float] = Field(default_factory=dict)
    fitness_score: Optional[float] = None
    generation: int = 0
    parent_ids: List[str] = Field(default_factory=list)
    time_since_prediction_years: Optional[float] = None # Added from historical loader

# ProbabilisticNN should be defined in hybrid_probabilistic_forecaster
try:
    from hybrid_probabilistic_forecaster import ProbabilisticNN
except ImportError:
    logger.error("Failed to import ProbabilisticNN from hybrid_probabilistic_forecaster. Ensure it's defined there.")
    # Define a placeholder if import fails, to allow script to run further for structure check
    class ProbabilisticNN(nn.Module):
        def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1) # Dummy
        def forward(self, x): return torch.sigmoid(self.fc(x)), torch.ones_like(self.fc(x)) * 0.1


config_loader = ConfigLoader(config_path=os.getenv("GAPS_CONFIG_PATH", r"F:\TheFutureHumanManifesto\config\config.yaml"))

# --- Configuration from YAML or defaults ---
gapse_settings = config_loader.get("gapse_settings", {})
forecaster_config = gapse_settings.get("forecaster", {})
training_config = gapse_settings.get("training", {})

print(type(gapse_settings), type(forecaster_config), type(training_config))

MODEL_SAVE_PATH = gapse_settings['forecaster']['model_save_path']
SCALER_SAVE_PATH = gapse_settings['forecaster']['scaler_save_path']
VECTORIZER_SAVE_PATH = gapse_settings['forecaster']['vectorizer_save_path']
BAYESIAN_MODEL_SAVE_PATH = gapse_settings['forecaster']['bayesian_model_save_path']

NN_INPUT_DIM = gapse_settings['forecaster']['nn_input_dim']
TFIDF_MAX_FEATURES = gapse_settings['forecaster']['tfidf_max_features']
HIDDEN_DIM1 = gapse_settings['forecaster']['nn_hidden1']
HIDDEN_DIM2 = gapse_settings['forecaster']['nn_hidden2']
DROPOUT_RATE = gapse_settings['forecaster']['nn_dropout']

LEARNING_RATE = gapse_settings['training']['learning_rate']
NUM_EPOCHS = gapse_settings['training']['num_epochs']
BATCH_SIZE = gapse_settings['training']['batch_size']
DUMMY_DATA_SIZE = gapse_settings['training']['dummy_data_size'] # Fallback if no other data

# Paths for the new data sources
HISTORICAL_DATA_PATH = gapse_settings['training']['historical_data_path']
REFINED_DATA_PATH = gapse_settings['training']['refined_data_with_plausibility_path']
ORIGINAL_SYNTHETIC_DATA_PATH = gapse_settings['training']['original_synthetic_scenarios_path']


# --- Feature Extractor --- (Keep as is)
def extract_features_for_dataset(scenario: ScenarioGenome, vectorizer: TfidfVectorizer, max_total_features: int) -> np.ndarray:
    text_parts = (scenario.technological_factors or []) + \
                 (scenario.social_factors or []) + \
                 (scenario.economic_factors or []) + \
                 (scenario.key_events or []) + \
                 ([scenario.timeline] if scenario.timeline else [])
    full_text = " ".join(filter(None, text_parts))
    
    text_features_sparse = vectorizer.transform([full_text if full_text else ""])
    text_features = text_features_sparse.toarray().flatten()

    if len(text_features) < TFIDF_MAX_FEATURES:
        text_features = np.pad(text_features, (0, TFIDF_MAX_FEATURES - len(text_features)), 'constant', constant_values=0.0)
    elif len(text_features) > TFIDF_MAX_FEATURES:
        text_features = text_features[:TFIDF_MAX_FEATURES]

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


# --- Dataset Class --- (Keep as is)
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
    def __len__(self): return len(self.genomes)
    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

# --- Loss Function --- (Keep as is)
class NLLLossGaussian(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, mean_pred: torch.Tensor, var_pred: torch.Tensor, target: torch.Tensor):
        var_pred_stable = torch.clamp(var_pred, min=1e-6)
        log_var = torch.log(var_pred_stable)
        sq_error = (target - mean_pred).pow(2)
        loss = 0.5 * (log_var + sq_error / var_pred_stable + np.log(2 * np.pi))
        return loss.mean()

# --- Training Function --- (Keep as is)
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
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
            log_msg += f", Val Loss: {avg_val_loss:.6f}"
        else: log_msg += ", No validation data."
        logger.info(log_msg)
    logger.info("Training finished.")

# --- Data Loading and Preprocessing Functions ---
# (Keep outcome_to_target_probability and ensure_list as they are used by historical loader)
def outcome_to_target_probability(outcome_str: Optional[str], prediction_date_str: Optional[str], 
                                  actual_outcome_date_str: Optional[str], timeline_str: Optional[str]) -> float:
    # ... (implementation from your script) ...
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
    if isinstance(val, list): return val
    elif isinstance(val, str): return [val]
    elif val is None: return []
    else: return list(val) if hasattr(val, '__iter__') else [val]

def load_historical_predictions_data(json_file_path: str) -> Tuple[List[ScenarioGenome], List[float]]:
    # ... (Keep your existing implementation) ...
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

# --- NEW: Function to convert refined brief to ScenarioGenome ---
def brief_to_genome(
    refined_brief_dict: Dict[str, Any], # This is the value of "refined_executive_brief"
    original_id: str,
    original_timeline: Optional[str],
    original_domains: Optional[List[str]]
) -> ScenarioGenome:
    """
    Transforms a refined executive brief dictionary into a ScenarioGenome object.
    """
    tech_factors = [
        f"{item.get('driver', '')}: {item.get('implication', '')}".strip(": ")
        for item in refined_brief_dict.get("core_technological_drivers", [])
    ]
    social_factors = [
        f"{item.get('dynamic', '')}: {item.get('implication', '')}".strip(": ")
        for item in refined_brief_dict.get("defining_social_dynamics", [])
    ]
    economic_factors = [
        f"{item.get('transformation', '')}: {item.get('implication', '')}".strip(": ")
        for item in refined_brief_dict.get("key_economic_transformations", [])
    ]
    
    key_events = refined_brief_dict.get("core_narrative_turning_points", [])
    
    # Incorporate strategic overview and challenge into textual factors
    # Option: Add to social_factors, or could be key_events if phrased as such
    strategic_overview = refined_brief_dict.get("strategic_coherence_overview", "")
    defining_challenge = refined_brief_dict.get("defining_strategic_challenge", "") # Or defining_challenge_overview
    
    # For simplicity, appending to social_factors. Could also be a new field or part of key_events.
    if strategic_overview:
        social_factors.append(f"Strategic Overview: {strategic_overview}")
    if defining_challenge:
        social_factors.append(f"Defining Challenge: {defining_challenge}")

    # Use original timeline and domains if available, otherwise make a best guess or default
    timeline_to_use = original_timeline
    if not timeline_to_use and key_events: # Try to infer from turning points if original_timeline is missing
        first_event_year_match = re.search(r'\b(20\d{2})\b', key_events[0])
        last_event_year_match = re.search(r'\b(20\d{2})\b', key_events[-1])
        if first_event_year_match and last_event_year_match:
            timeline_to_use = f"{first_event_year_match.group(1)}-{last_event_year_match.group(1)}"
        elif first_event_year_match: # If only first event has year, assume a 20 year span
             start_y = int(first_event_year_match.group(1))
             timeline_to_use = f"{start_y}-{start_y+20}"
        else:
            timeline_to_use = "2030-2050" # Fallback
    elif not timeline_to_use:
        timeline_to_use = "2030-2050"


    return ScenarioGenome(
        id=f"refined_{original_id}", # Create a new ID
        technological_factors=tech_factors,
        social_factors=social_factors,
        economic_factors=economic_factors,
        timeline=timeline_to_use,
        key_events=key_events,
        domains_focused=original_domains or [], # Use original domains, or empty list
        generation=-10, # Special generation number for LLM-refined data
        # Other fields like parent_ids, fitness_score can be defaults
    )

# --- NEW: Function to load LLM-refined scenarios with their plausibility scores ---
def load_llm_refined_scenarios_with_plausibility(
    plausibility_json_path: str,
    original_synthetic_scenarios_path: str # Path to the file like synthetic_scenarios_generated.json
) -> Tuple[List[ScenarioGenome], List[float]]:
    genomes: List[ScenarioGenome] = []
    targets: List[float] = []

    if not os.path.exists(plausibility_json_path):
        logger.error(f"LLM-refined plausibility data file not found: {plausibility_json_path}")
        return genomes, targets
    if not os.path.exists(original_synthetic_scenarios_path):
        logger.warning(f"Original synthetic scenarios file not found at {original_synthetic_scenarios_path}. Cannot fetch original timelines/domains.")
        original_scenarios_map = {}
    else:
        try:
            with open(original_synthetic_scenarios_path, 'r', encoding='utf-8') as f_orig:
                original_data_list = json.load(f_orig)
            original_scenarios_map = {
                item["id"]: {
                    "timeline": item.get("timeline", "2025-2050"),
                    "domains_focused": item.get("domains_focused", [])
                } for item in original_data_list
            }
        except Exception as e:
            logger.error(f"Failed to load or process original synthetic data from {original_synthetic_scenarios_path}: {e}")
            original_scenarios_map = {} # Proceed without it if loading fails

    try:
        with open(plausibility_json_path, "r", encoding="utf-8") as f:
            refined_briefs_with_plausibility_list = json.load(f)

        for item_with_plausibility in refined_briefs_with_plausibility_list:
            original_id = item_with_plausibility.get("original_scenario_id")
            refined_brief_content = item_with_plausibility.get("refined_executive_brief")
            
            target_prob = item_with_plausibility.get("llm_assigned_plausibility")

            if original_id is None or refined_brief_content is None:
                logger.warning(f"Skipping item due to missing original_id or refined_executive_brief: {str(item_with_plausibility)[:200]}")
                continue
            if target_prob is None:
                logger.warning(f"Scenario {original_id} missing LLM-assigned plausibility. Skipping.")
                continue

            original_data = original_scenarios_map.get(original_id, {})
            original_timeline = original_data.get("timeline")
            original_domains = original_data.get("domains_focused")

            genome = brief_to_genome(
                refined_brief_content,
                original_id,
                original_timeline,
                original_domains
            )
            genomes.append(genome)
            targets.append(float(target_prob))
        
        logger.info(f"Loaded {len(genomes)} LLM-refined scenarios with plausibility scores from {plausibility_json_path}")

    except FileNotFoundError: # Should be caught by the os.path.exists check already
        pass # Already logged
    except json.JSONDecodeError:
        logger.error(f"Error: Plausibility data file {plausibility_json_path} is not valid JSON.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading LLM-refined data: {e}")
        
    return genomes, targets


# --- Fallback Dummy Data Generator --- (Keep as is)
def generate_dummy_scenario_data(num_samples: int) -> Tuple[List[ScenarioGenome], List[float]]:
    # ... (Keep your existing implementation) ...
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
    # --- Configuration & Setup --- (Keep as is)
    logger.info(f"--- EFFECTIVE CONFIGURATION VALUES USED BY TRAINING SCRIPT (Path: {config_loader.config_path}) ---")
    logger.info(f"REFINED_DATA_PATH (for LLM-refined scenarios): {REFINED_DATA_PATH}")
    logger.info(f"ORIGINAL_SYNTHETIC_DATA_PATH (for timeline/domains lookup): {ORIGINAL_SYNTHETIC_DATA_PATH}")
    # ... (print other configs) ...

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    # ... (mkdirs for other paths) ...
    if REFINED_DATA_PATH and not os.path.exists(os.path.dirname(REFINED_DATA_PATH)):
        os.makedirs(os.path.dirname(REFINED_DATA_PATH), exist_ok=True)
    if ORIGINAL_SYNTHETIC_DATA_PATH and not os.path.exists(os.path.dirname(ORIGINAL_SYNTHETIC_DATA_PATH)):
         os.makedirs(os.path.dirname(ORIGINAL_SYNTHETIC_DATA_PATH), exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading Priority: LLM-Refined -> Historical -> Dummy ---
    all_genomes: List[ScenarioGenome] = []
    all_targets: List[float] = []

    # 1. Try loading LLM-refined scenarios with plausibility scores
    logger.info(f"Attempting to load LLM-refined scenarios from: {REFINED_DATA_PATH}")
    logger.info(f" (Will use original synthetic data from: {ORIGINAL_SYNTHETIC_DATA_PATH} for timeline/domain lookup)")
    
    # Create dummy files if they don't exist, for testing the loading logic
    if not os.path.exists(REFINED_DATA_PATH):
        logger.warning(f"{REFINED_DATA_PATH} not found. Creating a dummy file for testing.")
        # Create a dummy refined_briefs_with_plausibility.json
        # This dummy data should align with the structure your load_llm_refined_scenarios_with_plausibility expects
        dummy_refined_plausibility_data = [
            {
                "original_scenario_id": "dummy_synthetic_1", # Needs to match an ID in dummy original synthetic data
                "original_scenario_data_preview": {"timeline": "2030-2040"},
                "refined_executive_brief": {
                    "core_narrative_turning_points": ["Dummy Refined TP 1"],
                    "core_technological_drivers": [{"driver": "Dummy Refined Driver 1", "implication": "Dummy Implication 1"}],
                    "defining_social_dynamics": [{"dynamic": "Dummy Refined Dynamic 1", "implication": "Dummy Implication 1"}],
                    "key_economic_transformations": [{"transformation": "Dummy Refined Transformation 1", "implication": "Dummy Implication 1"}],
                    "strategic_coherence_overview": "Dummy refined overview.",
                    "defining_strategic_challenge": "Dummy refined challenge."
                },
                "llm_assigned_plausibility": 0.75,
                "llm_plausibility_reasoning": "Plausible dummy."
            }
        ]
        with open(REFINED_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(dummy_refined_plausibility_data, f, indent=2)

    if not os.path.exists(ORIGINAL_SYNTHETIC_DATA_PATH):
        logger.warning(f"{ORIGINAL_SYNTHETIC_DATA_PATH} not found. Creating a dummy file for testing.")
        # Create a dummy synthetic_scenarios_generated.json
        dummy_original_synthetic_data = [
            {
                "id": "dummy_synthetic_1",
                "timeline": "2030-2040",
                "domains_focused": ["artificial_general_intelligence"],
                "technological_factors": ["Dummy Tech Factor"],
                "social_factors": ["Dummy Social Factor"],
                "economic_factors": ["Dummy Econ Factor"],
                "key_events": ["Dummy Original Event"],
                "target_probability_synthetic": 0.6 # This won't be used if llm_assigned_plausibility is present
            }
        ]
        with open(ORIGINAL_SYNTHETIC_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(dummy_original_synthetic_data, f, indent=2)

    llm_refined_genomes, llm_refined_targets = load_llm_refined_scenarios_with_plausibility(
        REFINED_DATA_PATH,
        ORIGINAL_SYNTHETIC_DATA_PATH
    )
    if llm_refined_genomes:
        logger.info(f"Successfully loaded {len(llm_refined_genomes)} LLM-refined scenarios.")
        all_genomes.extend(llm_refined_genomes)
        all_targets.extend(llm_refined_targets)
    else:
        logger.warning(f"Could not load LLM-refined scenarios from {REFINED_DATA_PATH}.")

    # 2. If not enough LLM-refined data, try loading historical data
    if not all_genomes or len(all_genomes) < BATCH_SIZE * 2:
        logger.info(f"Not enough LLM-refined data (found {len(all_genomes)}). Attempting to load historical data from {HISTORICAL_DATA_PATH}...")
        # Create dummy historical data if file doesn't exist
        if not os.path.exists(HISTORICAL_DATA_PATH):
            logger.warning(f"{HISTORICAL_DATA_PATH} not found. Creating a dummy file for testing.")
            dummy_historical_data = [
                {
                    "id": "dummy_historical_1", "timeline": "2020-2030", "key_events": ["Past event A"],
                    "technological_factors": ["Old tech"], "social_factors": ["Past trend"], "economic_factors": ["Old econ"],
                    "domains_focused": ["space_colonization"],
                    "metadata": {"actual_outcome": "realized as predicted", "prediction_date": "2019-01-01", "actual_outcome_date": "2025-01-01"}
                }
            ]
            with open(HISTORICAL_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(dummy_historical_data, f, indent=2)

        historical_genomes, historical_targets = load_historical_predictions_data(HISTORICAL_DATA_PATH)
        if historical_genomes:
            logger.info(f"Successfully loaded {len(historical_genomes)} historical scenarios.")
            all_genomes.extend(historical_genomes)
            all_targets.extend(historical_targets)
        else:
            logger.warning(f"Could not load historical scenarios from {HISTORICAL_DATA_PATH}.")

    # 3. If still not enough data, fall back to dummy data generator
    if not all_genomes or len(all_genomes) < BATCH_SIZE * 2:
        logger.warning(f"Not enough usable LLM-refined or historical data (total {len(all_genomes)}). Falling back to generating dummy data.")
        dummy_genomes, dummy_targets = generate_dummy_scenario_data(DUMMY_DATA_SIZE)
        if dummy_genomes:
            all_genomes.extend(dummy_genomes)
            all_targets.extend(dummy_targets)
        if not all_genomes: # Should only happen if dummy generation also fails
             logger.critical("Failed to load any data (LLM-refined, historical, or dummy). Exiting.")
             exit()
    
    logger.info(f"Total samples for training/validation: {len(all_genomes)}")

    # --- Train/Test Split --- (Keep as is)
    test_size = 0.2
    if len(all_genomes) * test_size < 1 and len(all_genomes) > 1: test_size = 1.0 / len(all_genomes)
    elif len(all_genomes) <=1: test_size = 0.0
    
    if test_size > 0 and len(all_genomes) > 1 : # Ensure there's enough data to split
        train_genomes, val_genomes, train_targets, val_targets = train_test_split(
            all_genomes, all_targets, test_size=test_size, random_state=42, stratify=None
        )
    else:
        train_genomes, train_targets = all_genomes, all_targets
        val_genomes, val_targets = [], []
        logger.warning("Too few samples for a validation set or only one sample. Training on all available data.")

    # --- TF-IDF, Scaler, Bayesian Ridge Training --- (Keep as is, but ensure train_texts is not empty)
    if not train_genomes:
        logger.critical("No training genomes available after data loading and splitting. Exiting.")
        exit()
        
    train_texts = [" ".join(filter(None, (g.technological_factors or []) + 
                                     (g.social_factors or []) + 
                                     (g.economic_factors or []) + 
                                     (g.key_events or []) + 
                                     ([g.timeline] if g.timeline else [])))
                   for g in train_genomes]
    
    if not any(train_texts): # Check if all texts are empty strings
        logger.warning("All training texts are empty. TF-IDF might not learn effectively.")
        # Handle this case, e.g., by creating a dummy vocabulary or skipping TF-IDF if appropriate
        # For now, we'll let it proceed, TfidfVectorizer can handle empty docs but won't be useful.
    
    tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    try:
        tfidf_vectorizer.fit(train_texts) # Fit on potentially richer text
        logger.info(f"TfidfVectorizer fitted on training texts with {len(tfidf_vectorizer.vocabulary_ if hasattr(tfidf_vectorizer, 'vocabulary_') and tfidf_vectorizer.vocabulary_ else [])} features.")
        dump(tfidf_vectorizer, VECTORIZER_SAVE_PATH)
        logger.info(f"TfidfVectorizer saved to {VECTORIZER_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Could not fit or save TfidfVectorizer: {e}")
        # Fallback: load a pre-existing one or use a dummy? For now, error out or proceed with caution.
        # If you proceed, vectorizer might be None, which extract_features_for_dataset needs to handle.
        # For simplicity, we'll assume it either fits or we might have issues downstream.

    train_features_np_list = []
    for g in train_genomes:
        try:
            features = extract_features_for_dataset(g, tfidf_vectorizer, NN_INPUT_DIM)
            train_features_np_list.append(features)
        except Exception as e:
            logger.error(f"Error extracting features for genome ID {g.id}: {e}. Skipping this genome for scaler/Bayesian training.")
    
    if not train_features_np_list:
        logger.critical("No features could be extracted from training genomes. Cannot train scaler or Bayesian model. Exiting.")
        exit()
    train_features_np = np.array(train_features_np_list, dtype=np.float32)


    scaler = StandardScaler()
    scaler.fit(train_features_np)
    logger.info("Feature scaler fitted on training data.")
    dump(scaler, SCALER_SAVE_PATH)
    logger.info(f"Feature scaler saved to {SCALER_SAVE_PATH}")

    logger.info("Training BayesianRidge model...")
    bayesian_ridge_model = BayesianRidge(max_iter=500, tol=1e-3, fit_intercept=True, compute_score=True)
    train_features_scaled_for_bayes = scaler.transform(train_features_np)
    train_targets_np = np.array(train_targets[:len(train_features_np)], dtype=np.float32) # Ensure targets match features if some genomes were skipped
    
    if len(train_features_scaled_for_bayes) != len(train_targets_np):
        logger.error(f"Mismatch between number of features ({len(train_features_scaled_for_bayes)}) and targets ({len(train_targets_np)}) for Bayesian model. This should not happen.")
        # Potentially exit or handle this error
    elif len(train_targets_np) == 0:
        logger.error("No targets available for Bayesian model training.")
    else:
        bayesian_ridge_model.fit(train_features_scaled_for_bayes, train_targets_np.ravel())
        logger.info(f"BayesianRidge model trained. Coefs (first 5): {bayesian_ridge_model.coef_[:5]}..., Intercept: {bayesian_ridge_model.intercept_}")
        dump(bayesian_ridge_model, BAYESIAN_MODEL_SAVE_PATH)
        logger.info(f"BayesianRidge model saved to {BAYESIAN_MODEL_SAVE_PATH}")


    # --- NN DataLoaders and Training ---
    # Ensure train_genomes and train_targets used for Dataset are consistent with those used for scaler/Bayesian
    # If some genomes were skipped during feature extraction, they should also be skipped for NN Dataset
    # For simplicity, we'll assume train_genomes and train_targets are already filtered if necessary,
    # or that extract_features_for_dataset handles errors gracefully for ScenarioDataset.
    
    # Re-filter train_genomes and train_targets if some were skipped during feature extraction for scaler
    # This is a bit complex if IDs are not perfectly aligned. A safer way is to re-extract features inside Dataset
    # or ensure extract_features_for_dataset is robust and always returns a valid array.
    # The current ScenarioDataset re-extracts, which is fine.

    if not train_genomes: # Check again if train_genomes became empty
        logger.critical("No training data available for NN DataLoaders. Exiting.")
        exit()

    train_dataset = ScenarioDataset(train_genomes, train_targets, tfidf_vectorizer, scaler, NN_INPUT_DIM)
    if len(train_dataset) == 0:
        logger.critical("Train dataset is empty. Cannot create DataLoader. Exiting.")
        exit()
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
    
    val_loader = None
    if val_genomes and val_targets: # val_genomes might be empty if test_size was 0
        val_dataset = ScenarioDataset(val_genomes, val_targets, tfidf_vectorizer, scaler, NN_INPUT_DIM)
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, len(val_dataset)), shuffle=False)
    
    logger.info(f"Created DataLoaders. Train size: {len(train_dataset)}, Val size: {len(val_dataset) if val_loader else 0}")

    model = ProbabilisticNN(
        input_dim=NN_INPUT_DIM,
        hidden_dim1=HIDDEN_DIM1,
        hidden_dim2=HIDDEN_DIM2,
        dropout_rate=DROPOUT_RATE
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = NLLLossGaussian()
    logger.info("NN Model, optimizer, and criterion initialized.")

    if len(train_loader) > 0:
        train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device)
        try:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Trained NN model saved to {MODEL_SAVE_PATH}")
        except Exception as e:
            logger.error(f"Could not save trained NN model: {e}")
    else:
        logger.warning("No training data for NN model. Skipping NN training.")

    logger.info("--- Training script finished ---")