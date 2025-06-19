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
from logging import Logger

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
    domain_diversity = len(set(scenario.domains_focused)) if scenario.domains_focused else 0
    
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

# --- Dummy Data Generation (Replace with your actual data loading) ---
def generate_dummy_scenario_data(num_samples: int) -> Tuple[List[ScenarioGenome], List[float]]:
    """Generates dummy ScenarioGenomes and target probabilities."""
    genomes = []
    targets = [] # "True" probabilities for these dummy scenarios
    domains = [
        "artificial_general_intelligence",
        "biotechnology_longevity",
        "brain_computer_interfaces",
        "nanotechnology",
        "quantum_computing",
        "space_colonization",
        "genetic_engineering"
    ]
    for i in range(num_samples):
        tech_factors = [f"Tech factor {j} for scenario {i}" for j in range(np.random.randint(2, 5))]
        key_events = [f"Key event {j} at year {2030+j*2}" for j in range(np.random.randint(1, 4))]
        genome = ScenarioGenome(
            id=f"dummy_scn_{i}",
            technological_factors=tech_factors,
            social_factors=[f"Social factor {j}" for j in range(1,3)],
            economic_factors=[f"Economic factor {j}" for j in range(1,3)],
            timeline=f"{2025+i%10}-{2040+i%10}",
            key_events=key_events,
            domains=np.random.choice(domains, np.random.randint(1,3), replace=False).tolist()
        )
        genomes.append(genome)
        # Simulate target probability based on some features (e.g., number of tech factors)
        # This is highly artificial; real targets would come from historical data or expert assessment.
        target_prob = np.clip(0.1 + 0.15 * len(tech_factors) - 0.05 * len(key_events) + np.random.normal(0, 0.1), 0.05, 0.95)
        targets.append(target_prob)
    return genomes, targets


# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VECTORIZER_SAVE_PATH), exist_ok=True) # For TfidfVectorizer
    os.makedirs(os.path.dirname(BAYESIAN_MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Generating {DUMMY_DATA_SIZE} dummy scenario data samples for training...")
    all_genomes, all_targets = generate_dummy_scenario_data(DUMMY_DATA_SIZE)
    logger.info("Dummy data generation complete.")

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
    bayesian_ridge_model = BayesianRidge(n_iter=300, tol=1e-3, fit_intercept=True, compute_score=True)
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