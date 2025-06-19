import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
import uuid
import json
import os

# Assuming ScenarioGenome is defined in utils.models
try:
    from src.utils.models import ScenarioGenome, Domain as DomainEnum # Assuming Domain is an Enum
except ImportError:
    # Fallback dataclass for ScenarioGenome if not found (ensure fields match)
    @dataclass
    class ScenarioGenome:
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        technological_factors: List[str] = field(default_factory=list)
        social_factors: List[str] = field(default_factory=list)
        economic_factors: List[str] = field(default_factory=list)
        timeline: str = "2025-2050"
        key_events: List[str] = field(default_factory=list)
        domains_focused: List[str] = field(default_factory=list) # String list
        # Optional: if your Pydantic model has these, include them
        # domains: Optional[List[Any]] = field(default_factory=list) # For Enum version
        # time_since_prediction_years: Optional[float] = None
        probability_weights: Dict[str, float] = field(default_factory=dict)
        fitness_score: Optional[float] = None
        generation: int = 0
        parent_ids: List[str] = field(default_factory=list)

    class DomainEnum: # Dummy if not imported
        AGI = "artificial_general_intelligence"
        LONGEVITY = "biotechnology_longevity"
        # ... add others if needed for dummy ...

def get_domain_value(domain):
    return domain.value if hasattr(domain, 'value') else domain

from config_loader import ConfigLoader
from custom_logging import get_logger
logger = get_logger("synthetic_scenario_simulator")

# --- Configuration for Simulation Variables and Events ---

# These would ideally come from a more detailed config or be more dynamic
SIMULATION_VARIABLES = {
    "AGI_Capability": {"min": 0, "max": 10, "initial_avg": 1.0, "initial_std": 0.5},
    "Biotech_Longevity_Maturity": {"min": 0, "max": 10, "initial_avg": 2.0, "initial_std": 0.5},
    "BCI_Integration_Level": {"min": 0, "max": 10, "initial_avg": 0.5, "initial_std": 0.2},
    "Nanotech_Manufacturing_Scale": {"min": 0, "max": 10, "initial_avg": 0.5, "initial_std": 0.2},
    "Quantum_Computing_Impact": {"min": 0, "max": 10, "initial_avg": 0.2, "initial_std": 0.1},
    "Public_Acceptance_RadicalTech": {"min": 0, "max": 1, "initial_avg": 0.4, "initial_std": 0.1},
    "Global_Collaboration_Index": {"min": 0, "max": 1, "initial_avg": 0.5, "initial_std": 0.1},
    "Environmental_Stability_Index": {"min": 0, "max": 1, "initial_avg": 0.6, "initial_std": 0.1},
    "Funding_FutureTech_Level": {"min": 0, "max": 1, "initial_avg": 0.5, "initial_std": 0.2} # 0-1 scale representing relative funding
}

# Define events that can be triggered when variables cross thresholds
# (threshold, event_description_template, associated_factors_template, domains_involved)
SIMULATION_EVENTS_THRESHOLDS = {
    "AGI_Capability": [
        (5, "Significant AI breakthrough: Near-AGI capabilities demonstrated in multiple narrow domains.", 
         ["Advanced AI algorithms deployed", "AI surpasses human performance in specific complex tasks"], 
         ["artificial_general_intelligence", "future_of_work_education"]),
        (8, "True AGI Achieved: AI matches or exceeds human general intelligence across most cognitive tasks.", 
         ["Recursive self-improvement in AI observed", "Emergence of novel AI-driven scientific discoveries"], 
         ["artificial_general_intelligence", "global_governance"]),
        (9.5, "ASI Emergence: Artificial Superintelligence capabilities rapidly develop, posing existential questions.",
         ["Unforeseen emergent AI behaviors", "Global debate on AI control and alignment intensifies"],
         ["artificial_general_intelligence", "global_risk_opportunity_nexus"]) # Example of using a new domain
    ],
    "Biotech_Longevity_Maturity": [
        (6, "Major Longevity Breakthrough: Therapies extend healthy human lifespan by an average of 10-15 years.", 
         ["Rejuvenation technologies become clinically available", "Societal debates on access to life extension"], 
         ["biotechnology_longevity", "social_change"]), # Assuming social_change is a valid domain string
        (9, "Longevity Escape Velocity (LEV) Considered Achievable: Continuous advancements promise indefinite healthspan.", 
         ["Aging effectively 'cured' for those with access", "Radical societal restructuring due to agelessness"], 
         ["biotechnology_longevity", "economic_paradigm_shifts"]) # Example
    ],
    "BCI_Integration_Level": [
        (4, "Advanced BCIs for Therapeutic Use: Widespread adoption for restoring sensory/motor functions.",
         ["Neural prosthetics common", "Improved quality of life for disabled populations"],
         ["brain_computer_interfaces", "healthcare"]),
        (7, "BCIs for Cognitive Enhancement: Direct neural links for learning and communication become popular among early adopters.",
         ["Cognitive augmentation markets emerge", "Ethical concerns about 'neuro-divide'"],
         ["brain_computer_interfaces", "neurophilosophy", "education_reform"])
    ],
    # Add more for other variables...
    "Environmental_Stability_Index": [ # Example of a decreasing variable triggering an event
        (0.3, "Major Climate Tipping Point Reached: Irreversible environmental damage accelerates, causing global crises.",
         ["Mass climate migration", "Food and water security severely impacted", "Increased geopolitical instability"],
         ["climate_tech_adaptation", "global_governance", "resource_management"])
    ]
}


@dataclass
class SimulationState:
    variables: Dict[str, float]
    year: int
    key_events_log: List[str] = field(default_factory=list)
    triggered_event_flags: Dict[str, bool] = field(default_factory=dict) # To ensure events trigger once

    def get_variable(self, name: str) -> float:
        return self.variables.get(name, 0.0)

    def set_variable(self, name: str, value: float):
        min_val = SIMULATION_VARIABLES.get(name, {}).get("min", -float('inf'))
        max_val = SIMULATION_VARIABLES.get(name, {}).get("max", float('inf'))
        self.variables[name] = np.clip(value, min_val, max_val)

class SyntheticScenarioSimulator:
    def __init__(self, config_loader_instance: ConfigLoader):
        self.config_loader = config_loader_instance
        self.sim_vars_config = SIMULATION_VARIABLES # Could be loaded from config
        self.event_thresholds = SIMULATION_EVENTS_THRESHOLDS # Could be loaded

        # Define inter-variable influences (factor_effecting, factor_affected, strength, delay_years_avg, delay_std)
        # Strength: positive = enhances, negative = inhibits
        # This is a simplified model; a real one might use more complex functions or a graph.
        self.influences = [
            ("Funding_FutureTech_Level", "AGI_Capability", 0.2, 1, 0.5),
            ("Funding_FutureTech_Level", "Biotech_Longevity_Maturity", 0.15, 2, 1),
            ("Funding_FutureTech_Level", "BCI_Integration_Level", 0.1, 2, 1),
            ("Funding_FutureTech_Level", "Nanotech_Manufacturing_Scale", 0.1, 3, 1),
            ("Funding_FutureTech_Level", "Quantum_Computing_Impact", 0.05, 4, 2),
            ("AGI_Capability", "Biotech_Longevity_Maturity", 0.3, 1, 0.5), # AGI accelerates biotech
            ("AGI_Capability", "Quantum_Computing_Impact", 0.1, 2, 1),   # AGI helps solve quantum problems
            ("AGI_Capability", "Environmental_Stability_Index", -0.05, 3, 1), # Uncontrolled AGI could be bad, or AGI for climate solutions could be positive (this needs nuance)
            ("Public_Acceptance_RadicalTech", "AGI_Capability", 0.05, 0, 0), # Higher acceptance might mean less friction
            ("Public_Acceptance_RadicalTech", "Biotech_Longevity_Maturity", 0.1, 0, 0),
            ("Global_Collaboration_Index", "Funding_FutureTech_Level", 0.1, 0, 0), # More collaboration, more funding
            ("Environmental_Stability_Index", "Public_Acceptance_RadicalTech", -0.1, 1, 0.5) # Environmental stress might reduce risk appetite
        ]
        # Store influences that are currently "in flight" due to delays
        self.pending_influences: List[Tuple[int, str, float]] = [] # (target_year, affected_variable, effect_value)


    def _initialize_simulation_state(self, start_year: int) -> SimulationState:
        variables = {}
        for var_name, conf in self.sim_vars_config.items():
            initial_value = np.random.normal(conf["initial_avg"], conf["initial_std"])
            variables[var_name] = np.clip(initial_value, conf["min"], conf["max"])
        return SimulationState(variables=variables, year=start_year)

    def _apply_stochastic_drift_and_trends(self, state: SimulationState):
        # Each variable has some inherent tendency to change or random drift
        for var_name in state.variables.keys():
            current_val = state.get_variable(var_name)
            # Example: AGI and Biotech tend to increase, Environmental Stability might decrease
            if var_name in ["AGI_Capability", "Biotech_Longevity_Maturity", "BCI_Integration_Level", "Nanotech_Manufacturing_Scale"]:
                drift = np.random.normal(0.05, 0.02) # Small positive drift
            elif var_name == "Environmental_Stability_Index":
                drift = np.random.normal(-0.01, 0.01) # Small negative drift
            else:
                drift = np.random.normal(0, 0.01) # Minor random fluctuation
            
            state.set_variable(var_name, current_val + drift)

    def _apply_influences(self, state: SimulationState):
        # Apply immediate influences
        new_effects_this_year = {}

        for effecting_var, affected_var, strength, delay_avg, delay_std in self.influences:
            effecting_val_norm = (state.get_variable(effecting_var) - self.sim_vars_config[effecting_var]["min"]) / \
                                 (self.sim_vars_config[effecting_var]["max"] - self.sim_vars_config[effecting_var]["min"]) # Normalize 0-1
            
            # Non-linear response (e.g., sigmoid) could be better than linear strength
            # For simplicity, using linear strength * normalized value
            effect_magnitude = strength * effecting_val_norm * np.random.normal(1.0, 0.1) # Add some noise to strength

            if delay_avg == 0: # Immediate effect
                current_affected_val = new_effects_this_year.get(affected_var, state.get_variable(affected_var))
                new_effects_this_year[affected_var] = current_affected_val + effect_magnitude
            else: # Delayed effect
                delay = max(1, int(np.random.normal(delay_avg, delay_std)))
                target_year = state.year + delay
                self.pending_influences.append((target_year, affected_var, effect_magnitude))
        
        for var, val in new_effects_this_year.items():
            state.set_variable(var, val)

        # Apply matured pending influences
        remaining_pending = []
        for target_year, affected_var, effect_value in self.pending_influences:
            if state.year >= target_year:
                current_val = state.get_variable(affected_var)
                state.set_variable(affected_var, current_val + effect_value)
                logger.debug(f"Applied pending influence: {affected_var} += {effect_value:.3f} in year {state.year}")
            else:
                remaining_pending.append((target_year, affected_var, effect_value))
        self.pending_influences = remaining_pending


    def _check_and_trigger_events(self, state: SimulationState):
        for var_name, event_list in self.event_thresholds.items():
            current_val = state.get_variable(var_name)
            for threshold, desc_template, factors_template, domains in event_list:
                event_key = f"{var_name}_{threshold}" # Unique key for this event
                has_decreasing_threshold_logic = var_name == "Environmental_Stability_Index" # Example

                triggered_condition = (current_val >= threshold and not has_decreasing_threshold_logic) or \
                                      (current_val <= threshold and has_decreasing_threshold_logic)

                if triggered_condition and not state.triggered_event_flags.get(event_key):
                    event_description = f"Year {state.year}: {desc_template}"
                    state.key_events_log.append(event_description)
                    state.triggered_event_flags[event_key] = True
                    logger.debug(f"Event triggered: {event_description}")
                    # Potentially, events themselves could have immediate impacts on other variables
                    # e.g., AGI breakthrough sharply increases Funding_FutureTech_Level
                    if "AGI Achieved" in desc_template:
                        state.set_variable("Funding_FutureTech_Level", min(1.0, state.get_variable("Funding_FutureTech_Level") + 0.3))
                        state.set_variable("Public_Acceptance_RadicalTech", min(1.0, state.get_variable("Public_Acceptance_RadicalTech") + 0.2))


    def run_single_simulation(self, start_year: int, end_year: int) -> Tuple[SimulationState, List[Dict[str, Any]]]:
        state = self._initialize_simulation_state(start_year)
        history = [] # To store state at each year
        self.pending_influences = [] # Reset for each simulation

        for year in range(start_year, end_year + 1):
            state.year = year
            current_state_snapshot = {
                "year": year,
                "variables": state.variables.copy()
            }
            
            self._apply_stochastic_drift_and_trends(state)
            self._apply_influences(state)
            self._check_and_trigger_events(state)
            
            current_state_snapshot["key_events_this_year"] = [e for e in state.key_events_log if f"Year {year}" in e]
            history.append(current_state_snapshot)
            
        return state, history

    def _simulation_to_genome(self, final_state: SimulationState, history: List[Dict[str, Any]], start_year: int, end_year: int) -> ScenarioGenome:
        # Derive factors and domains from the simulation trajectory
        tech_factors = []
        social_factors = []
        economic_factors = [] # Needs more sophisticated mapping from variables
        domains_focused_set = set()

        # Analyze variable changes and event triggers to populate factors
        initial_vars = history[0]['variables']
        final_vars = final_state.variables

        for var_name, final_val in final_vars.items():
            initial_val = initial_vars.get(var_name, 0)
            change = final_val - initial_val
            # This mapping is very heuristic and needs refinement
            if "AGI_Capability" == var_name and final_val > 7:
                tech_factors.append("Advanced AGI development trajectory")
                domains_focused_set.add(get_domain_value(DomainEnum.AGI))
            elif "Biotech_Longevity_Maturity" == var_name and final_val > 6:
                tech_factors.append("Significant progress in longevity biotech")
                domains_focused_set.add(get_domain_value(DomainEnum.LONGEVITY))
            # ... more mappings ...
            if "Public_Acceptance_RadicalTech" == var_name:
                if final_val > 0.7: social_factors.append("High public acceptance of new technologies")
                elif final_val < 0.3: social_factors.append("Strong public resistance to radical tech")
            if "Funding_FutureTech_Level" == var_name and final_val > 0.7:
                economic_factors.append("High levels of investment in future technologies")


        # Add factors from triggered events
        for event_log_entry in final_state.key_events_log:
            # Try to find the original event template to get associated factors and domains
            for var_name_key, event_list_val in self.event_thresholds.items():
                for _thresh, _desc_template, factors_template_list, event_domains_list in event_list_val:
                    if _desc_template in event_log_entry: # Simple check
                        tech_factors.extend(factors_template_list)
                        domains_focused_set.update(event_domains_list)
                        break
        
        # Deduplicate
        tech_factors = list(set(tech_factors))
        social_factors = list(set(social_factors))
        economic_factors = list(set(economic_factors))


        return ScenarioGenome(
            id=str(uuid.uuid4()),
            technological_factors=tech_factors[:5], # Limit for brevity
            social_factors=social_factors[:3],
            economic_factors=economic_factors[:3],
            timeline=f"{start_year}-{end_year}",
            key_events=final_state.key_events_log,
            domains_focused=list(domains_focused_set)[:3], # Limit
            generation=-2 # Mark as synthetic simulation
        )

    def _assign_synthetic_probability(self, final_state: SimulationState, history: List[Dict[str, Any]]) -> float:
        # Heuristic probability based on final state and trajectory
        # This is highly subjective and needs careful design
        prob = 0.5 # Base
        
        # Penalize extreme outcomes or instability
        if final_state.get_variable("Environmental_Stability_Index") < 0.2:
            prob *= 0.5
        if final_state.get_variable("AGI_Capability") > 9.5 and final_state.get_variable("Global_Collaboration_Index") < 0.3: # ASI without collaboration
            prob *= 0.3
            
        # Reward "balanced" or "successful" outcomes (define what these mean)
        if final_state.get_variable("Biotech_Longevity_Maturity") > 8 and final_state.get_variable("Public_Acceptance_RadicalTech") > 0.7:
            prob *= 1.2
            
        # Penalize if too many critical events happened very quickly
        if len(final_state.key_events_log) > 5 and (history[-1]['year'] - history[0]['year']) < 15 :
            prob *= 0.7

        # Normalize by number of "high impact" events - more complex scenarios might be less likely
        num_major_events = sum(1 for event_key, triggered in final_state.triggered_event_flags.items() if triggered and any(t[0] >= 7 for v, t_list in self.event_thresholds.items() for t in t_list if f"{v}_{t[0]}" == event_key)) # Events with high thresholds
        prob *= (0.9 ** num_major_events)

        return np.clip(prob + np.random.normal(0, 0.05), 0.01, 0.99) # Add noise and clip


    def generate_synthetic_dataset(self, num_scenarios: int, start_year: int = 2025, sim_duration_years: int = 30) -> List[Tuple[ScenarioGenome, float]]:
        dataset = []
        logger.info(f"Generating {num_scenarios} synthetic scenarios via simulation...")
        for i in range(num_scenarios):
            if i % (num_scenarios // 10 if num_scenarios >=10 else 1) == 0:
                logger.info(f"Generated {i}/{num_scenarios} synthetic scenarios...")
            
            final_state, history = self.run_single_simulation(start_year, start_year + sim_duration_years -1)
            genome = self._simulation_to_genome(final_state, history, start_year, start_year + sim_duration_years -1)
            probability = self._assign_synthetic_probability(final_state, history)
            dataset.append((genome, probability))
        
        logger.info(f"Finished generating {len(dataset)} synthetic scenarios.")
        return dataset

# --- Main Execution for Generating Synthetic Data ---
if __name__ == "__main__":
    config = ConfigLoader() # Uses default path "config/config.yaml" or GAPS_CONFIG_PATH
    
    num_synthetic_scenarios_to_generate = config.get("gapse_settings.training.synthetic_data_size", 500) # Add to config
    output_file = config.get("gapse_settings.training.synthetic_data_output_path", "data/synthetic_scenarios_generated.json") # Add to config

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    simulator = SyntheticScenarioSimulator(config_loader_instance=config)
    synthetic_dataset = simulator.generate_synthetic_dataset(num_synthetic_scenarios_to_generate, sim_duration_years=25)

    # Save the dataset
    output_data_list = []
    for genome, prob in synthetic_dataset:
        # Convert genome to dict, add target_probability
        # This assumes ScenarioGenome is a dataclass or has a __dict__ method
        try:
            genome_dict = genome.__dict__.copy() # For dataclass
        except AttributeError: # If Pydantic model
            genome_dict = genome.model_dump().copy() 

        genome_dict["target_probability_synthetic"] = prob
        output_data_list.append(genome_dict)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data_list, f, indent=2)
        logger.info(f"Successfully saved {len(output_data_list)} synthetic scenarios to {output_file}")
    except Exception as e:
        logger.error(f"Error saving synthetic dataset: {e}")

    # Example of how to load this data in train_probabilistic_nn.py
    # def load_synthetic_generated_data(json_file_path: str) -> Tuple[List[ScenarioGenome], List[float]]:
    #     genomes = []
    #     targets = []
    #     with open(json_file_path, 'r') as f: data_list = json.load(f)
    #     for item_dict in data_list:
    #         target_prob = item_dict.pop("target_probability_synthetic")
    #         # Ensure all fields for ScenarioGenome are present or have defaults
    #         genomes.append(ScenarioGenome(**item_dict)) 
    #         targets.append(float(target_prob))
    #     return genomes, targets