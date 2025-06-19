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

# In src/gapse_subsystem/synthetic_scenario_simulator.py

# ... (imports and existing dataclasses/classes like ScenarioGenome, SimulationState, ConfigLoader, Logger) ...

# --- Expanded Configuration for Simulation Variables, Events, and Influences ---

SIMULATION_VARIABLES = {
    # Core Technological Capabilities (0-10 scale, 10 = highly advanced/transformative)
    "AGI_Capability":               {"min": 0, "max": 10, "initial_avg": 1.0, "initial_std": 0.5, "inherent_growth_factor": 0.05},
    "Biotech_Longevity_Maturity":   {"min": 0, "max": 10, "initial_avg": 2.0, "initial_std": 0.5, "inherent_growth_factor": 0.04},
    "BCI_Integration_Level":        {"min": 0, "max": 10, "initial_avg": 0.5, "initial_std": 0.2, "inherent_growth_factor": 0.03},
    "Nanotech_Manufacturing_Scale": {"min": 0, "max": 10, "initial_avg": 0.5, "initial_std": 0.2, "inherent_growth_factor": 0.03},
    "Quantum_Computing_Impact":     {"min": 0, "max": 10, "initial_avg": 0.2, "initial_std": 0.1, "inherent_growth_factor": 0.02},
    "Genetic_Engineering_Precision":{"min": 0, "max": 10, "initial_avg": 3.0, "initial_std": 0.6, "inherent_growth_factor": 0.04}, # Precision & Safety
    "Space_Colonization_Tech":      {"min": 0, "max": 10, "initial_avg": 0.3, "initial_std": 0.1, "inherent_growth_factor": 0.01}, # Launch cost reduction, ISRU, habitats
    "Renewable_Energy_Dominance":   {"min": 0, "max": 10, "initial_avg": 4.0, "initial_std": 1.0, "inherent_growth_factor": 0.06}, # 10 = 100% renewable grid
    "Robotics_Automation_Sophistication": {"min": 0, "max": 10, "initial_avg": 3.0, "initial_std": 0.5, "inherent_growth_factor": 0.05},

    # Societal Factors (0-1 scale, 1 = high acceptance/collaboration/stability)
    "Public_Acceptance_RadicalTech":{"min": 0, "max": 1, "initial_avg": 0.4, "initial_std": 0.15, "volatility": 0.05},
    "Global_Collaboration_Index":   {"min": 0, "max": 1, "initial_avg": 0.5, "initial_std": 0.15, "volatility": 0.03}, # Geopolitical stability & cooperation
    "Wealth_Inequality_Index":      {"min": 0, "max": 1, "initial_avg": 0.6, "initial_std": 0.1, "volatility": 0.02}, # 0 = perfect equality, 1 = max inequality
    "Trust_In_Institutions":      {"min": 0, "max": 1, "initial_avg": 0.4, "initial_std": 0.1, "volatility": 0.04},
    "Educational_Adaptability":     {"min": 0, "max": 1, "initial_avg": 0.3, "initial_std": 0.1, "volatility": 0.02}, # Speed of reskilling workforce

    # Economic & Resource Factors
    "Funding_FutureTech_Level":     {"min": 0, "max": 1, "initial_avg": 0.5, "initial_std": 0.2, "volatility": 0.05}, # Relative global R&D funding
    "Resource_Scarcity_Index":      {"min": 0, "max": 1, "initial_avg": 0.3, "initial_std": 0.1, "volatility": 0.03}, # 1 = extreme scarcity
    "Global_Economic_Stability":    {"min": 0, "max": 1, "initial_avg": 0.6, "initial_std": 0.15, "volatility": 0.05}, # 1 = very stable

    # Environmental Factors
    "Environmental_Stability_Index":{"min": 0, "max": 1, "initial_avg": 0.6, "initial_std": 0.1, "inherent_decline_factor": -0.01}, # Tends to decline without intervention
    "Climate_Mitigation_Effort":    {"min": 0, "max": 1, "initial_avg": 0.3, "initial_std": 0.1, "volatility": 0.04} # Global effort level
}

SIMULATION_EVENTS_THRESHOLDS = {
    "AGI_Capability": [
        (5, "Significant AI Progress: Near-AGI capabilities (e.g., advanced multimodal reasoning, complex problem-solving) widely demonstrated.",
         ["AI tools boost scientific research output significantly", "Debates on AI job displacement become mainstream"],
         ["artificial_general_intelligence", "future_of_work_education", "AI_ethics"]),
        (8, "True AGI Emergence: AI achieves human-level general intelligence, capable of learning and reasoning across diverse domains autonomously.",
         ["Recursive self-improvement cycles begin in AI", "Global AI Alignment and Safety Summit convened urgently"],
         ["artificial_general_intelligence", "global_governance", "AI_safety"]),
        (9.5, "ASI Development: Artificial Superintelligence capabilities rapidly surpass human intellect, leading to unpredictable global transformations.",
         ["Humanity grapples with loss of cognitive dominance", "Fundamental questions about human purpose and future arise"],
         ["artificial_general_intelligence", "existential_risk", "transhumanism", "neurophilosophy"])
    ],
    "Biotech_Longevity_Maturity": [
        (4, "Advanced Rejuvenation Therapies: First generation of systemic rejuvenation treatments (e.g., advanced senolytics, partial epigenetic reprogramming) show success in human trials.",
         ["Healthspan extension by 5-10 years common in trial participants", "High cost and limited access raise equity concerns"],
         ["biotechnology_longevity", "healthcare_economics", "social_inequality"]),
        (7, "Significant Lifespan Extension: Multiple integrated therapies routinely extend healthy human lifespan by 20-30 years.",
         ["Retirement ages significantly rethought", "Multi-generational societal structures become more complex"],
         ["biotechnology_longevity", "social_structures", "economic_planning"]),
        (9.2, "Longevity Escape Velocity (LEV) Achieved: Therapeutic advancements consistently outpace aging, promising indefinite healthspan for those with access.",
         ["Concept of 'biological death' from aging becomes largely obsolete", "Profound philosophical and societal shifts regarding human limits"],
         ["biotechnology_longevity", "transhumanism", "resource_management", "global_ethics"])
    ],
    "BCI_Integration_Level": [
        (3, "High-Fidelity Therapeutic BCIs: BCIs reliably restore complex motor and communication functions for patients.",
         ["Brain-controlled prosthetics offer near-natural movement", "Direct speech synthesis from thought becomes viable"],
         ["brain_computer_interfaces", "assistive_technology", "medical_ethics"]),
        (6, "Mainstream Augmentative BCIs: BCIs for cognitive enhancement (memory, learning speed, direct data access) become available and sought after by a segment of the population.",
         ["'Neuro-enhanced' individuals gain competitive advantages", "Concerns about mental privacy and cognitive liberty grow"],
         ["brain_computer_interfaces", "neurophilosophy", "education_reform", "social_stratification"]),
        (8.5, "Seamless Brain-to-Brain/Cloud Communication: Direct neural data exchange between individuals and with AI/cloud systems becomes possible.",
         ["Collective intelligence networks form", "Redefinition of individuality and consciousness", "Potential for direct thought manipulation or hacking"],
         ["brain_computer_interfaces", "decentralized_systems_web3", "consciousness_studies", "cybersecurity"])
    ],
    "Nanotech_Manufacturing_Scale": [
        (4, "Advanced Nanomaterials Common: Custom-designed nanomaterials are widely used in manufacturing, medicine, and energy.",
         ["Ultra-strong lightweight materials", "Highly efficient catalysts and sensors", "Targeted drug delivery systems improve"],
         ["nanotechnology", "materials_science", "advanced_manufacturing"]),
        (7, "Early Molecular Assemblers Demonstrated: Lab-scale devices show capability for atomically precise construction of simple structures.",
         ["Debate on 'grey goo' and self-replication risks intensifies", "Potential for radical resource abundance emerges"],
         ["nanotechnology", "molecular_manufacturing", "existential_risk_mitigation"]),
        (9, "Widespread Desktop Molecular Manufacturing: Affordable personal fabricators allow on-demand creation of complex products, disrupting global supply chains.",
         ["Transition to post-scarcity material economy begins", "Intellectual property and regulation of designs become critical issues"],
         ["nanotechnology", "economic_paradigm_shifts", "intellectual_property", "global_trade"])
    ],
    "Quantum_Computing_Impact": [
        (4, "Quantum Supremacy for Niche Problems: Quantum computers solve specific problems intractable for classical supercomputers (e.g., complex material simulation, specific optimizations).",
         ["Accelerated discovery in chemistry and materials science", "New quantum algorithms developed"],
         ["quantum_computing", "computational_science", "drug_discovery"]),
        (7, "Broad Quantum Advantage: Quantum computers provide significant speedups for a wide range of commercially and scientifically relevant problems.",
         ["Current public-key cryptography standards become vulnerable", "AI model training and optimization dramatically enhanced"],
         ["quantum_computing", "cybersecurity", "artificial_general_intelligence", "financial_modeling"]),
        (9, "Fault-Tolerant Universal Quantum Computers: Large-scale, error-corrected quantum computers become available, revolutionizing computation.",
         ["Simulation of complex quantum systems (e.g., human brain, universe) becomes feasible", "New scientific paradigms emerge"],
         ["quantum_computing", "fundamental_physics", "consciousness_studies"])
    ],
    "Genetic_Engineering_Precision": [
        (6, "Somatic Gene Therapy Mainstream: Safe and effective gene therapies for a wide range of genetic diseases are common.",
         ["Many inherited diseases largely eradicated", "Personalized genetic medicine becomes standard"],
         ["genetic_engineering", "personalized_medicine", "bioethics"]),
        (8, "Germline Gene Editing Debated and Piloted: Limited, highly regulated germline modifications for disease prevention are attempted in some regions.",
         ["Intense global ethical debate on heritable genetic changes", "Fears of 'designer babies' and eugenics resurface"],
         ["genetic_engineering", "bioethics", "human_enhancement", "international_law"]),
        (9.5, "Precise Human Genetic Enhancement Possible: Reliable methods for complex trait enhancement (e.g., intelligence, physical abilities) become technically feasible.",
         ["Societal divergence based on genetic status", "Philosophical questions about human nature and evolution intensify"],
         ["genetic_engineering", "transhumanism", "social_justice", "bioethics"])
    ],
    "Space_Colonization_Tech": [
        (4, "Sustainable Lunar Presence: A permanent, continuously inhabited international lunar base is established for research and resource prospecting.",
         ["ISRU (In-Situ Resource Utilization) for water/oxygen demonstrated on Moon", "Regular cislunar transport established"],
         ["space_colonization", "lunar_economy", "resource_management"]),
        (7, "First Mars Outpost: Small, crewed research outpost established on Mars, reliant on Earth for resupply.",
         ["Challenges of long-duration space travel and Martian habitats addressed", "Search for Martian life intensifies"],
         ["space_colonization", "planetary_science", "human_physiology_space"]),
        (9, "Self-Sufficient Mars Colony: Mars colony achieves a high degree of self-sufficiency, capable of local food production and manufacturing.",
         ["First Martian-born humans", "Unique Martian society and governance begin to form"],
         ["space_colonization", "terraforming_concepts", "off_world_civilization"])
    ],
    "Renewable_Energy_Dominance": [
        (7, "Global Tipping Point for Renewables: Renewable energy sources (solar, wind, storage) become the dominant and most cost-effective form of new electricity generation globally.",
         ["Fossil fuel industry enters structural decline", "Significant reductions in global carbon emissions from power sector"],
         ["renewable_energy", "climate_change_mitigation", "energy_transition"]),
        (9.5, "Near-Total Decarbonization of Energy: Global energy systems are overwhelmingly powered by renewables and other zero-carbon sources (e.g., advanced nuclear, green hydrogen).",
         ["Atmospheric CO2 levels begin to stabilize or decline", "Energy abundance enables new technological possibilities (e.g., large-scale desalination)"],
         ["renewable_energy", "decarbonization", "sustainable_development", "geoengineering_alternatives"])
    ],
    "Robotics_Automation_Sophistication": [
        (6, "Ubiquitous Advanced Robotics: Highly capable robots and automated systems are common in manufacturing, logistics, services, and homes.",
         ["Significant labor displacement in manual and routine cognitive tasks", "Productivity surges in automated sectors"],
         ["robotics_automation", "future_of_work_education", "universal_basic_income_debates"]),
        (8.5, "Humanoid General-Purpose Robots: Dexterous, mobile humanoid robots capable of performing a wide variety_of_tasks in human environments become available.",
         ["Robots as personal assistants, caregivers, and co-workers", "Redefinition of many human job roles"],
         ["robotics_automation", "human_robot_interaction", "social_robotics", "elder_care_tech"])
    ],
    # Societal Factor Events (example: can be triggered by themselves or by tech)
    "Wealth_Inequality_Index": [ # Higher value = more inequality
        (0.8, "Extreme Wealth Concentration Crisis: Global wealth inequality reaches critical levels, causing widespread social unrest and political instability.",
         ["Calls for radical wealth redistribution", "Breakdown of social cohesion in some regions"],
         ["social_inequality", "political_instability", "economic_reform"])
    ],
    "Environmental_Stability_Index": [ # Lower value = worse environment
        (0.3, "Major Climate Tipping Point Reached: Irreversible environmental damage accelerates, causing global crises.",
         ["Mass climate migration", "Food and water security severely impacted", "Increased geopolitical instability"],
         ["climate_tech_adaptation", "global_governance", "resource_management", "disaster_relief"])
    ],
    "Global_Collaboration_Index": [ # Lower value = less collaboration
        (0.2, "Resurgence of Extreme Nationalism & Isolationism: Global collaboration collapses, leading to trade wars, arms races, and stalled progress on global challenges.",
         ["Breakdown of international institutions", "Increased risk of major conflicts"],
         ["geopolitics", "international_relations", "global_risk"])
    ]
}

# Expanded Influence Network
# (Effecting_Var, Affected_Var, Strength, Delay_Yrs_Avg, Delay_Yrs_Std, Condition_Var_Optional, Condition_Threshold_Optional, Condition_Above_Optional)
# Condition: Influence only applies if Condition_Var is above/below Condition_Threshold
# Strength can be positive (enhances) or negative (inhibits)
# Delay is in years.
INFLUENCES = [
    # Funding Impacts
    ("Funding_FutureTech_Level", "AGI_Capability", 0.25, 1, 0.5),
    ("Funding_FutureTech_Level", "Biotech_Longevity_Maturity", 0.20, 2, 1),
    ("Funding_FutureTech_Level", "BCI_Integration_Level", 0.15, 2, 1),
    ("Funding_FutureTech_Level", "Nanotech_Manufacturing_Scale", 0.15, 3, 1),
    ("Funding_FutureTech_Level", "Quantum_Computing_Impact", 0.10, 4, 2),
    ("Funding_FutureTech_Level", "Genetic_Engineering_Precision", 0.18, 2, 1),
    ("Funding_FutureTech_Level", "Space_Colonization_Tech", 0.08, 5, 2),
    ("Funding_FutureTech_Level", "Renewable_Energy_Dominance", 0.15, 2, 1), # Funding for green tech

    # AGI Impacts
    ("AGI_Capability", "Biotech_Longevity_Maturity", 0.35, 1, 0.5, "AGI_Capability", 4, True), # Stronger effect once AGI is somewhat capable
    ("AGI_Capability", "Quantum_Computing_Impact", 0.20, 2, 1, "AGI_Capability", 5, True),
    ("AGI_Capability", "Nanotech_Manufacturing_Scale", 0.25, 1, 0.5, "AGI_Capability", 6, True),
    ("AGI_Capability", "Genetic_Engineering_Precision", 0.30, 1, 0.5, "AGI_Capability", 5, True),
    ("AGI_Capability", "Robotics_Automation_Sophistication", 0.40, 0, 0.5, "AGI_Capability", 3, True),
    ("AGI_Capability", "Wealth_Inequality_Index", 0.1, 3, 1, "AGI_Capability", 7, True), # Advanced AGI could exacerbate inequality if not managed
    ("AGI_Capability", "Global_Economic_Stability", -0.1, 4, 1, "AGI_Capability", 8, True), # Disruptive potential
    ("AGI_Capability", "Resource_Scarcity_Index", -0.15, 5, 2, "AGI_Capability", 8.5, True), # AGI might solve resource problems

    # Biotech/Longevity Impacts
    ("Biotech_Longevity_Maturity", "Public_Acceptance_RadicalTech", 0.1, 1, 0.5, "Biotech_Longevity_Maturity", 5, True), # Success breeds acceptance
    ("Biotech_Longevity_Maturity", "Wealth_Inequality_Index", 0.05, 5, 2, "Biotech_Longevity_Maturity", 7, True), # If expensive, increases inequality

    # BCI Impacts
    ("BCI_Integration_Level", "Educational_Adaptability", 0.15, 2, 1, "BCI_Integration_Level", 5, True), # Enhanced learning
    ("BCI_Integration_Level", "Public_Acceptance_RadicalTech", 0.1, 1, 1, "BCI_Integration_Level", 4, True),

    # Nanotech Impacts
    ("Nanotech_Manufacturing_Scale", "Resource_Scarcity_Index", -0.2, 3, 1, "Nanotech_Manufacturing_Scale", 6, True), # Molecular manufacturing reduces scarcity
    ("Nanotech_Manufacturing_Scale", "Environmental_Stability_Index", 0.1, 4, 2, "Nanotech_Manufacturing_Scale", 5, True), # Potential for environmental cleanup tech

    # Quantum Impacts
    ("Quantum_Computing_Impact", "AGI_Capability", 0.1, 3, 1, "Quantum_Computing_Impact", 5, True), # QC aids AI research
    ("Quantum_Computing_Impact", "Funding_FutureTech_Level", 0.05, 1, 0.5, "Quantum_Computing_Impact", 6, True), # Breakthroughs attract funding

    # Societal Factor Impacts
    ("Public_Acceptance_RadicalTech", "Funding_FutureTech_Level", 0.15, 0, 0), # High acceptance -> more funding
    ("Public_Acceptance_RadicalTech", "AGI_Capability", 0.05, 1, 0.5), # Less friction for development
    ("Global_Collaboration_Index", "Funding_FutureTech_Level", 0.2, 0, 0),
    ("Global_Collaboration_Index", "Climate_Mitigation_Effort", 0.25, 0, 0),
    ("Wealth_Inequality_Index", "Trust_In_Institutions", -0.15, 2, 1), # High inequality erodes trust
    ("Wealth_Inequality_Index", "Global_Collaboration_Index", -0.1, 3, 1),
    ("Trust_In_Institutions", "Public_Acceptance_RadicalTech", 0.1, 1, 0.5),
    ("Educational_Adaptability", "Robotics_Automation_Sophistication", -0.05, 2, 1), # If workforce can't adapt, slows adoption of very advanced automation

    # Environmental Impacts
    ("Environmental_Stability_Index", "Public_Acceptance_RadicalTech", -0.1, 1, 0.5), # Crisis might make people wary of more tech
    ("Environmental_Stability_Index", "Global_Economic_Stability", -0.2, 0, 0.5), # Environmental crises destabilize economy
    ("Environmental_Stability_Index", "Resource_Scarcity_Index", 0.15, 0, 0.5), # Poor environment -> more scarcity
    ("Climate_Mitigation_Effort", "Environmental_Stability_Index", 0.1, 3, 1), # Mitigation efforts improve stability (with delay)
    ("Climate_Mitigation_Effort", "Renewable_Energy_Dominance", 0.15, 2, 1),

    # Feedback loops
    ("Robotics_Automation_Sophistication", "Wealth_Inequality_Index", 0.08, 3, 1, "Robotics_Automation_Sophistication", 6, True), # High automation can increase inequality
    ("Renewable_Energy_Dominance", "Environmental_Stability_Index", 0.15, 2, 1, "Renewable_Energy_Dominance", 7, True), # High renewables improve env.
]

# Ensure the SyntheticScenarioSimulator class uses these expanded definitions
class SyntheticScenarioSimulator:
    def __init__(self, config_loader_instance: ConfigLoader):
        self.config_loader = config_loader_instance
        self.sim_vars_config = SIMULATION_VARIABLES # Use expanded
        self.event_thresholds = SIMULATION_EVENTS_THRESHOLDS # Use expanded
        self.influences = INFLUENCES # Use expanded
        self.pending_influences: List[Tuple[int, str, float]] = []
        logger.info(f"SyntheticScenarioSimulator initialized with {len(self.sim_vars_config)} variables, {sum(len(v) for v in self.event_thresholds.values())} event thresholds, and {len(self.influences)} influence rules.")


    def _initialize_simulation_state(self, start_year: int) -> SimulationState:
        variables = {}
        for var_name, conf in self.sim_vars_config.items():
            initial_value = np.random.normal(conf["initial_avg"], conf["initial_std"])
            variables[var_name] = np.clip(initial_value, conf["min"], conf["max"])
        return SimulationState(variables=variables, year=start_year, key_events_log=[], triggered_event_flags={})

    def _apply_stochastic_drift_and_trends(self, state: SimulationState):
        for var_name in state.variables.keys():
            current_val = state.get_variable(var_name)
            conf = self.sim_vars_config[var_name]
            
            drift = 0.0
            if "inherent_growth_factor" in conf:
                drift += conf["inherent_growth_factor"] * np.random.uniform(0.5, 1.5) # Growth with some randomness
            if "inherent_decline_factor" in conf: # For things that tend to degrade
                drift += conf["inherent_decline_factor"] * np.random.uniform(0.5, 1.5)
            if "volatility" in conf: # General random fluctuation
                drift += np.random.normal(0, conf["volatility"])
            
            state.set_variable(var_name, current_val + drift)


    def _apply_influences(self, state: SimulationState):
        new_effects_this_year: Dict[str, float] = {var: state.get_variable(var) for var in state.variables} # Start with current values

        for influence_rule in self.influences:
            effecting_var, affected_var, strength, delay_avg, delay_std = influence_rule[:5]
            condition_var, condition_thresh, condition_above = (influence_rule[5:8] + [None]*3)[:3] if len(influence_rule) > 5 else (None, None, None)

            # Check condition if present
            if condition_var:
                cond_val = state.get_variable(condition_var)
                if condition_above and cond_val < condition_thresh:
                    continue # Condition not met
                if not condition_above and cond_val > condition_thresh:
                    continue # Condition not met
            
            effecting_val = state.get_variable(effecting_var)
            # Normalize effecting_val to 0-1 if it's not already (e.g. capability scores)
            # For simplicity, assume 0-10 scales are normalized by /10, 0-1 scales used directly
            norm_effecting_val = effecting_val / 10.0 if self.sim_vars_config[effecting_var]["max"] == 10 else effecting_val
            
            effect_magnitude = strength * norm_effecting_val * np.random.normal(1.0, 0.15) # More noise in effect

            if delay_avg == 0:
                new_effects_this_year[affected_var] = new_effects_this_year.get(affected_var, state.get_variable(affected_var)) + effect_magnitude
            else:
                delay = max(1, int(np.random.normal(delay_avg, delay_std)))
                target_year = state.year + delay
                self.pending_influences.append((target_year, affected_var, effect_magnitude))
        
        # Apply aggregated immediate effects
        for var, val in new_effects_this_year.items():
            state.set_variable(var, val)

        # Apply matured pending influences
        remaining_pending = []
        updated_by_pending: Dict[str, float] = {}
        for target_year, affected_var, effect_value in self.pending_influences:
            if state.year >= target_year:
                current_val_for_pending = updated_by_pending.get(affected_var, state.get_variable(affected_var))
                updated_by_pending[affected_var] = current_val_for_pending + effect_value
                logger.debug(f"Applying pending influence: {affected_var} += {effect_value:.3f} in year {state.year} (was {current_val_for_pending:.3f})")
            else:
                remaining_pending.append((target_year, affected_var, effect_value))
        
        for var, val in updated_by_pending.items():
            state.set_variable(var, val)
        self.pending_influences = remaining_pending

    # _check_and_trigger_events method (remains largely the same logic, ensure it uses the expanded SIMULATION_EVENTS_THRESHOLDS)
    def _check_and_trigger_events(self, state: SimulationState):
        for var_name, event_list in self.event_thresholds.items():
            current_val = state.get_variable(var_name)
            for threshold, desc_template, factors_template, event_domains in event_list:
                event_key = f"{var_name}_{threshold}" # Unique key for this event
                
                # Determine if condition is based on decreasing or increasing variable
                is_decreasing_threshold_beneficial = (var_name == "Environmental_Stability_Index" and threshold < self.sim_vars_config[var_name]["initial_avg"]) or \
                                                    (var_name == "Wealth_Inequality_Index" and threshold < self.sim_vars_config[var_name]["initial_avg"]) # e.g. low inequality is good
                
                triggered_condition = False
                if is_decreasing_threshold_beneficial: # Event triggers if value goes BELOW threshold
                    triggered_condition = (current_val <= threshold)
                else: # Event triggers if value goes ABOVE threshold
                    triggered_condition = (current_val >= threshold)

                if triggered_condition and not state.triggered_event_flags.get(event_key):
                    event_description = f"Year {state.year}: {desc_template}"
                    state.key_events_log.append(event_description)
                    state.triggered_event_flags[event_key] = True
                    logger.debug(f"Event triggered: {event_description}")
                    
                    # Example event feedback: AGI breakthrough boosts related funding and acceptance
                    if "AGI Achieved" in desc_template or "ASI Emergence" in desc_template:
                        state.set_variable("Funding_FutureTech_Level", state.get_variable("Funding_FutureTech_Level") + 0.3)
                        state.set_variable("Public_Acceptance_RadicalTech", state.get_variable("Public_Acceptance_RadicalTech") + 0.2)
                        if "ASI Emergence" in desc_template: # ASI could destabilize collaboration
                             state.set_variable("Global_Collaboration_Index", state.get_variable("Global_Collaboration_Index") - 0.2)
                    elif "Longevity Escape Velocity" in desc_template:
                        state.set_variable("Wealth_Inequality_Index", state.get_variable("Wealth_Inequality_Index") + 0.1) # If LEV is expensive
                    elif "Climate Tipping Point Reached" in desc_template:
                        state.set_variable("Global_Economic_Stability", state.get_variable("Global_Economic_Stability") - 0.3)
                        state.set_variable("Global_Collaboration_Index", state.get_variable("Global_Collaboration_Index") - 0.15) # Crisis can strain collaboration
                        state.set_variable("Climate_Mitigation_Effort", state.get_variable("Climate_Mitigation_Effort") + 0.2) # Spur more effort
                        

    # run_single_simulation method (remains the same structure)
    def run_single_simulation(self, start_year: int, end_year: int) -> Tuple[SimulationState, List[Dict[str, Any]]]:
        state = self._initialize_simulation_state(start_year)
        history = [] 
        self.pending_influences = [] 

        for year_sim in range(start_year, end_year + 1):
            state.year = year_sim # Update current year of the state
            
            # Store state at the beginning of the year BEFORE updates for that year
            current_state_snapshot = {
                "year": year_sim,
                "variables": state.variables.copy(),
                "key_events_triggered_this_year": [] # Will be populated by _check_and_trigger_events
            }
            
            self._apply_stochastic_drift_and_trends(state)
            self._apply_influences(state) # This applies matured pending and calculates new pending
            
            # Store events triggered in THIS year
            events_before_check = len(state.key_events_log)
            self._check_and_trigger_events(state)
            events_after_check = len(state.key_events_log)
            if events_after_check > events_before_check:
                current_state_snapshot["key_events_triggered_this_year"] = state.key_events_log[events_before_check:]

            history.append(current_state_snapshot)
            
        return state, history

    # _simulation_to_genome method (needs more robust mapping based on expanded vars/events)
    def _simulation_to_genome(self, final_state: SimulationState, history: List[Dict[str, Any]], start_year: int, end_year: int) -> ScenarioGenome:
        tech_factors = []
        social_factors = []
        economic_factors = []
        domains_focused_set = set()

        initial_vars = history[0]['variables']
        final_vars = final_state.variables

        # Derive factors based on significant changes or high/low final values
        for var_name, final_val in final_vars.items():
            initial_val = initial_vars.get(var_name, self.sim_vars_config[var_name]["initial_avg"])
            var_config = self.sim_vars_config[var_name]
            var_range = var_config["max"] - var_config["min"]
            
            # Heuristic: significant change is > 30% of range or crosses a major threshold
            if abs(final_val - initial_val) > 0.3 * var_range or \
               (final_val > var_config["min"] + 0.7 * var_range and initial_val < var_config["min"] + 0.4 * var_range) or \
               (final_val < var_config["min"] + 0.3 * var_range and initial_val > var_config["min"] + 0.6 * var_range):

                if var_name == "AGI_Capability" and final_val > 6:
                    tech_factors.append(f"High AGI Capability ({final_val:.1f}/10) achieved.")
                    domains_focused_set.add(DomainEnum.AGI.value) # Use .value for Pydantic string list
                elif var_name == "Biotech_Longevity_Maturity" and final_val > 5:
                    tech_factors.append(f"Advanced Longevity Biotech ({final_val:.1f}/10) developed.")
                    domains_focused_set.add(DomainEnum.LONGEVITY.value)
                elif var_name == "BCI_Integration_Level" and final_val > 5:
                    tech_factors.append(f"Significant BCI Integration ({final_val:.1f}/10) occurs.")
                    domains_focused_set.add(DomainEnum.BCI.value)
                elif var_name == "Nanotech_Manufacturing_Scale" and final_val > 5:
                    tech_factors.append(f"Large-scale Nanotech Manufacturing ({final_val:.1f}/10) becomes reality.")
                    domains_focused_set.add(DomainEnum.NANOTECH.value)
                elif var_name == "Quantum_Computing_Impact" and final_val > 4:
                    tech_factors.append(f"Quantum Computing makes substantial impact ({final_val:.1f}/10).")
                    domains_focused_set.add(DomainEnum.QUANTUM.value)
                elif var_name == "Genetic_Engineering_Precision" and final_val > 6:
                    tech_factors.append(f"High-precision Genetic Engineering ({final_val:.1f}/10) available.")
                    domains_focused_set.add(DomainEnum.GENETICS.value)
                elif var_name == "Space_Colonization_Tech" and final_val > 3:
                    tech_factors.append(f"Notable progress in Space Colonization Tech ({final_val:.1f}/10).")
                    domains_focused_set.add(DomainEnum.SPACE.value)
                elif var_name == "Renewable_Energy_Dominance" and final_val > 7:
                    economic_factors.append(f"Energy sector dominated by renewables ({final_val:.1f}/10).")
                    domains_focused_set.add("renewable_energy") # Assuming this is a string domain
                elif var_name == "Robotics_Automation_Sophistication" and final_val > 6:
                    tech_factors.append(f"Highly sophisticated robotics & automation ({final_val:.1f}/10).")
                    domains_focused_set.add("robotics_automation")

                elif var_name == "Public_Acceptance_RadicalTech":
                    if final_val > 0.7: social_factors.append("High public embrace of transformative technologies.")
                    elif final_val < 0.3: social_factors.append("Strong public skepticism or resistance to new tech.")
                elif var_name == "Global_Collaboration_Index":
                    if final_val > 0.7: social_factors.append("Era of strong international cooperation.")
                    elif final_val < 0.3: social_factors.append("Period of intense geopolitical fragmentation.")
                elif var_name == "Wealth_Inequality_Index": # Higher is worse
                    if final_val > 0.75: economic_factors.append("Extreme wealth inequality dominates society.")
                    elif final_val < 0.35: economic_factors.append("Significant reduction in wealth inequality achieved.")
                elif var_name == "Trust_In_Institutions":
                    if final_val < 0.3: social_factors.append("Widespread erosion of trust in established institutions.")
                elif var_name == "Educational_Adaptability":
                    if final_val > 0.7: social_factors.append("Education systems rapidly adapt to new skill demands.")
                    elif final_val < 0.3: social_factors.append("Workforce struggles with skill gaps due to slow educational adaptation.")
                elif var_name == "Funding_FutureTech_Level":
                    if final_val > 0.7: economic_factors.append("Massive global investment in frontier technologies.")
                elif var_name == "Resource_Scarcity_Index": # Higher is worse
                    if final_val > 0.7: economic_factors.append("Severe global resource scarcity impacts development.")
                elif var_name == "Global_Economic_Stability":
                    if final_val < 0.3: economic_factors.append("Period of major global economic instability and crises.")
                elif var_name == "Environmental_Stability_Index": # Lower is worse
                    if final_val < 0.3: social_factors.append("Severe environmental degradation impacts global society.")
                elif var_name == "Climate_Mitigation_Effort":
                    if final_val > 0.7: social_factors.append("Strong global commitment and action on climate mitigation.")

        # Add factors from explicitly triggered events
        for event_log_entry in final_state.key_events_log:
            for var_key, event_definitions in self.event_thresholds.items():
                for _thresh, _desc_template, factors_template_list, event_domains_list in event_definitions:
                    # Check if the core part of the description template is in the logged event
                    # This makes the matching more robust to the "Year XXXX:" prefix
                    if _desc_template.split(':')[0] in event_log_entry: # Match based on first part of template
                        tech_factors.extend(factors_template_list)
                        domains_focused_set.update(event_domains_list)
                        break 
        
        tech_factors = list(set(tech_factors))
        social_factors = list(set(social_factors))
        economic_factors = list(set(economic_factors))

        return ScenarioGenome(
            id=str(uuid.uuid4()),
            technological_factors=tech_factors[:7], # Limit number of factors
            social_factors=social_factors[:5],
            economic_factors=economic_factors[:5],
            timeline=f"{start_year}-{end_year}",
            key_events=final_state.key_events_log,
            domains_focused=list(domains_focused_set)[:5], # Limit number of domains
            generation=-2 # Mark as synthetic simulation
        )

    # _assign_synthetic_probability method (can be further refined)
    def _assign_synthetic_probability(self, final_state: SimulationState, history: List[Dict[str, Any]]) -> float:
        prob = 0.5  # Base
        num_years_simulated = history[-1]['year'] - history[0]['year'] + 1

        # Stability and positive outcomes
        if final_state.get_variable("Environmental_Stability_Index") > 0.7 and \
           final_state.get_variable("Global_Economic_Stability") > 0.7 and \
           final_state.get_variable("Global_Collaboration_Index") > 0.6:
            prob += 0.2
        elif final_state.get_variable("Environmental_Stability_Index") < 0.2 or \
             final_state.get_variable("Global_Economic_Stability") < 0.2 or \
             final_state.get_variable("Global_Collaboration_Index") < 0.2:
            prob -= 0.3
        
        # Tech advancement vs. societal adaptation
        avg_tech_progress = np.mean([
            final_state.get_variable("AGI_Capability"),
            final_state.get_variable("Biotech_Longevity_Maturity"),
            final_state.get_variable("BCI_Integration_Level"),
            final_state.get_variable("Nanotech_Manufacturing_Scale")
        ])
        avg_societal_factors = np.mean([
            final_state.get_variable("Public_Acceptance_RadicalTech"),
            final_state.get_variable("Trust_In_Institutions"),
            final_state.get_variable("Educational_Adaptability")
        ])

        if avg_tech_progress > 7 and avg_societal_factors < 0.4: # High tech, low adaptation
            prob *= 0.6
        elif avg_tech_progress > 5 and avg_societal_factors > 0.6: # Good tech, good adaptation
            prob *= 1.15

        # Number of major negative events triggered
        num_negative_events = 0
        for event_log in final_state.key_events_log:
            if "Crisis" in event_log or "Collapse" in event_log or "Unrest" in event_log or "Catastrophe" in event_log or "Tipping Point Reached" in event_log:
                num_negative_events += 1
        prob *= (0.8 ** num_negative_events)

        # Number of major positive breakthroughs
        num_positive_breakthroughs = 0
        for event_log in final_state.key_events_log:
            if "Breakthrough" in event_log or "Achieved" in event_log or "Dominance" in event_log or "Revolution" in event_log:
                if not ("Crisis" in event_log or "Existential" in event_log): # Exclude negative breakthroughs
                    num_positive_breakthroughs +=1
        prob *= (1.05 ** min(num_positive_breakthroughs, 3)) # Diminishing returns for too many breakthroughs

        # Penalize scenarios that are "too wild" too quickly
        if (num_positive_breakthroughs + num_negative_events) / num_years_simulated > 0.3: # More than 30% of years have a major event
            prob *= 0.7

        return np.clip(prob + np.random.normal(0, 0.03), 0.01, 0.99) # Less noise for synthetic targets

    # generate_synthetic_dataset method (remains the same structure)
    def generate_synthetic_dataset(self, num_scenarios: int, start_year: int = 2025, sim_duration_years: int = 30) -> List[Tuple[ScenarioGenome, float]]:
        # ... (same as before)
        dataset = []
        logger.info(f"Generating {num_scenarios} synthetic scenarios via simulation...")
        for i in range(num_scenarios):
            if i > 0 and i % (max(1, num_scenarios // 20)) == 0: # Log every 5%
                logger.info(f"Generated {i}/{num_scenarios} synthetic scenarios...")
            
            final_state, history = self.run_single_simulation(start_year, start_year + sim_duration_years -1)
            genome = self._simulation_to_genome(final_state, history, start_year, start_year + sim_duration_years -1)
            probability = self._assign_synthetic_probability(final_state, history)
            dataset.append((genome, probability))
        
        logger.info(f"Finished generating {len(dataset)} synthetic scenarios.")
        return dataset

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