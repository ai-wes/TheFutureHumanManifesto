from typing import List, Dict, Any, Optional
# Assuming ScenarioGenome is defined in evolutionary_scenario_generator
from .evolutionary_scenario_generator import ScenarioGenome

class ScenarioDatabase:
    """
    A simple in-memory database for storing and retrieving ScenarioGenomes
    and their associated metadata or evaluation results.
    """
    def __init__(self):
        self.scenarios_storage: Dict[str, ScenarioGenome] = {}
        self.metadata_storage: Dict[str, Dict[str, Any]] = {} # Stores fitness, probability, etc.

    def add_scenario(self, scenario: ScenarioGenome, metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a scenario to the database.
        The scenario's 'id' field is used as the key.
        """
        if not scenario.id:
            raise ValueError("ScenarioGenome must have an 'id' to be added to the database.")

        self.scenarios_storage[scenario.id] = scenario
        if metadata:
            self.metadata_storage[scenario.id] = metadata
        elif scenario.id not in self.metadata_storage: # Initialize if no metadata provided
             self.metadata_storage[scenario.id] = {}


    def get_scenario_by_id(self, scenario_id: str) -> Optional[ScenarioGenome]:
        """Retrieves a scenario by its ID."""
        return self.scenarios_storage.get(scenario_id)

    def get_metadata_by_id(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a scenario by its ID."""
        return self.metadata_storage.get(scenario_id)

    def update_scenario_metadata(self, scenario_id: str, new_metadata: Dict[str, Any], overwrite_all=False):
        """Updates or adds metadata for an existing scenario."""
        if scenario_id not in self.scenarios_storage:
            # Or raise an error: raise KeyError(f"Scenario with ID '{scenario_id}' not found.")
            print(f"Warning: Scenario with ID '{scenario_id}' not found. Cannot update metadata.")
            return

        if overwrite_all or scenario_id not in self.metadata_storage:
            self.metadata_storage[scenario_id] = new_metadata
        else:
            self.metadata_storage[scenario_id].update(new_metadata)


    def get_all_scenarios(self) -> List[ScenarioGenome]:
        """Returns a list of all scenarios in the database."""
        return list(self.scenarios_storage.values())

    def get_all_scenarios_with_metadata(self) -> List[Dict[str, Any]]:
        """Returns a list of scenarios, each combined with its metadata."""
        combined_list = []
        for scenario_id, scenario in self.scenarios_storage.items():
            combined_data = {
                "scenario": scenario, # Or scenario.dict() if pydantic model
                "metadata": self.metadata_storage.get(scenario_id, {})
            }
            combined_list.append(combined_data)
        return combined_list

    def get_scenarios_by_filter(self, filter_func) -> List[ScenarioGenome]:
        """
        Returns scenarios that match a given filter function.
        The filter_func takes a ScenarioGenome and its metadata as input and returns bool.
        """
        matched_scenarios = []
        for scenario_id, scenario in self.scenarios_storage.items():
            metadata = self.metadata_storage.get(scenario_id, {})
            if filter_func(scenario, metadata):
                matched_scenarios.append(scenario)
        return matched_scenarios

    def count(self) -> int:
        """Returns the total number of scenarios in the database."""
        return len(self.scenarios_storage)

    def clear_database(self):
        """Clears all scenarios and metadata from the database."""
        self.scenarios_storage.clear()
        self.metadata_storage.clear()
        print("Scenario database cleared.")

# Example Usage:
if __name__ == '__main__':
    db = ScenarioDatabase()

    # Assuming ScenarioGenome is defined (e.g., from evolutionary_scenario_generator)
    # For this example, let's mock it if not directly available
    from dataclasses import dataclass, field
    import uuid

    @dataclass
    class ScenarioGenome:
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        technological_factors: List[str] = field(default_factory=list)
        social_factors: List[str] = field(default_factory=list)
        economic_factors: List[str] = field(default_factory=list)
        timeline: str = "2025-2050"
        key_events: List[str] = field(default_factory=list)
        domains_focused: List[str] = field(default_factory=list)
        probability_weights: Dict[str, float] = field(default_factory=dict)
        fitness_score: Optional[float] = None
        generation: int = 0
        parent_ids: List[str] = field(default_factory=list)


    # Create some sample scenarios
    scenario1 = ScenarioGenome(
        id="scn_001",
        technological_factors=["AGI achieved"],
        key_events=["Global AI summit"],
        fitness_score=0.85,
        generation=5
    )
    scenario2 = ScenarioGenome(
        id="scn_002",
        technological_factors=["Longevity breakthrough"],
        key_events=["First human lives to 150"],
        fitness_score=0.72,
        generation=5
    )
    scenario3 = ScenarioGenome(
        id="scn_003",
        technological_factors=["AGI achieved", "BCI widespread"],
        key_events=["AI-human cognitive merge"],
        fitness_score=0.91,
        generation=8,
        domains_focused=["artificial_general_intelligence", "brain_computer_interfaces"]
    )

    # Add scenarios to the database
    db.add_scenario(scenario1, metadata={"theme": "AGI", "probability": 0.6, "consistency": 0.9})
    db.add_scenario(scenario2, metadata={"theme": "Longevity", "probability": 0.4})
    db.add_scenario(scenario3) # Add without initial full metadata

    print(f"Total scenarios in DB: {db.count()}")

    # Retrieve a scenario
    retrieved_scn = db.get_scenario_by_id("scn_001")
    if retrieved_scn:
        print(f"\nRetrieved Scenario ID scn_001: {retrieved_scn.technological_factors}")

    retrieved_meta = db.get_metadata_by_id("scn_001")
    if retrieved_meta:
        print(f"Metadata for scn_001: {retrieved_meta}")

    # Update metadata for scenario3
    db.update_scenario_metadata("scn_003", {"theme": "AGI+BCI", "probability": 0.75, "consistency": 0.85})
    print(f"\nUpdated metadata for scn_003: {db.get_metadata_by_id('scn_003')}")


    # Get all scenarios with metadata
    print("\nAll scenarios with metadata:")
    all_data = db.get_all_scenarios_with_metadata()
    for item in all_data:
        print(f"  ID: {item['scenario'].id}, Fitness: {item['scenario'].fitness_score}, Theme: {item['metadata'].get('theme', 'N/A')}")

    # Filter scenarios (e.g., fitness > 0.8)
    def high_fitness_filter(scenario, metadata):
        return scenario.fitness_score is not None and scenario.fitness_score > 0.8

    print("\nHigh fitness scenarios (>0.8):")
    high_fitness_scenarios = db.get_scenarios_by_filter(high_fitness_filter)
    for scn in high_fitness_scenarios:
        print(f"  ID: {scn.id}, Fitness: {scn.fitness_score}")

    # Filter by domain
    def agi_domain_filter(scenario, metadata):
        return "artificial_general_intelligence" in scenario.domains_focused

    print("\nAGI domain scenarios:")
    agi_scenarios = db.get_scenarios_by_filter(agi_domain_filter)
    for scn in agi_scenarios:
        print(f"  ID: {scn.id}, Domains: {scn.domains_focused}")


    # Clear the database
    # db.clear_database()
    # print(f"Total scenarios after clearing: {db.count()}")
