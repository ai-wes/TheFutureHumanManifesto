from typing import List, Dict, Any
from .evolutionary_scenario_generator import ScenarioGenome

class ScenarioDatabase:
    def __init__(self):
        self.scenarios: List[ScenarioGenome] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_scenario(self, scenario: ScenarioGenome, meta: Dict[str, Any] = None):
        self.scenarios.append(scenario)
        self.metadata.append(meta or {})

    def get_scenarios(self) -> List[ScenarioGenome]:
        return self.scenarios

    def get_metadata(self) -> List[Dict[str, Any]]:
        return self.metadata

    def clear(self):
        self.scenarios.clear()
        self.metadata.clear()
