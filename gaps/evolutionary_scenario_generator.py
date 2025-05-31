import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json # For potential LLM interaction if it returns JSON

# Define ScenarioGenome dataclass (as it's central to this module)
@dataclass
class ScenarioGenome:
    id: str # Unique ID for the scenario
    technological_factors: List[str] = field(default_factory=list)
    social_factors: List[str] = field(default_factory=list)
    economic_factors: List[str] = field(default_factory=list)
    timeline: str = "2025-2050" # Default timeline
    key_events: List[str] = field(default_factory=list) # Ordered list of key events
    domains_focused: List[str] = field(default_factory=list) # Domains this scenario focuses on
    # Probability weights might be assigned by a separate forecasting module
    probability_weights: Dict[str, float] = field(default_factory=dict)
    fitness_score: Optional[float] = None # For evolutionary algorithm
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)


class EvolutionaryScenarioGenerator:
    def __init__(self, llm_model_name: str = "gpt-4-turbo", population_size: int = 50, config: Optional[Dict[str, Any]] = None):
        self.llm_model_name = llm_model_name # Name of the LLM model to be used
        # In a real implementation, you'd initialize an LLM client here, e.g.:
        # self.llm_client = YourLLMClient(api_key="...", model_name=llm_model_name)
        self.population_size = population_size

        # Domains for scenario generation, can be loaded from config
        self.available_domains = config.get("scenario_domains", [
            "artificial_general_intelligence", "biotechnology_longevity",
            "brain_computer_interfaces", "nanotechnology", "quantum_computing",
            "space_colonization", "genetic_engineering", "climate_tech_adaptation",
            "decentralized_systems_web3", "future_of_work_education"
        ]) if config else [
            "artificial_general_intelligence", "biotechnology_longevity",
            "brain_computer_interfaces", "nanotechnology", "quantum_computing",
            "space_colonization", "genetic_engineering"
        ]

        # Base prompts or themes for initializing diverse scenarios
        self.base_themes = config.get("base_scenario_themes", {
            "agi_breakthrough": "A scenario where AGI is achieved rapidly through unforeseen methods.",
            "longevity_escape_velocity": "A future where LEV is reached, focusing on societal impacts.",
            "bci_integration": "Widespread adoption of advanced BCIs for cognitive enhancement and communication.",
            "nanotech_revolution": "Molecular manufacturing becomes a reality, transforming industries.",
            "quantum_dominance": "Quantum computers solve major scientific problems, impacting cryptography and materials.",
            "space_settlement": "First permanent off-world human settlements are established.",
            "genetic_mastery": "Human genetic engineering becomes common for disease eradication and enhancement."
        }) if config else {
            "agi_breakthrough": "AGI achieved through recursive self-improvement",
            "longevity_escape": "Longevity escape velocity reached via genetic therapies",
            # ... add more default themes
        }


    def _generate_unique_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Placeholder for actual LLM API call.
        This should interact with an LLM (e.g., OpenAI, Anthropic, local model).
        """
        print(f"\n--- LLM PROMPT (SIMULATED) ---\n{prompt}\n-----------------------------\n")
        # Simulate LLM response structure (JSON-like string)
        # In a real scenario, this would be the actual text response from the LLM.
        # The LLM would be instructed to return JSON.
        example_json_output = {
            "technological_factors": ["AI-driven drug discovery accelerates", "Fusion power becomes viable"],
            "social_factors": ["Debates on AI rights intensify", "Mass reskilling programs for new jobs"],
            "economic_factors": ["Carbon-negative industries boom", "Gig economy evolves with AI agents"],
            "timeline": "2030-2060",
            "key_events": [
                "2032: First AI discovers novel physics principle.",
                "2040: Commercial fusion reactors online in 3 countries.",
                "2045: Global treaty on AI personhood rights (limited).",
                "2055: Average human healthspan extends to 110 years due to biotech."
            ],
            "critical_dependencies": ["Stable global supply chains for rare earths", "Breakthroughs in AI alignment"]
        }
        # Simulate a slight variation for different calls
        if "AGI" in prompt:
            example_json_output["technological_factors"].append("Self-improving AI research agents")
        if "longevity" in prompt:
             example_json_output["social_factors"].append("Multi-generational households become common again")

        return json.dumps(example_json_output) # LLM is instructed to return JSON

    def _parse_llm_response_to_genome_data(self, llm_response_str: str) -> Dict[str, Any]:
        """Parses the LLM's JSON string response into a dictionary."""
        try:
            data = json.loads(llm_response_str)
            # Basic validation of expected keys
            expected_keys = ["technological_factors", "social_factors", "economic_factors", "timeline", "key_events"]
            for key in expected_keys:
                if key not in data:
                    print(f"Warning: LLM response missing key '{key}'. Using empty list/default.")
                    data[key] = [] if isinstance(data.get(key), list) else ""
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding LLM JSON response: {e}. Response was: {llm_response_str}")
            # Return a default structure on error to prevent crashes
            return {
                "technological_factors": [], "social_factors": [], "economic_factors": [],
                "timeline": "2025-2050 (default due to parse error)", "key_events": ["LLM response parse error"]
            }

    def initialize_population(self, num_to_initialize: Optional[int] = None) -> List[ScenarioGenome]:
        """Creates an initial population of diverse scenarios using LLM."""
        if num_to_initialize is None:
            num_to_initialize = self.population_size

        population: List[ScenarioGenome] = []
        theme_keys = list(self.base_themes.keys())

        for i in range(num_to_initialize):
            # Mix base themes and selected domains for diversity
            np.random.seed(i) # For reproducibility if needed during dev
            num_domains_to_mix = np.random.randint(2, 4) # Mix 2 or 3 domains
            selected_domains = list(np.random.choice(self.available_domains, size=num_domains_to_mix, replace=False))

            # Pick a base theme to guide the scenario
            base_theme_description = self.base_themes.get(np.random.choice(theme_keys), "A general future scenario.")

            prompt = f"""
            Generate a detailed and plausible future scenario focusing on the interplay between these domains: {', '.join(selected_domains)}.
            The scenario should be guided by the following theme: "{base_theme_description}".

            Consider the timeline from 2025 up to 2060.

            The output MUST be a single JSON object with the following keys:
            - "technological_factors": (list of strings) Key scientific or technological developments.
            - "social_factors": (list of strings) Significant societal changes, public reactions, or ethical considerations.
            - "economic_factors": (list of strings) Major economic shifts, new industries, or resource implications.
            - "timeline": (string) The overall timeframe this scenario primarily covers (e.g., "2030-2045").
            - "key_events": (list of strings) A chronological sequence of 3-5 pivotal events that define this scenario. Each event should be a concise descriptive sentence, ideally with a potential year.
            - "critical_dependencies": (list of strings, optional) Key enabling factors or breakthroughs this scenario relies upon.

            Ensure the key_events form a coherent narrative progression. Be creative yet grounded.
            """

            llm_response_str = self._call_llm(prompt)
            scenario_data = self._parse_llm_response_to_genome_data(llm_response_str)

            genome = ScenarioGenome(
                id=self._generate_unique_id(),
                technological_factors=scenario_data.get("technological_factors", []),
                social_factors=scenario_data.get("social_factors", []),
                economic_factors=scenario_data.get("economic_factors", []),
                timeline=scenario_data.get("timeline", "2025-2050"),
                key_events=scenario_data.get("key_events", []),
                domains_focused=selected_domains,
                generation=0
            )
            population.append(genome)
        return population

    def crossover_scenarios(self, parent1: ScenarioGenome, parent2: ScenarioGenome) -> ScenarioGenome:
        """Creates an offspring scenario by combining elements from two parent scenarios using LLM."""

        # Simple mixing for factors (can be made more sophisticated)
        tech_factors = list(set(parent1.technological_factors + parent2.technological_factors))
        np.random.shuffle(tech_factors)
        tech_factors = tech_factors[:np.random.randint(3, 6)] # Select a random number of combined factors

        social_factors = list(set(parent1.social_factors + parent2.social_factors))
        np.random.shuffle(social_factors)
        social_factors = social_factors[:np.random.randint(3, 6)]

        economic_factors = list(set(parent1.economic_factors + parent2.economic_factors))
        np.random.shuffle(economic_factors)
        economic_factors = economic_factors[:np.random.randint(3, 6)]

        combined_domains = list(set(parent1.domains_focused + parent2.domains_focused))

        # Use LLM to synthesize a new coherent timeline and key events
        synthesis_prompt = f"""
        Two future scenarios are being combined. Create a new, coherent, and plausible synthesized scenario.

        Parent 1 focused on: {', '.join(parent1.domains_focused)}. Key events were: {'; '.join(parent1.key_events)}
        Parent 2 focused on: {', '.join(parent2.domains_focused)}. Key events were: {'; '.join(parent2.key_events)}

        The new scenario should draw inspiration from both parents, focusing on these combined domains: {', '.join(combined_domains)}.
        Consider these combined factors as context:
        - Technological: {'; '.join(tech_factors)}
        - Social: {'; '.join(social_factors)}
        - Economic: {'; '.join(economic_factors)}

        Generate a new timeline (e.g., "2030-2055") and a new chronological sequence of 3-5 key_events that logically flow from the combined context.
        The output MUST be a single JSON object with keys: "timeline" (string) and "key_events" (list of strings).
        """

        llm_response_str = self._call_llm(synthesis_prompt, max_tokens=500)
        synthesized_data = self._parse_llm_response_to_genome_data(llm_response_str)

        return ScenarioGenome(
            id=self._generate_unique_id(),
            technological_factors=tech_factors,
            social_factors=social_factors,
            economic_factors=economic_factors,
            timeline=synthesized_data.get('timeline', f"{min(parent1.timeline.split('-')[0], parent2.timeline.split('-')[0])}-{max(parent1.timeline.split('-')[-1], parent2.timeline.split('-')[-1])}"),
            key_events=synthesized_data.get('key_events', ["Synthesized event 1", "Synthesized event 2"]),
            domains_focused=combined_domains,
            parent_ids=[parent1.id, parent2.id]
            # probability_weights will be recalculated by the forecaster
        )

    def mutate_scenario(self, scenario: ScenarioGenome, mutation_strength: float = 0.3) -> ScenarioGenome:
        """Applies mutations to a scenario genome using LLM to introduce variations."""

        mutation_type = np.random.choice(["add_event", "modify_factor", "shift_timeline", "introduce_wildcard"])

        prompt_parts = [
            f"Consider the following future scenario (ID: {scenario.id}):",
            f"- Timeline: {scenario.timeline}",
            f"- Domains: {', '.join(scenario.domains_focused)}",
            f"- Technological Factors: {'; '.join(scenario.technological_factors)}",
            f"- Social Factors: {'; '.join(scenario.social_factors)}",
            f"- Economic Factors: {'; '.join(scenario.economic_factors)}",
            f"- Key Events: {'; '.join(scenario.key_events)}\n"
        ]

        if mutation_type == "add_event" or (not scenario.key_events and mutation_type != "shift_timeline"):
            prompt_parts.append(f"Suggest one plausible new key_event that could occur within or slightly extending this scenario's timeline ({scenario.timeline}), consistent with its themes. This event should be a consequence or a new development.")
            instruction = "Return ONLY the text of the new key event as a single string."
        elif mutation_type == "modify_factor" and scenario.technological_factors:
            factor_to_modify = np.random.choice(scenario.technological_factors + scenario.social_factors + scenario.economic_factors)
            prompt_parts.append(f"Suggest a plausible modification or alternative to the factor: '{factor_to_modify}'. The modification should be a nuanced change, not a complete negation.")
            instruction = "Return ONLY the text of the modified factor as a single string."
        elif mutation_type == "shift_timeline":
            shift_direction = np.random.choice(["earlier", "later"])
            shift_amount = np.random.randint(3, 8) # years
            prompt_parts.append(f"How would the key events of this scenario plausibly shift if the overall timeline started {shift_amount} years {shift_direction}? Focus on the sequence and timing of key events.")
            instruction = "Return a JSON object with a new 'timeline' (string) and a revised list of 'key_events' (list of strings)."
        else: # wildcard or if other conditions not met
            wildcard_domain = np.random.choice(self.available_domains)
            prompt_parts.append(f"Introduce an unexpected 'wildcard' development from the domain of '{wildcard_domain}' into this scenario. How would it plausibly alter one of the existing key_events or add a new consequential event?")
            instruction = "Return a JSON object with potentially modified 'key_events' (list of strings) and optionally a new 'technological_factors' (list of strings) if a new tech is introduced."

        prompt_parts.append(f"\n{instruction}")
        mutation_prompt = "\n".join(prompt_parts)

        llm_response_str = self._call_llm(mutation_prompt, max_tokens=300)

        mutated_scenario = scenario # Start with a copy

        try:
            if mutation_type == "add_event" and llm_response_str:
                mutated_scenario.key_events.append(llm_response_str.strip())
            elif mutation_type == "modify_factor" and llm_response_str:
                # This is simplistic; would need to identify which list the factor came from
                if factor_to_modify in mutated_scenario.technological_factors:
                    mutated_scenario.technological_factors.remove(factor_to_modify)
                    mutated_scenario.technological_factors.append(llm_response_str.strip())
                # Add similar logic for social and economic factors
            elif mutation_type == "shift_timeline" or mutation_type == "introduce_wildcard":
                mutation_data = json.loads(llm_response_str)
                if "timeline" in mutation_data:
                    mutated_scenario.timeline = mutation_data["timeline"]
                if "key_events" in mutation_data:
                    mutated_scenario.key_events = mutation_data["key_events"]
                if "technological_factors" in mutation_data and mutation_type == "introduce_wildcard":
                     mutated_scenario.technological_factors.extend(mutation_data["technological_factors"])
                     mutated_scenario.technological_factors = list(set(mutated_scenario.technological_factors))


        except json.JSONDecodeError:
            print(f"Mutation LLM response parse error for type {mutation_type}. No mutation applied.")
        except Exception as e:
            print(f"Error applying mutation of type {mutation_type}: {e}")

        mutated_scenario.id = self._generate_unique_id() # New ID for mutated version
        mutated_scenario.parent_ids = [scenario.id]
        return mutated_scenario

# Example Usage:
if __name__ == '__main__':
    generator = EvolutionaryScenarioGenerator(population_size=2) # Small pop for demo

    print("Initializing population...")
    initial_population = generator.initialize_population()
    for i, genome in enumerate(initial_population):
        print(f"\n--- Initial Genome {i+1} (ID: {genome.id}) ---")
        print(f"  Timeline: {genome.timeline}")
        print(f"  Domains: {genome.domains_focused}")
        print(f"  Tech Factors: {genome.technological_factors}")
        print(f"  Social Factors: {genome.social_factors}")
        print(f"  Economic Factors: {genome.economic_factors}")
        print(f"  Key Events: {genome.key_events}")

    if len(initial_population) >= 2:
        print("\nPerforming crossover...")
        offspring = generator.crossover_scenarios(initial_population[0], initial_population[1])
        print(f"\n--- Offspring Genome (ID: {offspring.id}) ---")
        print(f"  Timeline: {offspring.timeline}")
        print(f"  Domains: {offspring.domains_focused}")
        print(f"  Tech Factors: {offspring.technological_factors}")
        print(f"  Social Factors: {offspring.social_factors}")
        print(f"  Economic Factors: {offspring.economic_factors}")
        print(f"  Key Events: {offspring.key_events}")
        print(f"  Parent IDs: {offspring.parent_ids}")

        print("\nPerforming mutation on offspring...")
        mutated_offspring = generator.mutate_scenario(offspring)
        print(f"\n--- Mutated Offspring Genome (ID: {mutated_offspring.id}) ---")
        print(f"  Timeline: {mutated_offspring.timeline}")
        print(f"  Key Events: {mutated_offspring.key_events}")
        print(f"  Parent ID: {mutated_offspring.parent_ids}")
