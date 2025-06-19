import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import uuid # Moved import to top
import os
from config_loader import ConfigLoader
from custom_logging import get_logger

# Attempt to import OpenAI client, or use a placeholder if not available
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None # Placeholder if openai library is not installed

# Assuming ScenarioGenome is defined here or imported if moved to models.py
# For consistency, let's assume it might be in utils.models

from utils.models import ScenarioGenome # If you move it there


logger = get_logger("evolutionary_scenario_generator")

class EvolutionaryScenarioGenerator:
    def __init__(self, population_size: int = 50, config_loader_instance: Optional[ConfigLoader] = None):
        if config_loader_instance:
            self.config_loader = config_loader_instance
        else:
            # Default path, ensure GAPS_CONFIG_PATH env var can override
            config_path = os.getenv("GAPS_CONFIG_PATH", "config/config.yaml")
            self.config_loader = ConfigLoader(config_path=config_path)

        self.llm_config = self.config_loader.get_openai_config() # Fetches model, api_key etc.
        self.llm_model_name = self.llm_config.get("model", "gpt-4-turbo-preview") # Default from your config

        if OpenAI and self.llm_config.get("api_key"):
            self.llm_client = OpenAI(api_key=self.llm_config.get("api_key"))
            logger.info(f"OpenAI client initialized with model: {self.llm_model_name}")
        else:
            self.llm_client = None
            logger.warning("OpenAI client could not be initialized. API key might be missing or library not installed. LLM calls will be simulated.")

        self.population_size = population_size
        gapse_config = self.config_loader.get("gapse_settings", {})

        self.available_domains = gapse_config.get("scenario_generator.available_domains", [
            "artificial_general_intelligence", "biotechnology_longevity",
            "brain_computer_interfaces", "nanotechnology", "quantum_computing",
            "space_colonization", "genetic_engineering", "climate_tech_adaptation",
            "decentralized_systems_web3", "future_of_work_education"
        ])
        self.base_themes = gapse_config.get("scenario_generator.base_themes", {
            "agi_breakthrough": "A scenario where AGI is achieved rapidly through unforeseen methods.",
            "longevity_escape_velocity": "A future where LEV is reached, focusing on societal impacts.",
            "bci_integration": "Widespread adoption of advanced BCIs for cognitive enhancement and communication.",
            # Add more default themes if needed
        })
        self.max_tokens_llm = self.llm_config.get("max_tokens", 1000)


    def _generate_unique_id(self):
        return str(uuid.uuid4())

    def _call_llm(self, prompt: str, max_tokens: Optional[int] = None):
        """Actual LLM API call or simulation if client not available."""
        if not max_tokens:
            max_tokens = self.max_tokens_llm

        if self.llm_client:
            try:
                logger.debug(f"Sending prompt to LLM ({self.llm_model_name}):\n{prompt[:300]}...")
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert futurist and scenario planner. Provide responses in JSON format as requested."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7, # Make temperature configurable if needed
                    response_format={"type": "json_object"} # For newer OpenAI models that support JSON mode
                )
                content = response.choices[0].message.content
                logger.debug(f"LLM response received: {content[:300]}...")
                return content if content else "{}" # Return empty JSON if content is None
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                return json.dumps({"error": str(e), "message": "LLM call failed"}) # Return error as JSON
        else:
            # Fallback to simulation if LLM client is not initialized
            logger.warning(f"LLM client not available. Simulating LLM call for prompt:\n{prompt[:300]}...")
            example_json_output = {
                "technological_factors": ["Simulated: AI-driven drug discovery", "Simulated: Fusion power viable"],
                "social_factors": ["Simulated: Debates on AI rights", "Simulated: Mass reskilling programs"],
                "economic_factors": ["Simulated: Carbon-negative industries", "Simulated: Gig economy evolves"],
                "timeline": "2030-2060 (Simulated)",
                "key_events": [
                    "Simulated Event 1: AI discovers novel physics principle (2032).",
                    "Simulated Event 2: Commercial fusion reactors online (2040)."
                ],
                "critical_dependencies": ["Simulated: Stable global supply chains", "Simulated: AI alignment breakthroughs"]
            }
            if "AGI" in prompt:
                example_json_output["technological_factors"].append("Simulated: Self-improving AI research agents")
            if "longevity" in prompt:
                 example_json_output["social_factors"].append("Simulated: Multi-generational households common")
            return json.dumps(example_json_output)

    def _parse_llm_response_to_genome_data(self, llm_response_str: str) -> Dict[str, Any]:
        try:
            data = json.loads(llm_response_str)
            if "error" in data:
                logger.error(f"LLM returned an error: {data['error']}")
                # Fallback to default structure
                return {
                    "technological_factors": [], "social_factors": [], "economic_factors": [],
                    "timeline": "Eror in LLM response", "key_events": [f"LLM Error: {data.get('message', 'Unknown error')}"]
                }

            expected_keys = ["technological_factors", "social_factors", "economic_factors", "timeline", "key_events"]
            parsed_data = {}
            for key in expected_keys:
                if key in data and isinstance(data[key], list if key != "timeline" else str):
                    parsed_data[key] = data[key]
                else:
                    logger.warning(f"LLM response missing or malformed key '{key}'. Using default. Response: {llm_response_str[:200]}")
                    parsed_data[key] = [] if key != "timeline" else "2025-2050 (Default)"
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding LLM JSON response: {e}. Response was: {llm_response_str[:500]}")
            return {
                "technological_factors": [], "social_factors": [], "economic_factors": [],
                "timeline": "2025-2050 (JSON parse error)", "key_events": ["LLM response JSON parse error"]
            }

    # initialize_population, crossover_scenarios, mutate_scenario methods remain largely the same
    # but will now use the actual _call_llm method.

    def initialize_population(self, num_to_initialize: Optional[int] = None) -> List[ScenarioGenome]:
        if num_to_initialize is None:
            num_to_initialize = self.population_size
        population: List[ScenarioGenome] = []
        theme_keys = list(self.base_themes.keys())

        for i in range(num_to_initialize):
            np.random.seed(i)
            num_domains_to_mix = np.random.randint(2, min(4, len(self.available_domains) + 1))
            selected_domains = list(np.random.choice(self.available_domains, size=num_domains_to_mix, replace=False))
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
            (Optional) "critical_dependencies": (list of strings) Key enabling factors or breakthroughs this scenario relies upon.
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
        tech_factors = list(set(parent1.technological_factors + parent2.technological_factors))
        np.random.shuffle(tech_factors)
        tech_factors = tech_factors[:np.random.randint(3, min(6, len(tech_factors) + 1))]
        social_factors = list(set(parent1.social_factors + parent2.social_factors))
        np.random.shuffle(social_factors)
        social_factors = social_factors[:np.random.randint(3, min(6, len(social_factors) + 1))]
        economic_factors = list(set(parent1.economic_factors + parent2.economic_factors))
        np.random.shuffle(economic_factors)
        economic_factors = economic_factors[:np.random.randint(3, min(6, len(economic_factors) + 1))]
        combined_domains = list(set(parent1.domains_focused + parent2.domains_focused))

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
        )

    def mutate_scenario(self, scenario: ScenarioGenome, mutation_strength: float = 0.3) -> ScenarioGenome:
        # Ensure there are factors to modify if that mutation type is chosen
        available_factors = scenario.technological_factors + scenario.social_factors + scenario.economic_factors
        possible_mutation_types = ["add_event", "shift_timeline", "introduce_wildcard"]
        if available_factors:
            possible_mutation_types.append("modify_factor")

        mutation_type = np.random.choice(possible_mutation_types)
        prompt_parts = [
            f"Consider the following future scenario (ID: {scenario.id}):",
            f"- Timeline: {scenario.timeline}",
            f"- Domains: {', '.join(scenario.domains_focused)}",
            f"- Technological Factors: {'; '.join(scenario.technological_factors)}",
            f"- Social Factors: {'; '.join(scenario.social_factors)}",
            f"- Economic Factors: {'; '.join(scenario.economic_factors)}",
            f"- Key Events: {'; '.join(scenario.key_events)}\n"
        ]
        instruction = ""

        if mutation_type == "add_event" or (not scenario.key_events and mutation_type != "shift_timeline"):
            prompt_parts.append(f"Suggest one plausible new key_event that could occur within or slightly extending this scenario's timeline ({scenario.timeline}), consistent with its themes. This event should be a consequence or a new development.")
            instruction = "Return ONLY the text of the new key event as a single string."
        elif mutation_type == "modify_factor" and available_factors: # Check available_factors
            factor_to_modify = np.random.choice(available_factors)
            prompt_parts.append(f"Suggest a plausible modification or alternative to the factor: '{factor_to_modify}'. The modification should be a nuanced change, not a complete negation.")
            instruction = "Return ONLY the text of the modified factor as a single string."
        elif mutation_type == "shift_timeline":
            shift_direction = np.random.choice(["earlier", "later"])
            shift_amount = np.random.randint(3, 8)
            prompt_parts.append(f"How would the key events of this scenario plausibly shift if the overall timeline started {shift_amount} years {shift_direction}? Focus on the sequence and timing of key events.")
            instruction = "Return a JSON object with a new 'timeline' (string) and a revised list of 'key_events' (list of strings)."
        elif mutation_type == "introduce_wildcard": # Wildcard
            wildcard_domain = np.random.choice(self.available_domains)
            prompt_parts.append(f"Introduce an unexpected 'wildcard' development from the domain of '{wildcard_domain}' into this scenario. How would it plausibly alter one of the existing key_events or add a new consequential event?")
            instruction = "Return a JSON object with potentially modified 'key_events' (list of strings) and optionally a new 'technological_factors' (list of strings) if a new tech is introduced."
        else: # Should not happen if logic is correct, but as a fallback
            logger.warning(f"Unexpected mutation scenario for type {mutation_type}. Defaulting to add_event logic.")
            prompt_parts.append(f"Suggest one plausible new key_event for this scenario.")
            instruction = "Return ONLY the text of the new key event as a single string."


        prompt_parts.append(f"\n{instruction}")
        mutation_prompt = "\n".join(prompt_parts)
        llm_response_str = self._call_llm(mutation_prompt, max_tokens=300)

        # Create a deep copy for mutation to avoid modifying the original object in the population list directly
        mutated_scenario = ScenarioGenome(
            id=self._generate_unique_id(), # New ID for mutated version
            technological_factors=list(scenario.technological_factors),
            social_factors=list(scenario.social_factors),
            economic_factors=list(scenario.economic_factors),
            timeline=scenario.timeline,
            key_events=list(scenario.key_events),
            domains_focused=list(scenario.domains_focused),
            probability_weights=dict(scenario.probability_weights),
            fitness_score=None, # Will be re-evaluated
            generation=scenario.generation, # Will be updated by GAPSESystem
            parent_ids=[scenario.id]
        )

        try:
            if mutation_type == "add_event" and llm_response_str:
                new_event = llm_response_str.strip()
                if new_event and not new_event.startswith("{"): # Ensure it's not an unexpected JSON
                    mutated_scenario.key_events.append(new_event)
            elif mutation_type == "modify_factor" and llm_response_str and available_factors:
                modified_factor_text = llm_response_str.strip()
                if modified_factor_text and not modified_factor_text.startswith("{"):
                    # Simplistic replacement: find and replace. More robust would be needed.
                    if factor_to_modify in mutated_scenario.technological_factors:
                        mutated_scenario.technological_factors = [modified_factor_text if f == factor_to_modify else f for f in mutated_scenario.technological_factors]
                    elif factor_to_modify in mutated_scenario.social_factors:
                        mutated_scenario.social_factors = [modified_factor_text if f == factor_to_modify else f for f in mutated_scenario.social_factors]
                    elif factor_to_modify in mutated_scenario.economic_factors:
                        mutated_scenario.economic_factors = [modified_factor_text if f == factor_to_modify else f for f in mutated_scenario.economic_factors]
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
            logger.error(f"Mutation LLM response parse error for type {mutation_type}. No mutation applied. Response: {llm_response_str[:200]}")
        except Exception as e:
            logger.error(f"Error applying mutation of type {mutation_type}: {e}")
        return mutated_scenario


if __name__ == '__main__':
    # Example: Ensure GAPS_CONFIG_PATH is set or config/config.yaml exists with openai section
    # Create a dummy config/config.yaml if it doesn't exist for testing
    if not os.path.exists("config/config.yaml"):
        os.makedirs("config", exist_ok=True)
        with open("config/config.yaml", "w") as f:
            f.write("openai:\n  api_key: YOUR_KEY_HERE_OR_SET_ENV_VAR\n  model: gpt-3.5-turbo\n") # Use a cheaper model for testing
        print("Created dummy config/config.yaml. Please add your OpenAI API key or ensure OPENAI_API_KEY env var is set.")

    generator = EvolutionaryScenarioGenerator(population_size=2)
    logger.info("Initializing population...")
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
