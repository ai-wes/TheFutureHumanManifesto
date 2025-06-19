from typing import Dict, List, Optional, Tuple
import numpy as np
from .evolutionary_scenario_generator import EvolutionaryScenarioGenerator, ScenarioGenome
from .hybrid_probabilistic_forecaster import HybridProbabilisticForecaster
from .contradiction_analysis_engine import ContradictionAnalysisEngine
from .scenario_database import ScenarioDatabase # Assuming this is for storing/retrieving
from typing import Any
class GAPSESystem:
    """
    Generative Assistive Prediction System for Evolution (GAPS-E).
    Integrates scenario generation, probabilistic forecasting, and contradiction analysis.
    """
    def __init__(self, config: Optional[Dict[str, any]] = None):
        self.config = config if config else {} # Load default config if none provided

        # Initialize components
        self.scenario_generator = EvolutionaryScenarioGenerator(
            llm_model_name=self.config.get("llm_model", "gpt-4-turbo"),
            population_size=self.config.get("evolution_population_size", 50),
            config=self.config.get("scenario_generator_config", {})
        )
        self.probabilistic_forecaster = HybridProbabilisticForecaster(
            config=self.config.get("forecaster_config", {})
        )
        self.contradiction_analyzer = ContradictionAnalysisEngine()

        # Database for storing generated and evaluated scenarios (optional)
        self.scenario_database = ScenarioDatabase() # In-memory for now

        # Default book structure if not in config
        self.default_book_structure = {
            "agi_emergence": "Scenarios for artificial general intelligence development and societal impact.",
            "longevity_breakthrough": "Exploring futures shaped by radical life extension and biological enhancement.",
            "neural_integration": "The implications of advanced brain-computer interfaces and cognitive augmentation.",
            "space_expansion": "Narratives of human colonization and adaptation beyond Earth.",
            "consciousness_evolution": "Futures involving digital consciousness, AI rights, and identity transformation.",
            "global_governance_futures": "Evolution of political and social structures in response to disruptive tech.",
            "economic_paradigm_shifts": "Scenarios of post-scarcity, AI-driven economies, and resource allocation."
        }

    def _evaluate_fitness(self, scenario: ScenarioGenome) -> float:
        """
        Evaluates the fitness of a scenario based on its probability and consistency.
        """
        # Get probability forecast for the scenario
        # This might involve more complex feature extraction if the forecaster needs it
        prob_result = self.probabilistic_forecaster.predict_scenario_probability(scenario)

        # Analyze consistency
        consistency_result = self.contradiction_analyzer.analyze_scenario_consistency(scenario)

        # Define weights for combining scores (can be configurable)
        prob_weight = self.config.get("fitness_prob_weight", 0.5)
        consistency_weight = self.config.get("fitness_consistency_weight", 0.5)

        # Normalize scores if they are not already in a comparable range (e.g., 0-1)
        # Assuming probability is 0-1, consistency_score is 0-1

        fitness = (prob_result.get('probability', 0.0) * prob_weight +
                   consistency_result.get('consistency_score', 0.0) * consistency_weight)

        # Optional: Penalize for lack of novelty or diversity if those metrics are implemented
        # novelty_score = self._calculate_novelty(scenario, current_population)
        # diversity_penalty = self._calculate_diversity_penalty(scenario, current_population)
        # fitness = fitness * novelty_score * (1 - diversity_penalty)

        return round(fitness, 4)

    def _evolve_population(self, population: List[ScenarioGenome], generation_num: int) -> List[ScenarioGenome]:
        """
        Evolves the population for one generation using selection, crossover, and mutation.
        """
        # 1. Evaluate fitness for all scenarios in the current population
        for scenario in population:
            if scenario.fitness_score is None: # Evaluate only if not already scored
                scenario.fitness_score = self._evaluate_fitness(scenario)

        # Sort population by fitness (descending)
        population.sort(key=lambda s: s.fitness_score if s.fitness_score is not None else 0.0, reverse=True)

        new_population: List[ScenarioGenome] = []

        # 2. Elitism: Carry over the top N best scenarios
        elite_count = self.config.get("evolution_elite_count", max(1, int(len(population) * 0.1))) # e.g., top 10%
        new_population.extend(population[:elite_count])

        # 3. Selection (e.g., Tournament Selection or Roulette Wheel)
        # For simplicity, using fitness-proportionate selection (roulette wheel like) from the sorted list
        # More robust: Implement tournament selection
        num_parents_to_select = len(population) - elite_count

        # Create offspring through crossover and mutation
        while len(new_population) < self.population_size:
            if len(population) < 2: # Not enough parents
                if population: new_population.append(population[0]) # Add if one left
                break

            # Simple selection: pick two parents, biased towards higher fitness
            # A more robust method like tournament selection is recommended for larger populations
            parent1_idx = np.random.choice(len(population), p=self._get_selection_probabilities(population))
            parent2_idx = np.random.choice(len(population), p=self._get_selection_probabilities(population))
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            if parent1.id == parent2.id and len(population) > 1: # Avoid self-crossover if possible
                 parent2_idx = (parent1_idx + 1) % len(population)
                 parent2 = population[parent2_idx]

            # 4. Crossover
            if np.random.rand() < self.config.get("evolution_crossover_rate", 0.7):
                offspring = self.scenario_generator.crossover_scenarios(parent1, parent2)
            else: # No crossover, pick one parent to pass through (potentially mutate)
                offspring = parent1 if np.random.rand() < 0.5 else parent2

            # 5. Mutation
            if np.random.rand() < self.config.get("evolution_mutation_rate", 0.15):
                offspring = self.scenario_generator.mutate_scenario(offspring)

            offspring.generation = generation_num
            offspring.fitness_score = None # Needs re-evaluation
            new_population.append(offspring)

            if len(new_population) >= self.population_size:
                break

        return new_population[:self.population_size] # Ensure population size constraint

    def _get_selection_probabilities(self, population: List[ScenarioGenome]) -> List[float]:
        """Helper for fitness-proportionate selection probabilities."""
        fitness_values = np.array([s.fitness_score if s.fitness_score is not None else 0.0 for s in population])
        if np.sum(fitness_values) == 0: # Avoid division by zero if all fitnesses are 0
            return [1.0 / len(population)] * len(population)
        probabilities = fitness_values / np.sum(fitness_values)
        return probabilities


    def generate_scenarios_for_theme(self, theme_name: str, theme_description: str, num_generations: int, scenarios_to_return: int) -> List[ScenarioGenome]:
        """
        Generates and evolves scenarios for a specific theme.
        """
        print(f"\n--- Generating scenarios for theme: {theme_name} ---")
        print(f"Description: {theme_description}")

        # Initialize population (can be themed if generator supports it)
        # For now, using the general initializer
        current_population = self.scenario_generator.initialize_population()

        for gen in range(num_generations):
            print(f"  Generation {gen + 1}/{num_generations} for theme '{theme_name}'...")
            current_population = self._evolve_population(current_population, generation_num=gen + 1)

            # Log best fitness in generation (optional)
            if current_population and current_population[0].fitness_score is not None:
                 print(f"    Best fitness in Gen {gen+1}: {current_population[0].fitness_score:.3f} (ID: {current_population[0].id})")


        # Final evaluation and sorting of the last generation
        for scenario in current_population:
            if scenario.fitness_score is None:
                scenario.fitness_score = self._evaluate_fitness(scenario)
        current_population.sort(key=lambda s: s.fitness_score if s.fitness_score is not None else 0.0, reverse=True)

        # Store all evolved scenarios for this theme (optional)
        # for s in current_population:
        #     self.scenario_database.add_scenario(s, meta={'theme': theme_name, 'generation': 'final_evolved'})

        return current_population[:scenarios_to_return]


    def generate_book_content_plan(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates a plan of scenarios for a book, structured by chapters/themes.
        Returns a dictionary where keys are chapter themes and values are lists of scenario details.
        """
        book_structure = self.config.get("book_structure", self.default_book_structure)
        num_generations = self.config.get("evolution_num_generations", 10)
        scenarios_per_chapter = self.config.get("scenarios_per_chapter", 3) # Top N scenarios per theme

        all_chapter_scenarios_detailed: Dict[str, List[Dict[str, Any]]] = {}

        for chapter_theme, description in book_structure.items():
            evolved_scenarios = self.generate_scenarios_for_theme(
                theme_name=chapter_theme,
                theme_description=description,
                num_generations=num_generations,
                scenarios_to_return=scenarios_per_chapter
            )

            chapter_scenario_details_list = []
            for scenario in evolved_scenarios:
                prob_result = self.probabilistic_forecaster.predict_scenario_probability(scenario)
                consistency_result = self.contradiction_analyzer.analyze_scenario_consistency(scenario)

                scenario_detail = {
                    "scenario_id": scenario.id,
                    "timeline": scenario.timeline,
                    "domains_focused": scenario.domains_focused,
                    "technological_factors": scenario.technological_factors,
                    "social_factors": scenario.social_factors,
                    "economic_factors": scenario.economic_factors,
                    "key_events": scenario.key_events,
                    "fitness_score": scenario.fitness_score,
                    "probability_forecast": prob_result,
                    "consistency_analysis": consistency_result,
                    "generation": scenario.generation,
                    "parent_ids": scenario.parent_ids
                }
                chapter_scenario_details_list.append(scenario_detail)

            all_chapter_scenarios_detailed[chapter_theme] = chapter_scenario_details_list

        return all_chapter_scenarios_detailed

# Example Usage:
if __name__ == '__main__':
    # Example configuration (can be loaded from a YAML file)
    sample_config = {
        "llm_model": "simulated_llm_fast", # For faster demo
        "evolution_population_size": 10, # Smaller for demo
        "evolution_num_generations": 3,  # Fewer for demo
        "scenarios_per_chapter": 2,      # Fewer for demo
        "evolution_elite_count": 1,
        "evolution_crossover_rate": 0.8,
        "evolution_mutation_rate": 0.2,
        "fitness_prob_weight": 0.6,
        "fitness_consistency_weight": 0.4,
        "book_structure": {
            "AI_Advancements": "Exploring rapid AI progress and its immediate consequences.",
            "BioTech_Futures": "Scenarios focusing on breakthroughs in longevity and genetic engineering."
        },
        "scenario_generator_config": {
            # "base_scenario_themes": { ... } # Can override defaults here
        },
        "forecaster_config": {
            # "neural_input_dim": 50 # Example specific config for forecaster
        }
    }

    gapse_system = GAPSESystem(config=sample_config)

    print("Starting GAPS-E System to generate book content plan...")
    book_plan = gapse_system.generate_book_content_plan()

    print("\n\n--- Generated Book Content Plan ---")
    for chapter, scenarios_list in book_plan.items():
        print(f"\nChapter Theme: {chapter}")
        if not scenarios_list:
            print("  No scenarios generated for this theme.")
            continue
        for i, scn_detail in enumerate(scenarios_list):
            print(f"  Scenario {i+1} (ID: {scn_detail['scenario_id']}):")
            print(f"    Timeline: {scn_detail['timeline']}")
            print(f"    Fitness: {scn_detail['fitness_score']:.3f}")
            print(f"    Probability: {scn_detail['probability_forecast'].get('probability', 'N/A'):.3f}")
            print(f"    Consistency: {scn_detail['consistency_analysis'].get('consistency_score', 'N/A'):.2f}")
            print(f"    Key Events: {scn_detail['key_events'][:2]}...") # Show first few
            if scn_detail['consistency_analysis'].get('contradictions'):
                print(f"    Contradictions: {scn_detail['consistency_analysis']['contradictions']}")
