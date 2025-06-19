# src/scenarios/evolutionary_gen.py
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from langchain.chains import LLMChain
from ..utils.models import ScenarioGenome

class ScenarioOptimizationProblem(ElementwiseProblem):
    def __init__(self, llm, domains):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0)
        self.llm_chain = LLMChain(llm=llm, prompt=SCENARIO_PROMPT)
        self.domains = domains

    def _evaluate(self, X, out, *args, **kwargs):
        genome = self.decode_genome(X)
        scenario = self.generate_scenario(genome)
        
        # Multi-objective fitness calculation
        fitness = (
            0.4 * self.probability_score(scenario) +
            0.3 * self.novelty_score(scenario) +
            0.2 * self.consistency_score(scenario) +
            0.1 * self.diversity_score(scenario)
        )
        
        out["F"] = -fitness  # Minimize negative fitness

    def decode_genome(self, X):
        return ScenarioGenome(
            technological_factors=self._decode_section(X[0:3]),
            social_factors=self._decode_section(X[3:6]),
            economic_factors=self._decode_section(X[6:9]),
            timeline_structure=int(X[9])
        )

class EvolutionaryGenerator:
    def __init__(self, config):
        self.pop_size = config.get('scenarios.population_size')
        self.llm = load_llm(config)
        self.domains = config.get('forecasting.domains')
        
    def generate_scenarios(self):
        problem = ScenarioOptimizationProblem(self.llm, self.domains)
        algorithm = GA(pop_size=self.pop_size)
        res = minimize(problem, algorithm, ('n_gen', 50), verbose=False)
        return self.post_process(res.X)
