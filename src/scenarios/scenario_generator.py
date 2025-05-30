# src/scenarios/scenario_generator.py
import numpy as np
import pandas as pd # pandas is imported but not used in this snippet, can be removed if not needed later
import random

class ScenarioGenerator:
    def __init__(self, forecast_outputs: dict):
        """
        Initializes the ScenarioGenerator with forecast outputs from STS models.

        Args:
            forecast_outputs (dict): A dictionary where keys are milestone names 
                                     (e.g., "AGI_Achievement", "Longevity_Escape_Velocity") 
                                     and values are dictionaries containing at least:
                                     - "samples": A NumPy array of forecast samples 
                                                  (shape: num_samples, num_steps_forecast).
                                     - "forecast_years": A NumPy array of the actual years 
                                                       corresponding to the forecast steps.
        """
        if not isinstance(forecast_outputs, dict):
            raise TypeError("forecast_outputs must be a dictionary.")
        
        for milestone, data in forecast_outputs.items():
            if not all(key in data for key in ["samples", "forecast_years"]):
                raise ValueError(f"Missing 'samples' or 'forecast_years' in forecast_outputs for milestone '{milestone}'.")
            if not isinstance(data["samples"], np.ndarray):
                raise TypeError(f"'samples' for milestone '{milestone}' must be a NumPy array.")
            if not isinstance(data["forecast_years"], np.ndarray):
                raise TypeError(f"'forecast_years' for milestone '{milestone}' must be a NumPy array.")
            if data["samples"].shape[1] != len(data["forecast_years"]):
                raise ValueError(
                    f"Mismatch between number of forecast steps in 'samples' ({data['samples'].shape[1]}) "
                    f"and length of 'forecast_years' ({len(data['forecast_years'])}) for milestone '{milestone}'."
                )

        self.forecast_outputs = forecast_outputs

    def generate_scenarios(self, num_scenarios=1000):
        """
        Generates a specified number of scenarios based on the forecast samples.

        Each scenario represents a single Monte Carlo draw across all forecasted milestones.
        The "events" within each scenario detail the predicted value for each milestone 
        at each forecasted year for that particular Monte Carlo sample path.

        Args:
            num_scenarios (int): The number of scenarios to generate.

        Returns:
            list: A list of scenario dictionaries.
        """
        scenarios = []
        
        milestone_names = list(self.forecast_outputs.keys())
        if not milestone_names:
            print("Warning: No milestones provided in forecast_outputs. Returning empty list of scenarios.")
            return []
            
        # Get the number of available samples from the first milestone (assuming all have the same)
        # A more robust check would ensure all milestones have the same num_forecast_samples.
        first_milestone_data = self.forecast_outputs[milestone_names[0]]
        num_available_samples = first_milestone_data["samples"].shape[0]

        if num_available_samples == 0:
            print("Warning: No forecast samples available in the provided data. Returning empty list of scenarios.")
            return []

        if num_scenarios > num_available_samples:
            print(
                f"Warning: Requested num_scenarios ({num_scenarios}) is greater than available "
                f"forecast samples ({num_available_samples}). Generating {num_available_samples} scenarios instead."
            )
            num_scenarios_to_generate = num_available_samples
        else:
            num_scenarios_to_generate = num_scenarios

        for i in range(num_scenarios_to_generate):
            scenario_id = f"scenario_{i+1}"
            scenario_events = {}
            
            # For each scenario, pick a consistent random sample index across all milestones.
            # This ensures that the i-th scenario uses the i-th sample path from *each* milestone's forecast distribution.
            sample_idx = random.randint(0, num_available_samples - 1) # Or simply use `i` if num_scenarios_to_generate <= num_available_samples
            if num_scenarios_to_generate <= num_available_samples and i < num_available_samples:
                 sample_idx = i # Iterate through available samples if generating fewer/equal scenarios than samples
            else:
                 sample_idx = random.randint(0, num_available_samples - 1) # Random sample if more scenarios than samples (with replacement)

            for milestone_name, forecast_data in self.forecast_outputs.items():
                milestone_path_sample = forecast_data["samples"][sample_idx, :] # Shape (num_steps_forecast,)
                forecast_years_for_milestone = forecast_data["forecast_years"]
                
                event_details_for_milestone = {}
                for step_idx, year_value in enumerate(milestone_path_sample):
                    actual_year = forecast_years_for_milestone[step_idx]
                    event_details_for_milestone[f"year_{int(actual_year)}"] = float(year_value)
                
                scenario_events[milestone_name] = event_details_for_milestone

            # Simplified probability: each scenario is one path out of the sampled paths.
            # True joint probability is complex and depends on correlations not modeled here.
            scenario_probability_estimate = 1.0 / num_available_samples 

            scenarios.append({
                "scenario_id": scenario_id,
                "events": scenario_events,
                "probability_estimate": scenario_probability_estimate
            })
            
        return scenarios

if __name__ == "__main__":
    # Dummy forecast outputs ( mimicking the structure from STSModel.forecast() )
    num_samples = 1000
    num_years_forecast = 10
    start_year = 2025

    dummy_forecast_agi = {
        "samples": np.random.normal(loc=2035, scale=5, size=(num_samples, num_years_forecast)),
        "forecast_years": np.arange(start_year, start_year + num_years_forecast)
    }
    dummy_forecast_lev = { # Longevity Escape Velocity
        "samples": np.random.normal(loc=2040, scale=7, size=(num_samples, num_years_forecast)),
        "forecast_years": np.arange(start_year, start_year + num_years_forecast)
    }
    
    all_forecasts_example = {
        "AGI_Achievement_Year_Prediction": dummy_forecast_agi,
        "Longevity_Escape_Velocity_Year_Prediction": dummy_forecast_lev
    }
    
    try:
        scenario_gen = ScenarioGenerator(all_forecasts_example)
        generated_scenarios = scenario_gen.generate_scenarios(num_scenarios=5)
        
        print(f"--- Generated {len(generated_scenarios)} Scenarios ---")
        for scn in generated_scenarios:
            print(f"\nScenario ID: {scn['scenario_id']}")
            print(f"  Probability Estimate (simplistic): {scn['probability_estimate']:.4f}")
            for event_name, event_details in scn["events"].items():
                print(f"  Event: {event_name}")
                # Limiting output for brevity
                limited_event_details = {k: v for i, (k, v) in enumerate(event_details.items()) if i < 3}
                for year_key, val in limited_event_details.items():
                     print(f"    {year_key}: {val:.2f}")
                if len(event_details) > 3:
                    print("    ...")

        # Test with num_scenarios > num_samples
        print("\n--- Testing with num_scenarios > num_samples ---")
        generated_scenarios_more = scenario_gen.generate_scenarios(num_scenarios=1005)
        print(f"Generated {len(generated_scenarios_more)} scenarios (should be {num_samples}).")
        if generated_scenarios_more:
             print(f"First scenario ID from this batch: {generated_scenarios_more[0]['scenario_id']}")   

        # Test with empty forecast_outputs
        print("\n--- Testing with empty forecast_outputs ---")
        empty_gen = ScenarioGenerator({})
        empty_scenarios = empty_gen.generate_scenarios()
        print(f"Generated {len(empty_scenarios)} scenarios from empty input (should be 0).")

    except (TypeError, ValueError) as e:
        print(f"Error during ScenarioGenerator initialization or use: {e}")
