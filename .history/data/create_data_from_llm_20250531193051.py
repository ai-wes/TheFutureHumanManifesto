import json
import os
import time
import requests
import re
import aiohttp
import asyncio

# --- Configuration ---

# --- API Configuration ---
API_URL = "https://server.build-a-bf.com/v1/chat/completions"
MODEL_NAME = "deepseek-r1-0528-qwen3-8b"
MAX_TOKENS = -1  # -1 for unlimited, as per curl example
TEMPERATURE = 0.7

# --- File Paths ---
INPUT_JSON_FILE = "synthetic_scenarios_generated.json"
OUTPUT_JSON_FILE = "refined_scenarios_briefs.json"
PROGRESS_FILE = "processing_progress.json"

# --- Script Settings ---
SAVE_INTERVAL = 3
RETRY_DELAY = 60
MAX_RETRIES = 5

# --- System Prompt (as refined) ---
SYSTEM_PROMPT_TEMPLATE = """
You are a Lead Scenario Architect, tasked with developing a concise Executive Scenario Brief from simulated future data. Your goal is to distill the essence of the scenario, highlighting its core narrative, fundamental drivers, defining dynamics, key transformations, and strategic implications.

Original Timeline: {start_year}-{end_year}

Original Key Events Log (Raw log of occurrences; identify the truly *transformative turning points* that shape the narrative):
{key_events_list_as_string}

Original Derived Technological Factors (Initial observations):
{tech_factors_list_as_string}

Original Derived Social Factors (Initial observations):
{social_factors_list_as_string}

Original Derived Economic Factors (Initial observations):
{economic_factors_list_as_string}

Tasks:

1.  **Core Narrative Turning Points (3-5 points):**
    *   Identify 3-5 pivotal *turning points* from the 'Original Key Events Log' that collectively define this scenario's core narrative trajectory and its defining shifts.
    *   For each turning point, craft a concise statement that not only describes the event but also alludes to its immediate significant consequence or the shift it represents.
    *   These turning points should flow logically, telling an interconnected story of the scenario's evolution.

2.  **Core Technological Drivers & Implications (2-4 drivers):**
    *   Distill 2-4 *fundamental technological drivers* that underpin the scenario's technological landscape.
    *   For each driver, provide:
        *   `driver`: A concise statement defining the technological driver.
        *   `implication`: A brief statement on its primary strategic implication (e.g., for society, economy, governance).

3.  **Defining Social Dynamics & Implications (2-3 dynamics):**
    *   Articulate 2-3 *defining social dynamics* that characterize the societal fabric, prevailing attitudes, or structural changes.
    *   For each dynamic, provide:
        *   `dynamic`: A concise statement defining the social dynamic.
        *   `implication`: A brief statement on its primary strategic implication.

4.  **Key Economic Transformations & Implications (2-3 transformations):**
    *   Identify 2-3 *key economic transformations* that describe major shifts in economic structures, value creation, or resource paradigms.
    *   For each transformation, provide:
        *   `transformation`: A concise statement defining the economic transformation.
        *   `implication`: A brief statement on its primary strategic implication.

5.  **Strategic Coherence & Core Challenge Overview:**
    *   Provide a concise strategic overview assessing the scenario's internal coherence and plausibility.
    *   Crucially, identify and articulate the scenario's *defining strategic challenge, core tension, or central dilemma* that emerges from the interplay of these events and factors.

Output the refined information in the following JSON format:
{{
  "core_narrative_turning_points": [
    "Statement for turning point 1, highlighting shift/consequence.",
    "Statement for turning point 2, highlighting shift/consequence."
  ],
  "core_technological_drivers": [
    {{"driver": "Statement of driver 1.", "implication": "Primary strategic implication of driver 1."}}
  ],
  "defining_social_dynamics": [
    {{"dynamic": "Statement of dynamic 1.", "implication": "Primary strategic implication of dynamic 1."}}
  ],
  "key_economic_transformations": [
    {{"transformation": "Statement of transformation 1.", "implication": "Primary strategic implication of transformation 1."}}
  ],
  "strategic_coherence_overview": "Your strategic overview, including the defining challenge/tension/dilemma."
}}
"""

def format_list_for_prompt(data_list):
    """Formats a list into a newline-separated string for the prompt."""
    if not data_list:
        return "N/A"
    return "\n".join([f"- {item}" for item in data_list])

def format_system_prompt(template, scenario_obj):
    """Injects scenario data into the system prompt template."""
    start_year, end_year = scenario_obj.get("timeline", "Unknown-Unknown").split('-')
    return template.format(
        start_year=start_year,
        end_year=end_year,
        key_events_list_as_string=format_list_for_prompt(scenario_obj.get("key_events", [])),
        tech_factors_list_as_string=format_list_for_prompt(scenario_obj.get("technological_factors", [])),
        social_factors_list_as_string=format_list_for_prompt(scenario_obj.get("social_factors", [])),
        economic_factors_list_as_string=format_list_for_prompt(scenario_obj.get("economic_factors", []))
    )

def strip_think_tags(text):
    """
    Removes <think>...</think> blocks from the text, including multiline content.
    """
    # Remove all <think>...</think> blocks, including newlines and any content inside
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def extract_json_from_text(text):
    """
    Extracts the first JSON object from a string, ignoring any non-JSON text.
    """
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    return None

def get_chat_completion(system_prompt_content, scenario_id):
    """
    Submits the prompt to the local API (https://server.build-a-bf.com/v1/chat/completions).
    Includes retry logic for errors.
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            print(f"  Attempting API call for scenario ID: {scenario_id} (Attempt {retries + 1})")
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": "Please output the JSON in the specified format ONLY, nothing else."}
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300)
            if response.status_code != 200:
                print(f"    API returned status code {response.status_code}: {response.text}")
                retries += 1
                time.sleep(RETRY_DELAY)
                continue
            response_json = response.json()
            # The model's output is expected in response_json['choices'][0]['message']['content']
            response_content = response_json['choices'][0]['message']['content']
            return response_content
        except requests.exceptions.RequestException as e:
            print(f"    Request error for scenario ID {scenario_id}: {e}. Retrying in {RETRY_DELAY} seconds...")
            retries += 1
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"    An unexpected error occurred for scenario ID {scenario_id}: {e}")
            return None
    print(f"  Failed to get completion for scenario ID {scenario_id} after {MAX_RETRIES} retries.")
    return None

def load_progress():
    """Loads the set of already processed scenario IDs."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            print(f"Warning: Progress file {PROGRESS_FILE} is corrupted. Starting fresh.")
            return set()
    return set()

def save_progress(processed_ids):
    """Saves the set of processed scenario IDs."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(processed_ids), f)
    print(f"Progress saved. {len(processed_ids)} items processed so far.")

def save_results(results_list, filepath):
    """Saves the list of results to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(results_list)} refined scenarios to {filepath}")
    except Exception as e:
        print(f"Error saving results to {filepath}: {e}")

async def process_scenario_with_endpoints(system_prompt, scenario_id):
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_chat_completion_async(session, endpoint, system_prompt, scenario_id)
            for endpoint in ENDPOINTS
        ]
        results = await asyncio.gather(*tasks)
        return results  # List of LLM outputs from both endpoints

def main():
    print("Starting scenario refinement process...")

    # Load input data
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            input_scenarios = json.load(f)
        print(f"Loaded {len(input_scenarios)} scenarios from {INPUT_JSON_FILE}")
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_JSON_FILE} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Input file {INPUT_JSON_FILE} is not valid JSON.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading input data: {e}")
        return

    if not isinstance(input_scenarios, list):
        print(f"Error: Expected a list of scenarios in {INPUT_JSON_FILE}, but got {type(input_scenarios)}.")
        return

    processed_ids = load_progress()
    all_refined_scenarios = []

    # Load previously processed results if output file exists, to append new ones
    if os.path.exists(OUTPUT_JSON_FILE):
        try:
            with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
                all_refined_scenarios = json.load(f)
            print(f"Loaded {len(all_refined_scenarios)} previously refined scenarios from {OUTPUT_JSON_FILE}")
        except json.JSONDecodeError:
            print(f"Warning: Output file {OUTPUT_JSON_FILE} is corrupted. Starting with an empty list.")
            all_refined_scenarios = []
        except Exception as e:
            print(f"Could not load existing output file {OUTPUT_JSON_FILE}: {e}. Starting with an empty list.")
            all_refined_scenarios = []

    total_scenarios = len(input_scenarios)
    processed_count = 0

    for i, scenario_obj in enumerate(input_scenarios):
        scenario_id = scenario_obj.get("id", f"unknown_id_{i}")
        print(f"\nProcessing scenario {i + 1}/{total_scenarios} (ID: {scenario_id})...")

        if scenario_id in processed_ids:
            print(f"  Scenario ID {scenario_id} already processed. Skipping.")
            continue

        if not isinstance(scenario_obj, dict):
            print(f"  Skipping item {i+1} as it is not a valid dictionary: {scenario_obj}")
            continue

        # Construct the system prompt with injected data
        system_prompt = format_system_prompt(SYSTEM_PROMPT_TEMPLATE, scenario_obj)

        # Get completion from API
        raw_llm_output = get_chat_completion(system_prompt, scenario_id)

        if raw_llm_output:
            # Remove <think>...</think> blocks before attempting to parse JSON
            cleaned_llm_output = strip_think_tags(raw_llm_output)
            json_str = extract_json_from_text(cleaned_llm_output)
            if json_str:
                try:
                    refined_data = json.loads(json_str)
                    output_item = {
                        "original_scenario_id": scenario_id,
                        "original_scenario_data_preview": {
                            "timeline": scenario_obj.get("timeline"),
                            "key_events_count": len(scenario_obj.get("key_events", [])),
                            "target_probability_synthetic": scenario_obj.get("target_probability_synthetic")
                        },
                        "refined_executive_brief": refined_data,
                        "llm_model_used": MODEL_NAME,
                        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                    }
                    all_refined_scenarios.append(output_item)
                    processed_ids.add(scenario_id)
                    processed_count += 1
                    print(f"  Successfully processed and parsed LLM output for scenario ID: {scenario_id}")

                except json.JSONDecodeError:
                    print(f"  Error: LLM output for scenario ID {scenario_id} was not valid JSON.")
                    print(f"  Raw LLM Output:\n{raw_llm_output}")
                    with open(f"error_output_{scenario_id}.txt", "w", encoding="utf-8") as err_f:
                        err_f.write(f"System Prompt:\n{system_prompt}\n\nRaw LLM Output:\n{raw_llm_output}")
                except Exception as e:
                    print(f"  An unexpected error occurred while processing LLM output for {scenario_id}: {e}")
            else:
                print(f"  Error: No JSON found in LLM output for scenario ID {scenario_id}.")
                with open(f"error_output_{scenario_id}.txt", "w", encoding="utf-8") as err_f:
                    err_f.write(f"System Prompt:\n{system_prompt}\n\nRaw LLM Output:\n{raw_llm_output}")
        else:
            print(f"  Failed to get LLM output for scenario ID: {scenario_id}. Skipping.")

        # Periodic saving
        if processed_count > 0 and (processed_count % SAVE_INTERVAL == 0 or (i + 1) == total_scenarios):
            print(f"\n--- Saving progress and results ({processed_count} new items processed since last save) ---")
            save_results(all_refined_scenarios, OUTPUT_JSON_FILE)
            save_progress(processed_ids)
            print("--- Save complete ---\n")

    print("\nScenario refinement process finished.")
    # Final save
    if processed_count > 0:
        print("Performing final save of all results and progress...")
        save_results(all_refined_scenarios, OUTPUT_JSON_FILE)
        save_progress(processed_ids)
        print("Final save complete.")
    else:
        print("No new scenarios were processed in this run.")

if __name__ == "__main__":
    main()