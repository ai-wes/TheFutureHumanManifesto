import json
import os
import time
import requests  # Keep for synchronous fallback or other uses if any
import re
import aiohttp
import asyncio
import logging

logging.getLogger("aiohttp.connector").setLevel(logging.CRITICAL)

# --- Configuration ---

# --- API Configuration ---
API_URL = "https://server.build-a-bf.com/v1/chat/completions"
MODEL_NAME = "deepseek-r1-0528-qwen3-8b"
MAX_TOKENS = -1  # -1 for unlimited, as per curl example
TEMPERATURE = 0.6

# --- File Paths ---
INPUT_JSON_FILE = r"F:\TheFutureHumanManifesto\data\synthetic_scenarios_generated.json"
OUTPUT_JSON_FILE = r"F:\TheFutureHumanManifesto\data\refined_scenarios_briefs.json"
PROGRESS_FILE = "processing_progress.json"

# --- Script Settings ---
SAVE_INTERVAL = 2
RETRY_DELAY = 60
MAX_RETRIES = 5

# --- System Prompt (as refined) ---
SYSTEM_PROMPT_TEMPLATE = """
You are a Lead Scenario Architect, tasked with developing a concise Executive Scenario Brief from simulated future data. Your goal is to distill the essence of the scenario, highlighting its core narrative, fundamental drivers, defining dynamics, key transformations, strategic implications, and its central challenge.

Original Timeline: {start_year}-{end_year}

Original Key Events Log (Raw log of occurrences; identify the truly *transformative turning points* that shape the narrative):
{key_events_list_as_string}

Original Derived Technological Factors (Initial observations):
{original_tech_factors_list_as_string}

Original Derived Social Factors (Initial observations):
{original_social_factors_list_as_string}

Original Derived Economic Factors (Initial observations):
{original_economic_factors_list_as_string}

Tasks:

1.  **Core Narrative Turning Points (3-5 points):**
    *   Identify 3-5 pivotal *turning points* from the 'Original Key Events Log' that collectively define this scenario's core narrative trajectory and its defining shifts.
    *   For each turning point, craft a concise statement (ideally without explicitly stating the year unless crucial for context) that not only describes the event but also alludes to its immediate significant consequence or the shift it represents.
    *   These turning points should flow logically, telling an interconnected story of the scenario's evolution.

2.  **Core Technological Drivers & Implications (Aim for 3, range 2-4 drivers):**
    *   Distill 2-4 (aiming for around 3 if the data supports it) *fundamental technological drivers* that underpin the scenario's technological landscape.
    *   For each driver, provide:
        *   `driver`: A concise statement defining the technological driver.
        *   `implication`: A brief statement on its primary strategic implication (e.g., for society, economy, governance).

3.  **Defining Social Dynamics & Implications (Aim for 2-3 dynamics):**
    *   Articulate 2-3 *defining social dynamics* that characterize the societal fabric, prevailing attitudes, or structural changes.
    *   For each dynamic, provide:
        *   `dynamic`: A concise statement defining the social dynamic.
        *   `implication`: A brief statement on its primary strategic implication.

4.  **Key Economic Transformations & Implications (Aim for 2-3 transformations):**
    *   Identify 2-3 *key economic transformations* that describe major shifts in economic structures, value creation, or resource paradigms.
    *   For each transformation, provide:
        *   `transformation`: A concise statement defining the economic transformation.
        *   `implication`: A brief statement on its primary strategic implication.

5.  **Strategic Coherence Overview & Defining Challenge:**
    *   `strategic_coherence_overview`: Provide a concise strategic overview assessing the scenario's internal coherence and plausibility, noting any complex interplay of conflicting trends.
    *   `defining_strategic_challenge`: Separately and clearly articulate the scenario's *single most defining strategic challenge, core tension, or central dilemma* that emerges from the interplay of these events and factors.

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

# --- Helper Functions ---

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
        original_tech_factors_list_as_string=format_list_for_prompt(scenario_obj.get("technological_factors", [])),
        original_social_factors_list_as_string=format_list_for_prompt(scenario_obj.get("social_factors", [])),
        original_economic_factors_list_as_string=format_list_for_prompt(scenario_obj.get("economic_factors", []))
    )

def strip_think_tags(text):
    """
    Removes <think>...</think> blocks from the text, including multiline content.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def extract_json_from_text(text):
    """
    Extracts the first JSON object from a string, ignoring any non-JSON text.
    """
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    return None

# --- Async API Call ---

async def get_chat_completion_async(session, api_url, system_prompt_content, scenario_id):
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
    try:
        async with session.post(api_url, headers=headers, json=payload, timeout=300) as response:
            if response.status != 200:
                print(f"    API returned status code {response.status}: {await response.text()}")
                return None
            response_json = await response.json()
            return response_json['choices'][0]['message']['content']
    except Exception as e:
        print(f"    Request error for scenario ID {scenario_id} to {api_url}: {e}")
        return None

# --- Progress/Result Saving ---

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

ENDPOINTS = [
    "https://server.build-a-bf.com/v1/chat/completions",
    "http://localhost:1234/v1/chat/completions"
]
CONCURRENCY = 8

# --- Scenario Processing Function ---

async def process_and_refine_scenario(session, api_url, system_prompt_template_text, scenario_obj, scenario_id):
    system_prompt = format_system_prompt(system_prompt_template_text, scenario_obj)
    raw_llm_output = await get_chat_completion_async(session, api_url, system_prompt, scenario_id)

    if raw_llm_output:
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
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                }
                print(f"  Successfully processed and parsed LLM output for scenario ID: {scenario_id}")
                return scenario_id, output_item
            except json.JSONDecodeError:
                print(f"  Error: LLM output for scenario ID {scenario_id} was not valid JSON.")
                print(f"  Raw LLM Output (first 500 chars):\n{raw_llm_output[:500]}...")
                try:
                    with open(f"error_output_{scenario_id}.txt", "w", encoding="utf-8") as err_f:
                        err_f.write(f"System Prompt:\n{SYSTEM_PROMPT_TEMPLATE}\n\nRaw LLM Output:\n{raw_llm_output}")
                except Exception as e:
                    print(f"    Could not write error file for scenario ID {scenario_id}: {e}")
                return scenario_id, None
            except Exception as e:
                print(f"  An unexpected error occurred while processing LLM output for {scenario_id}: {e}")
                return scenario_id, None
        else:
            print(f"  Error: No JSON found in LLM output for scenario ID {scenario_id}.")
            try:
                with open(f"error_output_{scenario_id}.txt", "w", encoding="utf-8") as err_f:
                    err_f.write(f"System Prompt:\n{SYSTEM_PROMPT_TEMPLATE}\n\nRaw LLM Output:\n{raw_llm_output}")
            except Exception as e:
                print(f"    Could not write error file for scenario ID {scenario_id}: {e}")
            return scenario_id, None
    else:
        print(f"  Failed to get LLM output for scenario ID: {scenario_id}. Skipping.")
        return scenario_id, None

# --- Main Async Logic ---

async def main_async_logic():
    print("Starting scenario refinement process...")

    # --- Load input data, progress, and existing results (same as before) ---
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            input_scenarios = json.load(f)
        print(f"Loaded {len(input_scenarios)} scenarios from {INPUT_JSON_FILE}")
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_JSON_FILE} not found.")
        return [], set(), 0
    except json.JSONDecodeError:
        print(f"Error: Input file {INPUT_JSON_FILE} is not valid JSON.")
        return [], set(), 0
    except Exception as e:
        print(f"An unexpected error occurred while loading input data: {e}")
        return [], set(), 0

    if not isinstance(input_scenarios, list):
        print(f"Error: Expected a list of scenarios in {INPUT_JSON_FILE}, but got {type(input_scenarios)}.")
        return [], set(), 0

    processed_ids = load_progress()
    all_refined_scenarios = []
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

    unprocessed_scenarios = [
        s for i, s in enumerate(input_scenarios)
        if s.get("id", f"unknown_id_{i}") not in processed_ids
    ]

    if not unprocessed_scenarios:
        print("No new scenarios to process.")
        return all_refined_scenarios, processed_ids, 0

    print(f"Found {len(unprocessed_scenarios)} new scenarios to process.")

    tasks_submitted_count = 0
    newly_processed_in_this_run = 0

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def sem_task_wrapper(scenario_obj, scenario_id, endpoint):
            async with semaphore:
                return await process_and_refine_scenario(session, endpoint, SYSTEM_PROMPT_TEMPLATE, scenario_obj, scenario_id)

        tasks = []
        for i, scenario_obj in enumerate(unprocessed_scenarios):
            scenario_id = scenario_obj.get("id", f"unknown_id_for_unprocessed_{i}")
            endpoint = ENDPOINTS[i % len(ENDPOINTS)]
            tasks.append(sem_task_wrapper(scenario_obj, scenario_id, endpoint))
            tasks_submitted_count += 1

        print(f"\n--- Submitting {len(tasks)} tasks for processing ---")

        for future in asyncio.as_completed(tasks):
            processed_scenario_id, output_item = await future

            if output_item:
                if processed_scenario_id in processed_ids and any(item['original_scenario_id'] == processed_scenario_id for item in all_refined_scenarios):
                    print(f"  Scenario ID {processed_scenario_id} was already in processed_ids and results. Skipping duplicate append.")
                else:
                    all_refined_scenarios.append(output_item)
                    processed_ids.add(processed_scenario_id)
                    newly_processed_in_this_run += 1

            if newly_processed_in_this_run > 0 and (newly_processed_in_this_run % SAVE_INTERVAL == 0):
                print(f"\n--- Periodic Save: {newly_processed_in_this_run} new items successfully processed. Saving progress and results. ---")
                save_results(all_refined_scenarios, OUTPUT_JSON_FILE)
                save_progress(processed_ids)
                print("--- Periodic Save complete ---\n")

    return all_refined_scenarios, processed_ids, newly_processed_in_this_run

# --- Synchronous Main Wrapper ---

def main():
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Error: Input file {INPUT_JSON_FILE} not found. Please provide the required input file.")
        return

    final_refined_scenarios, final_processed_ids, total_newly_processed = asyncio.run(main_async_logic())

    print("\nScenario refinement process finished.")
    if total_newly_processed > 0:
        print("Performing final save of all results and progress...")
        save_results(final_refined_scenarios, OUTPUT_JSON_FILE)
        save_progress(final_processed_ids)
        print("Final save complete.")
    else:
        print("No new scenarios were successfully processed in this run.")

if __name__ == "__main__":
    main()