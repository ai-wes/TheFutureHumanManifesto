      
import json
import os
import time
import re
import aiohttp
import asyncio
import logging

# Suppress excessive aiohttp connection logging if not needed for debugging
logging.getLogger("aiohttp.connector").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Configuration ---
# Define a list of endpoints. If only one, it will always use that.
# These should be the endpoints for your PLAUSIBILITY SCORING LLM.
ENDPOINTS = [
    "https://server.build-a-bf.com/v1/chat/completions", # Primary
    "http://localhost:1234/v1/chat/completions"          # Secondary/Local, if available
    # Add more endpoints if you have them
]
# Ensure PLAUSIBILITY_API_URL is not used directly if ENDPOINTS list is used for selection.

# Model for plausibility scoring (ideally a powerful one like GPT-4 or equivalent)
PLAUSIBILITY_MODEL_NAME = "deepseek-r1-0528-qwen3-8b" # Or your preferred model for this task

MAX_TOKENS_PLAUSIBILITY = -1 # Or 512 if -1 causes issues with your specific API
TEMPERATURE_PLAUSIBILITY = 0.2 # Lower temperature for more deterministic scoring

# --- File Paths ---
INPUT_REFINED_BRIEFS_FILE = r"F:\TheFutureHumanManifesto\data\refined_scenarios_briefs_intermediate.json"
OUTPUT_JSON_FILE_WITH_PLAUSIBILITY = r"F:\TheFutureHumanManifesto\data\refined_briefs_with_plausibility.json"
PROGRESS_FILE_PLAUSIBILITY = "plausibility_processing_progress.json"
ERROR_LOG_DIR = "llm_error_logs"
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

# --- Script Settings ---
SAVE_INTERVAL = 2
RETRY_DELAY = 60
MAX_API_RETRIES = 3
CONCURRENCY = 8 # Max concurrent requests overall, distributed among endpoints by the logic

# --- System Prompt for Plausibility Assessment --- (Keep as is)
PLAUSIBILITY_SYSTEM_PROMPT_TEMPLATE = """
You are an expert futurist and quantitative foresight analyst. Your task is to assess the overall plausibility and likelihood of the following refined future scenario brief.

Scenario Brief:
Timeline: {timeline_str}

Core Narrative Turning Points:
{core_narrative_turning_points_str}

Core Technological Drivers & Implications:
{core_technological_drivers_str}

Defining Social Dynamics & Implications:
{defining_social_dynamics_str}

Key Economic Transformations & Implications:
{key_economic_transformations_str}

Strategic Coherence Overview:
{strategic_coherence_overview_text}

Defining Strategic Challenge:
{defining_strategic_challenge_text}
---
Instructions:

Based on your expert judgment, considering the internal consistency, the scale and scope of transformations, the interplay of drivers and dynamics, and general real-world plausibility for the given timeframe:

1.  Provide a "plausibility_score" for this scenario occurring as described, on a scale from 0.0 (highly implausible, self-contradictory, or fantastical) to 1.0 (highly plausible, coherent, and a well-reasoned possibility for the future). The score should be a float.
2.  Provide a brief (1-2 sentences) "reasoning" for your score, highlighting key strengths or weaknesses that influenced your judgment.

YOUR ONLY OUTPUT MUST BE A JSON OBJECT WITH THE FOLLOWING KEYS:
{{
  "plausibility_score": <float_between_0.0_and_1.0>,
  "reasoning": "Your brief reasoning here."
}}
"""

# --- Helper Functions --- (format_list_for_prompt, format_dict_list_for_prompt, format_plausibility_system_prompt, strip_think_tags, extract_json_from_text, make_api_call_async, load_json_file, save_json_file, log_llm_error - Keep these as they are from your previous script)
def format_list_for_prompt(data_list, indent=False):
    if not data_list:
        return "N/A"
    prefix = "  - " if indent else "- "
    return "\n".join([f"{prefix}{item}" for item in data_list])


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

def format_dict_list_for_prompt(dict_list, key1_name, key2_name):
    if not dict_list:
        return "N/A"
    formatted_items = []
    for item in dict_list:
        formatted_items.append(f"  - {key1_name.capitalize()}: {item.get(key1_name, 'N/A')}\n    Implication: {item.get(key2_name, 'N/A')}")
    return "\n".join(formatted_items)

def format_plausibility_system_prompt(template, refined_brief_item):
    brief_content = refined_brief_item.get("refined_executive_brief", {})
    original_preview = refined_brief_item.get("original_scenario_data_preview", {})
    return template.format(
        timeline_str=original_preview.get("timeline", "N/A"),
        core_narrative_turning_points_str=format_list_for_prompt(brief_content.get("core_narrative_turning_points", []), indent=True),
        core_technological_drivers_str=format_dict_list_for_prompt(brief_content.get("core_technological_drivers", []), 'driver', 'implication'),
        defining_social_dynamics_str=format_dict_list_for_prompt(brief_content.get("defining_social_dynamics", []), 'dynamic', 'implication'),
        key_economic_transformations_str=format_dict_list_for_prompt(brief_content.get("key_economic_transformations", []), 'transformation', 'implication'),
        strategic_coherence_overview_text=brief_content.get("strategic_coherence_overview", "N/A"),
        defining_strategic_challenge_text=brief_content.get("defining_strategic_challenge", "N/A")
    )

def strip_think_tags(text):
    if not text: return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def extract_json_from_text(text):
    if not text: return None
    match = re.search(r'^\s*(\{[\s\S]*\})\s*$', text)
    if match: return match.group(1)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        logger.warning("Extracted JSON was embedded in other text. Attempting to parse.")
        return match.group(0)
    return None

async def make_api_call_async(session, api_url, model_name, system_prompt_content, user_prompt_content, max_tokens, temperature, scenario_id, call_type="plausibility"):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens != -1 else None,
        "stream": False
    }
    if payload["max_tokens"] is None: del payload["max_tokens"]
    headers = {"Content-Type": "application/json"}
    retries = 0
    response_text_for_error = ""
    while retries < MAX_API_RETRIES:
        try:
            logger.debug(f"  Attempting API call ({call_type}) for scenario ID: {scenario_id} to {api_url} (Attempt {retries + 1})")
            async with session.post(api_url, headers=headers, json=payload, timeout=300) as response:
                response_text_for_error = await response.text()
                if response.status != 200:
                    logger.error(f"    API ({call_type}) returned status code {response.status} for {scenario_id}: {response_text_for_error[:500]}")
                    retries += 1
                    if response.status == 429:
                        wait_time = RETRY_DELAY * (retries)
                        logger.info(f"    Rate limit hit. Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    else: await asyncio.sleep(RETRY_DELAY)
                    continue
                response_json = json.loads(response_text_for_error)
                content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                if not content: logger.warning(f"    API ({call_type}) response for {scenario_id} had no content. Full response: {response_json}")
                return content
        except aiohttp.ClientConnectorError as e: logger.error(f"    Connection error for API call ({call_type}) for {scenario_id} to {api_url}: {e}. Retrying in {RETRY_DELAY}s...")
        except asyncio.TimeoutError: logger.error(f"    Timeout error for API call ({call_type}) for {scenario_id} to {api_url}. Retrying in {RETRY_DELAY}s...")
        except json.JSONDecodeError as e:
            logger.error(f"    JSONDecodeError for API call ({call_type}) for {scenario_id} from {api_url}: {e}. Response text: {response_text_for_error[:500]}")
            return None
        except Exception as e: logger.error(f"    Unexpected error during API call ({call_type}) for {scenario_id} to {api_url}: {e}. Retrying in {RETRY_DELAY}s...")
        retries += 1
        await asyncio.sleep(RETRY_DELAY)
    logger.error(f"  Failed API call ({call_type}) for scenario ID {scenario_id} to {api_url} after {MAX_API_RETRIES} retries.")
    return None

def load_json_file(filepath, default_value):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except json.JSONDecodeError: logger.warning(f"File {filepath} is corrupted or not valid JSON. Using default: {default_value}")
        except Exception as e: logger.error(f"Could not load file {filepath}: {e}. Using default: {default_value}")
    else: logger.info(f"File {filepath} not found. Using default: {default_value}")
    return default_value

def save_json_file(data, filepath):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(data)} items to {filepath}")
    except Exception as e: logger.error(f"Error saving data to {filepath}: {e}")

def log_llm_error(scenario_id, call_type, system_prompt, raw_output):
    error_filename = os.path.join(ERROR_LOG_DIR, f"error_{call_type}_{scenario_id}_{time.strftime('%Y%m%d%H%M%S')}.txt")
    try:
        with open(error_filename, "w", encoding="utf-8") as err_f:
            err_f.write(f"--- System Prompt ({call_type}) ---\n{system_prompt}\n\n")
            err_f.write(f"--- Raw LLM Output ---\n{raw_output}\n")
        logger.info(f"Logged LLM error details to {error_filename}")
    except Exception as e: logger.error(f"Could not write LLM error file {error_filename}: {e}")

async def process_brief_for_plausibility_task(session, endpoint_url, refined_brief_item): # Changed 'endpoint' to 'endpoint_url' for clarity
    scenario_id = refined_brief_item.get("original_scenario_id", "unknown_id_in_brief")
    system_prompt = format_plausibility_system_prompt(PLAUSIBILITY_SYSTEM_PROMPT_TEMPLATE, refined_brief_item)
    user_prompt = "Please output the JSON in the specified format ONLY, nothing else, containing only 'plausibility_score' and 'reasoning'."
    
    raw_llm_output = await make_api_call_async(
        session, endpoint_url, PLAUSIBILITY_MODEL_NAME, # Pass endpoint_url here
        system_prompt, user_prompt, 
        MAX_TOKENS_PLAUSIBILITY, TEMPERATURE_PLAUSIBILITY, 
        scenario_id, "plausibility"
    )

    updated_item = refined_brief_item.copy()

    if raw_llm_output:
        cleaned_llm_output = strip_think_tags(raw_llm_output)
        json_str = extract_json_from_text(cleaned_llm_output)
        if json_str:
            try:
                plausibility_data = json.loads(json_str)
                updated_item['llm_assigned_plausibility'] = plausibility_data.get('plausibility_score')
                updated_item['llm_plausibility_reasoning'] = plausibility_data.get('reasoning')
                updated_item['llm_model_used_for_plausibility'] = PLAUSIBILITY_MODEL_NAME
                updated_item['plausibility_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                logger.info(f"  Successfully scored plausibility for scenario ID: {scenario_id} using {endpoint_url}")
            except json.JSONDecodeError:
                logger.error(f"  Error: Plausibility LLM output for scenario ID {scenario_id} (using {endpoint_url}) was not valid JSON.")
                log_llm_error(scenario_id, f"plausibility_{endpoint_url.split('/')[-1]}", system_prompt, raw_llm_output)
                updated_item['llm_assigned_plausibility'] = None
                updated_item['llm_plausibility_reasoning'] = "Error: LLM output was not valid JSON."
        else:
            logger.error(f"  Error: No JSON found in plausibility LLM output for scenario ID {scenario_id} (using {endpoint_url}).")
            log_llm_error(scenario_id, f"plausibility_{endpoint_url.split('/')[-1]}", system_prompt, raw_llm_output)
            updated_item['llm_assigned_plausibility'] = None
            updated_item['llm_plausibility_reasoning'] = "Error: No JSON found in LLM output."
    else:
        updated_item['llm_assigned_plausibility'] = None
        updated_item['llm_plausibility_reasoning'] = f"Error: Failed to get LLM response from {endpoint_url} after retries."
        
    return scenario_id, updated_item


async def main_async_pipeline():
    logger.info("--- Starting Plausibility Scoring Stage ---")

    refined_briefs_input = load_json_file(INPUT_REFINED_BRIEFS_FILE, [])
    if not refined_briefs_input:
        logger.error(f"Input file {INPUT_REFINED_BRIEFS_FILE} is empty or could not be loaded. Exiting.")
        if not os.path.exists(INPUT_REFINED_BRIEFS_FILE):
            logger.info(f"Creating dummy {INPUT_REFINED_BRIEFS_FILE} for testing.")
            dummy_refined_data = [
                {
                    "original_scenario_id": "dummy_refined_1",
                    "original_scenario_data_preview": {"timeline": "2030-2040"},
                    "refined_executive_brief": {
                        "core_narrative_turning_points": ["Dummy TP 1"],
                        "core_technological_drivers": [{"driver": "Dummy Driver 1", "implication": "Dummy Implication 1"}],
                        "defining_social_dynamics": [{"dynamic": "Dummy Dynamic 1", "implication": "Dummy Implication 1"}],
                        "key_economic_transformations": [{"transformation": "Dummy Transformation 1", "implication": "Dummy Implication 1"}],
                        "strategic_coherence_overview": "Dummy overview.",
                        "defining_strategic_challenge": "Dummy challenge."
                    },
                    "llm_model_used_for_refinement": "dummy_refiner_model",
                    "refinement_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                }
            ]
            save_json_file(dummy_refined_data, INPUT_REFINED_BRIEFS_FILE)
            refined_briefs_input = dummy_refined_data
        else: return

    if not isinstance(refined_briefs_input, list):
        logger.error(f"Input file {INPUT_REFINED_BRIEFS_FILE} must contain a list of refined brief objects.")
        return

    processed_ids_plausibility = set(load_json_file(PROGRESS_FILE_PLAUSIBILITY, []))
    final_results_map = {item['original_scenario_id']: item for item in load_json_file(OUTPUT_JSON_FILE_WITH_PLAUSIBILITY, [])}
    
    items_to_score = []
    for brief_item in refined_briefs_input:
        item_id = brief_item.get("original_scenario_id")
        if not item_id:
            logger.warning(f"Skipping item due to missing 'original_scenario_id': {str(brief_item)[:100]}")
            continue
        if item_id not in processed_ids_plausibility:
            items_to_score.append(brief_item)
        elif item_id not in final_results_map:
             final_results_map[item_id] = brief_item

    if not items_to_score:
        logger.info("No new refined briefs to score for plausibility.")
        if not os.path.exists(PROGRESS_FILE_PLAUSIBILITY) or os.path.getsize(PROGRESS_FILE_PLAUSIBILITY) < 5:
            if final_results_map: save_json_file(list(final_results_map.values()), OUTPUT_JSON_FILE_WITH_PLAUSIBILITY)
        return

    logger.info(f"Found {len(items_to_score)} refined briefs for plausibility scoring.")

    tasks_iterated_count = 0
    successfully_scored_this_run = 0

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(CONCURRENCY)

        tasks = []
        for i, brief_item_for_task in enumerate(items_to_score):
            # Round-robin through the ENDPOINTS list
            endpoint_url_for_task = ENDPOINTS[i % len(ENDPOINTS)]
            
            async def sem_task_wrapper(item, url): # Need to capture item and url in closure
                async with semaphore:
                    return await process_brief_for_plausibility_task(session, url, item)
            
            tasks.append(sem_task_wrapper(brief_item_for_task, endpoint_url_for_task))

        logger.info(f"Submitting {len(tasks)} tasks for plausibility scoring across {len(ENDPOINTS)} endpoint(s)...")

        for future in asyncio.as_completed(tasks):
            item_id_processed, updated_brief_data = await future
            tasks_iterated_count += 1

            if updated_brief_data:
                final_results_map[item_id_processed] = updated_brief_data
                if updated_brief_data.get("llm_assigned_plausibility") is not None and \
                   item_id_processed not in processed_ids_plausibility:
                    processed_ids_plausibility.add(item_id_processed)
                    successfully_scored_this_run += 1
            
            if successfully_scored_this_run > 0 and (tasks_iterated_count % SAVE_INTERVAL == 0):
                logger.info(f"\n--- Periodic Save (Plausibility): Iterated {tasks_iterated_count} tasks. Saving progress and results. ---")
                save_json_file(list(final_results_map.values()), OUTPUT_JSON_FILE_WITH_PLAUSIBILITY)
                save_json_file(list(processed_ids_plausibility), PROGRESS_FILE_PLAUSIBILITY)
                logger.info(f"--- Periodic Save (Plausibility) complete ---\n")
    
    logger.info(f"Plausibility scoring stage finished. {successfully_scored_this_run} items newly scored.")
    if successfully_scored_this_run > 0 or (not os.path.exists(OUTPUT_JSON_FILE_WITH_PLAUSIBILITY) and final_results_map):
        logger.info(f"Performing final save for plausibility scoring stage...")
        save_json_file(list(final_results_map.values()), OUTPUT_JSON_FILE_WITH_PLAUSIBILITY)
        save_json_file(list(processed_ids_plausibility), PROGRESS_FILE_PLAUSIBILITY)
        logger.info(f"Final save for plausibility scoring complete.")

def main():
    asyncio.run(main_async_pipeline())

if __name__ == "__main__":
    main()

    