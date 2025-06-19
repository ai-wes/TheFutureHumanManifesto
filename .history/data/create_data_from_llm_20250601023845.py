import json
import os
import time
import re # Keep for strip_think_tags
import aiohttp
import asyncio
import logging

# Suppress excessive aiohttp connection logging if not needed for debugging
logging.getLogger("aiohttp.connector").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Configuration ---
# These should match the endpoints used by your refinement LLM
ENDPOINTS = [
    "https://server.build-a-bf.com/v1/chat/completions",
    "http://localhost:1234/v1/chat/completions"
]
MODEL_NAME = "deepseek-r1-0528-qwen3-8b" # Model for refinement
MAX_TOKENS = -1  # Or a specific number like 2048, if -1 causes issues
TEMPERATURE = 0.6

# --- File Paths ---
# INPUT: Your actual historical predictions dataset
INPUT_JSON_FILE = r"F:\TheFutureHumanManifesto\src\gaps_subsystem\historical_predictions.json"
# OUTPUT: File where refined briefs of historical predictions will be saved
OUTPUT_JSON_FILE = r"F:\TheFutureHumanManifesto\data\historical_refined_scenarios_briefs.json"
PROGRESS_FILE = "historical_refinement_processing_progress.json" # Progress for this specific script
ERROR_LOG_DIR = "llm_error_logs"
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

# --- Script Settings ---
SAVE_INTERVAL = 2
RETRY_DELAY = 60
MAX_API_RETRIES = 3
CONCURRENCY = 8

# --- System Prompt (MODIFIED for historical_predictions.json structure) ---
SYSTEM_PROMPT_TEMPLATE = """
You are a Lead Scenario Architect, tasked with developing a concise Executive Scenario Brief from an analysis of a past prediction. Your goal is to distill the essence of the original prediction, its context, and its outcome (if known), highlighting its core narrative elements, fundamental drivers, defining dynamics, key transformations, strategic implications, and any central challenges or lessons learned.

Original Prediction Details:
- Prediction ID: {prediction_id}
- Original Timeline (if specified in prediction): {original_timeline}
- Prediction Date: {prediction_date}
- Actual Outcome: {actual_outcome}
- Actual Outcome Date: {actual_outcome_date}

Original Key Events Predicted:
{key_events}

Original Technological Factors Considered:
{technological_factors}

Original Social Factors Considered:
{social_factors}

Original Economic Factors Considered:
{economic_factors}

Unforeseen Influencing Factors (Positive, if noted):
{unforeseen_influencing_factors_positive}

Unforeseen Influencing Factors (Negative, if noted):
{unforeseen_influencing_factors_negative}

Original Outcome Notes (if any):
{outcome_notes}
---
Tasks:

Based on the provided information about this past prediction:

1.  **Core Narrative Turning Points (3-5 points):**
    *   From the 'Original Key Events Predicted' and any outcome information, identify 3-5 pivotal *turning points* that defined the predicted scenario's trajectory or the actual unfolding of events if different and known.
    *   For each turning point, craft a concise statement that describes the event/shift and alludes to its significance.
    *   These turning points should form a coherent story.

2.  **Core Technological Drivers & Implications (Aim for 3, range 2-4 drivers):**
    *   Based on the 'Original Technological Factors' and any unforeseen factors, distill 2-4 *fundamental technological drivers* that underpinned the prediction or its outcome.
    *   For each driver: `driver` (statement), `implication` (strategic implication).

3.  **Defining Social Dynamics & Implications (Aim for 2-3 dynamics):**
    *   Based on 'Original Social Factors' and unforeseen factors, articulate 2-3 *defining social dynamics*.
    *   For each dynamic: `dynamic` (statement), `implication` (strategic implication).

4.  **Key Economic Transformations & Implications (Aim for 2-3 transformations):**
    *   Based on 'Original Economic Factors' and unforeseen factors, identify 2-3 *key economic transformations*.
    *   For each transformation: `transformation` (statement), `implication` (strategic implication).

5.  **Strategic Coherence Overview & Defining Challenge/Lesson:**
    *   `strategic_coherence_overview`: Provide a concise overview assessing the original prediction's coherence and plausibility in hindsight (if outcome is known) or its inherent logic. Note any interplay of conflicting trends.
    *   `defining_strategic_challenge`: Articulate the *single most defining strategic challenge, core tension, or key lesson learned* from this prediction and its outcome (or its analysis if outcome is pending).

Output the refined information in the following JSON format:
{{
  "core_narrative_turning_points": [
    "Statement for turning point 1, highlighting shift/consequence."
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
  "strategic_coherence_overview": "Your strategic overview.",
  "defining_strategic_challenge": "The single most defining strategic challenge or lesson learned."
}}
"""

def format_list_for_prompt(data_list):
    if not data_list:
        return "N/A"
    if isinstance(data_list, str): # Handle if a single string is passed by mistake
        return f"- {data_list}"
    return "\n".join([f"- {str(item)}" for item in data_list])

def format_system_prompt(template, scenario_obj): # MODIFIED for historical_predictions.json
    """Injects historical scenario data into the system prompt template."""
    metadata = scenario_obj.get("metadata", {})
    
    return template.format(
        prediction_id=scenario_obj.get("id", "N/A"),
        original_timeline=scenario_obj.get("timeline", "N/A"), # The 'timeline' field from the root
        prediction_date=metadata.get("prediction_date", "N/A"),
        actual_outcome=metadata.get("actual_outcome", "N/A (or Pending)"),
        actual_outcome_date=metadata.get("actual_outcome_date", "N/A"),
        key_events=format_list_for_prompt(scenario_obj.get("key_events", [])),
        technological_factors=format_list_for_prompt(scenario_obj.get("technological_factors", [])),
        social_factors=format_list_for_prompt(scenario_obj.get("social_factors", [])),
        economic_factors=format_list_for_prompt(scenario_obj.get("economic_factors", [])),
        unforeseen_influencing_factors_positive=format_list_for_prompt(metadata.get("unforeseen_influencing_factors_positive", [])),
        unforeseen_influencing_factors_negative=format_list_for_prompt(metadata.get("unforeseen_influencing_factors_negative", [])),
        outcome_notes=metadata.get("outcome_notes", "N/A")
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

async def make_api_call_async(session, api_url, model_name, system_prompt_content, user_prompt_content, max_tokens, temperature, item_id, call_type="refinement"):
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
            logger.debug(f"  Attempting API call ({call_type}) for item ID: {item_id} to {api_url} (Attempt {retries + 1})")
            async with session.post(api_url, headers=headers, json=payload, timeout=300) as response:
                response_text_for_error = await response.text()
                if response.status != 200:
                    logger.error(f"    API ({call_type}) returned status code {response.status} for {item_id}: {response_text_for_error[:500]}")
                    retries += 1
                    if response.status == 429:
                        wait_time = RETRY_DELAY * (retries)
                        logger.info(f"    Rate limit hit. Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    else: await asyncio.sleep(RETRY_DELAY)
                    continue
                response_json = json.loads(response_text_for_error)
                content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                if not content: logger.warning(f"    API ({call_type}) response for {item_id} had no content. Full response: {response_json}")
                return content
        except aiohttp.ClientConnectorError as e: logger.error(f"    Connection error for API call ({call_type}) for {item_id} to {api_url}: {e}. Retrying in {RETRY_DELAY}s...")
        except asyncio.TimeoutError: logger.error(f"    Timeout error for API call ({call_type}) for {item_id} to {api_url}. Retrying in {RETRY_DELAY}s...")
        except json.JSONDecodeError as e:
            logger.error(f"    JSONDecodeError for API call ({call_type}) for {item_id} from {api_url}: {e}. Response text: {response_text_for_error[:500]}")
            return None
        except Exception as e: logger.error(f"    Unexpected error during API call ({call_type}) for {item_id} to {api_url}: {e}. Retrying in {RETRY_DELAY}s...")
        retries += 1
        await asyncio.sleep(RETRY_DELAY)
    logger.error(f"  Failed API call ({call_type}) for item ID {item_id} to {api_url} after {MAX_API_RETRIES} retries.")
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

def log_llm_error(item_id, call_type, system_prompt, raw_output):
    error_filename = os.path.join(ERROR_LOG_DIR, f"error_{call_type}_{item_id}_{time.strftime('%Y%m%d%H%M%S')}.txt")
    try:
        with open(error_filename, "w", encoding="utf-8") as err_f:
            err_f.write(f"--- System Prompt ({call_type}) ---\n{system_prompt}\n\n")
            err_f.write(f"--- Raw LLM Output ---\n{raw_output}\n")
        logger.info(f"Logged LLM error details to {error_filename}")
    except Exception as e: logger.error(f"Could not write LLM error file {error_filename}: {e}")


async def process_historical_prediction_for_refinement(session, endpoint_url, historical_pred_obj):
    item_id = historical_pred_obj.get("id", "unknown_historical_id")
    system_prompt = format_system_prompt(SYSTEM_PROMPT_TEMPLATE, historical_pred_obj)
    user_prompt = "Please output the JSON in the specified format ONLY, nothing else." # User prompt can be simple
    
    raw_llm_output = await make_api_call_async(
        session, endpoint_url, MODEL_NAME, 
        system_prompt, user_prompt, 
        MAX_TOKENS, TEMPERATURE, 
        item_id, "historical_refinement"
    )

    output_item_base = {
        "original_prediction_id": item_id,
        "original_prediction_data_preview": {
            "prediction_date": historical_pred_obj.get("metadata", {}).get("prediction_date"),
            "actual_outcome": historical_pred_obj.get("metadata", {}).get("actual_outcome"),
            "key_events_count": len(historical_pred_obj.get("key_events", []))
        },
        "llm_model_used_for_refinement": MODEL_NAME,
        "refinement_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    if raw_llm_output:
        cleaned_llm_output = strip_think_tags(raw_llm_output)
        json_str = extract_json_from_text(cleaned_llm_output)
        if json_str:
            try:
                refined_data = json.loads(json_str)
                output_item_base["refined_executive_brief"] = refined_data
                logger.info(f"  Successfully refined historical prediction ID: {item_id} using {endpoint_url}")
            except json.JSONDecodeError:
                logger.error(f"  Error: Refinement LLM output for historical ID {item_id} (using {endpoint_url}) was not valid JSON.")
                log_llm_error(item_id, f"historical_refinement_{endpoint_url.split('/')[-1]}", system_prompt, raw_llm_output)
                output_item_base["refined_executive_brief"] = {"error": "LLM output was not valid JSON."}
        else:
            logger.error(f"  Error: No JSON found in refinement LLM output for historical ID {item_id} (using {endpoint_url}).")
            log_llm_error(item_id, f"historical_refinement_{endpoint_url.split('/')[-1]}", system_prompt, raw_llm_output)
            output_item_base["refined_executive_brief"] = {"error": "No JSON found in LLM output."}
    else: # API call failed after retries
        output_item_base["refined_executive_brief"] = {"error": f"Failed to get LLM response from {endpoint_url} after retries."}
        
    return item_id, output_item_base


async def main_async_pipeline():
    logger.info("--- Starting Historical Prediction Refinement Stage ---")

    historical_predictions_input = load_json_file(INPUT_JSON_FILE, [])
    if not historical_predictions_input:
        logger.error(f"Input file {INPUT_JSON_FILE} is empty or could not be loaded. Exiting.")
        # No dummy data creation here as per request for this specific input
        return

    if not isinstance(historical_predictions_input, list):
        logger.error(f"Input file {INPUT_JSON_FILE} must contain a list of historical prediction objects.")
        return

    processed_ids = set(load_json_file(PROGRESS_FILE, []))
    
    # This map will store original_prediction_id -> refined_brief_item
    final_results_map = {item['original_prediction_id']: item for item in load_json_file(OUTPUT_JSON_FILE, [])}
    
    items_to_refine = []
    for pred_item in historical_predictions_input:
        item_id = pred_item.get("id")
        if not item_id:
            # If historical items might not have IDs, generate one or log a more specific warning
            item_id = f"historical_no_id_{historical_predictions_input.index(pred_item)}"
            logger.warning(f"Historical item missing 'id', assigned temporary: {item_id}")
            pred_item["id"] = item_id # Add to object for consistency if needed by format_system_prompt

        if item_id not in processed_ids:
            items_to_refine.append(pred_item)
        elif item_id not in final_results_map: # If processed but not in current results map
             # Attempt to find it in the loaded all_refined_scenarios (now final_results_map)
             # This case is mostly for resuming after deleting output file but not progress file
             if item_id in final_results_map:
                 pass # Already loaded
             else: # Should not happen if progress file is accurate
                 logger.warning(f"Item {item_id} in progress file but not in loaded results. Will re-process if encountered.")
                 # To be safe, re-add to items_to_refine if it's missing from results map
                 # This could lead to re-processing if output file was deleted.
                 # A more robust way is to always trust progress file and only load items_to_refine
                 # that are NOT in processed_ids.
                 # The current logic for items_to_refine already handles this.
                 pass


    if not items_to_refine:
        logger.info("No new historical predictions to refine.")
        if not os.path.exists(PROGRESS_FILE) or os.path.getsize(PROGRESS_FILE) < 5:
            if final_results_map: save_json_file(list(final_results_map.values()), OUTPUT_JSON_FILE)
        return

    logger.info(f"Found {len(items_to_refine)} historical predictions for refinement.")

    tasks_iterated_count = 0
    successfully_refined_this_run = 0

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        
        tasks = []
        for i, pred_item_for_task in enumerate(items_to_refine):
            endpoint_url_for_task = ENDPOINTS[i % len(ENDPOINTS)]
            
            # Wrapper to correctly capture loop variables for async tasks
            async def sem_task_wrapper(item, url):
                async with semaphore:
                    return await process_historical_prediction_for_refinement(session, url, item)
            
            tasks.append(sem_task_wrapper(pred_item_for_task, endpoint_url_for_task))

        logger.info(f"Submitting {len(tasks)} tasks for historical prediction refinement across {len(ENDPOINTS)} endpoint(s)...")

        for future in asyncio.as_completed(tasks):
            item_id_processed, refined_data_item = await future
            tasks_iterated_count += 1

            if refined_data_item:
                final_results_map[item_id_processed] = refined_data_item
                # Check if refinement was successful for this item in this run
                if refined_data_item.get("refined_executive_brief") and \
                   not refined_data_item["refined_executive_brief"].get("error") and \
                   item_id_processed not in processed_ids: # Ensure it's a new success
                    processed_ids.add(item_id_processed)
                    successfully_refined_this_run += 1
            
            if successfully_refined_this_run > 0 and (tasks_iterated_count % SAVE_INTERVAL == 0):
                logger.info(f"\n--- Periodic Save (Historical Refinement): Iterated {tasks_iterated_count} tasks. Saving progress and results. ---")
                save_json_file(list(final_results_map.values()), OUTPUT_JSON_FILE)
                save_json_file(list(processed_ids), PROGRESS_FILE)
                logger.info(f"--- Periodic Save (Historical Refinement) complete ---\n")
    
    logger.info(f"Historical prediction refinement stage finished. {successfully_refined_this_run} items newly refined.")
    if successfully_refined_this_run > 0 or (not os.path.exists(OUTPUT_JSON_FILE) and final_results_map):
        logger.info(f"Performing final save for historical prediction refinement stage...")
        save_json_file(list(final_results_map.values()), OUTPUT_JSON_FILE)
        save_json_file(list(processed_ids), PROGRESS_FILE)
        logger.info(f"Final save for historical prediction refinement complete.")

def main():
    asyncio.run(main_async_pipeline())

if __name__ == "__main__":
    main()