import json
import os
import time
import requests # Kept for potential synchronous fallbacks if ever needed, but not used in async path
import re
import aiohttp
import asyncio
import logging

# Suppress excessive aiohttp connection logging if not needed for debugging
logging.getLogger("aiohttp.connector").setLevel(logging.WARNING) # Changed from CRITICAL to WARNING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- API Configuration ---
# These would ideally be managed more securely, e.g., via environment variables or a config file
# For simplicity in this example, they are here.
# Replace with your actual API details if needed.
REFINEMENT_API_URL = "https://server.build-a-bf.com/v1/chat/completions" # Your endpoint for refinement
PLAUSIBILITY_API_URL = "https://server.build-a-bf.com/v1/chat/completions" # Can be same or different for plausibility
# If using OpenAI, you'd use the OpenAI client and set API keys via environment variables.
# This script uses a generic requests-like structure for the provided API_URL.

REFINEMENT_MODEL_NAME = "deepseek-r1-0528-qwen3-8b" # Model for initial refinement
PLAUSIBILITY_MODEL_NAME = "deepseek-r1-0528-qwen3-8b" # Model for plausibility scoring (ideally a more powerful one like GPT-4)

MAX_TOKENS_REFINEMENT = -1  # Or a specific number like 2048
MAX_TOKENS_PLAUSIBILITY = 512 # Plausibility output is smaller
TEMPERATURE_REFINEMENT = 0.6
TEMPERATURE_PLAUSIBILITY = 0.2 # Lower temperature for more deterministic scoring

# --- File Paths ---
INPUT_JSON_FILE = r"F:\TheFutureHumanManifesto\data\synthetic_scenarios_generated.json"
# Intermediate file for refined briefs before plausibility scoring
REFINED_BRIEFS_INTERMEDIATE_FILE = r"F:\TheFutureHumanManifesto\data\refined_scenarios_briefs_intermediate.json"
# Final output file with plausibility scores
OUTPUT_JSON_FILE_WITH_PLAUSIBILITY = r"F:\TheFutureHumanManifesto\data\refined_briefs_with_plausibility.json"
PROGRESS_FILE_REFINEMENT = "refinement_processing_progress.json"
PROGRESS_FILE_PLAUSIBILITY = "plausibility_processing_progress.json"
ERROR_LOG_DIR = "llm_error_logs"
os.makedirs(ERROR_LOG_DIR, exist_ok=True)


# --- Script Settings ---
SAVE_INTERVAL = 2 # Save after every N successfully processed items
RETRY_DELAY = 60    # Seconds to wait before retrying API calls
MAX_API_RETRIES = 3 # Max retries for a single API call

# --- System Prompt for Scenario Refinement (Option 1 from previous discussion) ---
REFINEMENT_SYSTEM_PROMPT_TEMPLATE = """
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
  "strategic_coherence_overview": "Your strategic overview assessing coherence and plausibility.",
  "defining_strategic_challenge": "The single most defining strategic challenge, core tension, or central dilemma."
}}
"""

# --- System Prompt for Plausibility Assessment ---
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

Output ONLY a JSON object with the following keys:
{{
  "plausibility_score": <float_between_0.0_and_1.0>,
  "reasoning": "Your brief reasoning here."
}}
"""

def format_list_for_prompt(data_list, indent=False):
    if not data_list:
        return "N/A"
    prefix = "  - " if indent else "- "
    return "\n".join([f"{prefix}{item}" for item in data_list])

def format_dict_list_for_prompt(dict_list, key1_name, key2_name):
    if not dict_list:
        return "N/A"
    formatted_items = []
    for item in dict_list:
        formatted_items.append(f"  - {key1_name.capitalize()}: {item.get(key1_name, 'N/A')}\n    Implication: {item.get(key2_name, 'N/A')}")
    return "\n".join(formatted_items)

def format_refinement_system_prompt(template, scenario_obj):
    start_year, end_year = scenario_obj.get("timeline", "Unknown-Unknown").split('-')
    return template.format(
        start_year=start_year,
        end_year=end_year,
        key_events_list_as_string=format_list_for_prompt(scenario_obj.get("key_events", [])),
        original_tech_factors_list_as_string=format_list_for_prompt(scenario_obj.get("technological_factors", [])), # Corrected key name
        original_social_factors_list_as_string=format_list_for_prompt(scenario_obj.get("social_factors", [])), # Corrected key name
        original_economic_factors_list_as_string=format_list_for_prompt(scenario_obj.get("economic_factors", [])) # Corrected key name
    )

def format_plausibility_system_prompt(template, refined_brief_item):
    brief = refined_brief_item.get("refined_executive_brief", {})
    return template.format(
        timeline_str=refined_brief_item.get("original_scenario_data_preview", {}).get("timeline", "N/A"),
        core_narrative_turning_points_str=format_list_for_prompt(brief.get("core_narrative_turning_points", []), indent=True),
        core_technological_drivers_str=format_dict_list_for_prompt(brief.get("core_technological_drivers", []), 'driver', 'implication'),
        defining_social_dynamics_str=format_dict_list_for_prompt(brief.get("defining_social_dynamics", []), 'dynamic', 'implication'),
        key_economic_transformations_str=format_dict_list_for_prompt(brief.get("key_economic_transformations", []), 'transformation', 'implication'),
        strategic_coherence_overview_text=brief.get("strategic_coherence_overview", "N/A"),
        defining_strategic_challenge_text=brief.get("defining_strategic_challenge", "N/A")
    )


def strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def extract_json_from_text(text):
    # Try to find JSON object specifically, robust to leading/trailing text
    match = re.search(r'^\s*(\{[\s\S]*\})\s*$', text)
    if match:
        return match.group(1)
    # Fallback for cases where JSON might be embedded
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        logger.warning("Extracted JSON was embedded in other text. Attempting to parse.")
        return match.group(0)
    return None

async def make_api_call_async(session, api_url, model_name, system_prompt_content, user_prompt_content, max_tokens, temperature, scenario_id, call_type="refinement"):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens != -1 else None, # API might not like -1
        "stream": False
    }
    # Remove max_tokens if None, as some APIs might error
    if payload["max_tokens"] is None:
        del payload["max_tokens"]

    headers = {"Content-Type": "application/json"}
    # Add Authorization header if your API needs it, e.g.
    # headers["Authorization"] = f"Bearer {YOUR_API_KEY_VAR}"

    retries = 0
    while retries < MAX_API_RETRIES:
        try:
            logger.debug(f"  Attempting API call ({call_type}) for scenario ID: {scenario_id} to {api_url} (Attempt {retries + 1})")
            async with session.post(api_url, headers=headers, json=payload, timeout=300) as response:
                response_text = await response.text()
                if response.status != 200:
                    logger.error(f"    API ({call_type}) returned status code {response.status} for {scenario_id}: {response_text[:500]}")
                    retries += 1
                    if response.status == 429: # Rate limit
                        logger.info(f"    Rate limit hit. Waiting {RETRY_DELAY * (retries +1)}s before retry...")
                        await asyncio.sleep(RETRY_DELAY * (retries+1)) # Exponential backoff might be better
                    else:
                        await asyncio.sleep(RETRY_DELAY)
                    continue
                
                response_json = json.loads(response_text) # Parse after checking status
                content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                if not content:
                    logger.warning(f"    API ({call_type}) response for {scenario_id} had no content in choices[0].message.content. Full response: {response_json}")
                return content
        except aiohttp.ClientConnectorError as e:
            logger.error(f"    Connection error for API call ({call_type}) for {scenario_id} to {api_url}: {e}. Retrying in {RETRY_DELAY}s...")
        except asyncio.TimeoutError:
            logger.error(f"    Timeout error for API call ({call_type}) for {scenario_id} to {api_url}. Retrying in {RETRY_DELAY}s...")
        except json.JSONDecodeError as e:
            logger.error(f"    JSONDecodeError for API call ({call_type}) for {scenario_id} from {api_url}: {e}. Response text: {response_text[:500]}")
            # Don't retry JSON decode errors usually, it means malformed response from server
            return None # Or handle as a permanent failure for this item
        except Exception as e:
            logger.error(f"    Unexpected error during API call ({call_type}) for {scenario_id} to {api_url}: {e}. Retrying in {RETRY_DELAY}s...")
        
        retries += 1
        await asyncio.sleep(RETRY_DELAY)

    logger.error(f"  Failed API call ({call_type}) for scenario ID {scenario_id} to {api_url} after {MAX_API_RETRIES} retries.")
    return None


def load_json_file(filepath, default_value):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"File {filepath} is corrupted or not valid JSON. Using default.")
        except Exception as e:
            logger.error(f"Could not load file {filepath}: {e}. Using default.")
    return default_value

def save_json_file(data, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(data)} items to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")

def log_llm_error(scenario_id, call_type, system_prompt, raw_output):
    error_filename = os.path.join(ERROR_LOG_DIR, f"error_{call_type}_{scenario_id}_{time.strftime('%Y%m%d%H%M%S')}.txt")
    try:
        with open(error_filename, "w", encoding="utf-8") as err_f:
            err_f.write(f"--- System Prompt ({call_type}) ---\n{system_prompt}\n\n")
            err_f.write(f"--- Raw LLM Output ---\n{raw_output}\n")
        logger.info(f"Logged LLM error details to {error_filename}")
    except Exception as e:
        logger.error(f"Could not write LLM error file {error_filename}: {e}")


async def process_scenario_for_refinement(session, endpoint, scenario_obj, scenario_id):
    system_prompt = format_refinement_system_prompt(REFINEMENT_SYSTEM_PROMPT_TEMPLATE, scenario_obj)
    user_prompt = "Please output the JSON in the specified format ONLY, nothing else."
    raw_llm_output = await make_api_call_async(session, endpoint, REFINEMENT_MODEL_NAME, system_prompt, user_prompt, MAX_TOKENS_REFINEMENT, TEMPERATURE_REFINEMENT, scenario_id, "refinement")

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
                    "refined_executive_brief": refined_data, # This is the LLM's refined output
                    "llm_model_used_for_refinement": REFINEMENT_MODEL_NAME,
                    "refinement_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                }
                logger.info(f"  Successfully refined scenario ID: {scenario_id}")
                return scenario_id, output_item
            except json.JSONDecodeError:
                logger.error(f"  Error: Refinement LLM output for scenario ID {scenario_id} was not valid JSON.")
                log_llm_error(scenario_id, "refinement", system_prompt, raw_llm_output)
                return scenario_id, None
        else:
            logger.error(f"  Error: No JSON found in refinement LLM output for scenario ID {scenario_id}.")
            log_llm_error(scenario_id, "refinement", system_prompt, raw_llm_output)
            return scenario_id, None
    return scenario_id, None


async def process_brief_for_plausibility(session, endpoint, refined_brief_item, scenario_id):
    system_prompt = format_plausibility_system_prompt(PLAUSIBILITY_SYSTEM_PROMPT_TEMPLATE, refined_brief_item)
    user_prompt = "Please output the JSON in the specified format ONLY, nothing else, containing only 'plausibility_score' and 'reasoning'."
    raw_llm_output = await make_api_call_async(session, endpoint, PLAUSIBILITY_MODEL_NAME, system_prompt, user_prompt, MAX_TOKENS_PLAUSIBILITY, TEMPERATURE_PLAUSIBILITY, scenario_id, "plausibility")

    if raw_llm_output:
        cleaned_llm_output = strip_think_tags(raw_llm_output)
        json_str = extract_json_from_text(cleaned_llm_output)
        if json_str:
            try:
                plausibility_data = json.loads(json_str)
                # Update the refined_brief_item directly
                refined_brief_item['llm_assigned_plausibility'] = plausibility_data.get('plausibility_score')
                refined_brief_item['llm_plausibility_reasoning'] = plausibility_data.get('reasoning')
                refined_brief_item['llm_model_used_for_plausibility'] = PLAUSIBILITY_MODEL_NAME
                refined_brief_item['plausibility_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                logger.info(f"  Successfully scored plausibility for scenario ID: {scenario_id}")
                return scenario_id, refined_brief_item # Return the updated item
            except json.JSONDecodeError:
                logger.error(f"  Error: Plausibility LLM output for scenario ID {scenario_id} was not valid JSON.")
                log_llm_error(scenario_id, "plausibility", system_prompt, raw_llm_output)
                return scenario_id, refined_brief_item # Return original item, maybe with error flags
        else:
            logger.error(f"  Error: No JSON found in plausibility LLM output for scenario ID {scenario_id}.")
            log_llm_error(scenario_id, "plausibility", system_prompt, raw_llm_output)
            return scenario_id, refined_brief_item
    return scenario_id, refined_brief_item # Return original item if API call failed


async def run_processing_stage(
    stage_name: str,
    input_list: list,
    progress_file: str,
    output_file: str, # File to save results of this stage
    processing_function: callable, # e.g., process_scenario_for_refinement
    endpoints: list,
    concurrency: int
):
    logger.info(f"--- Starting {stage_name} Stage ---")
    processed_ids_stage = load_json_file(progress_file, []) # Load IDs processed in this stage
    
    # Load existing results for this stage to append/update
    stage_results_map = {item['original_scenario_id']: item for item in load_json_file(output_file, [])}
    
    # Determine items that need processing for *this stage*
    items_to_process_stage = []
    for item_data in input_list: # item_data is either original scenario or refined_brief_item
        item_id = item_data.get("original_scenario_id") if stage_name == "Plausibility Scoring" else item_data.get("id")
        if not item_id: # Generate temp ID if original scenario has no ID yet
            item_id = f"temp_id_{input_list.index(item_data)}"
            if 'id' not in item_data and stage_name != "Plausibility Scoring": # Add to original if it's the first stage
                 item_data['id'] = item_id

        if item_id not in processed_ids_stage:
            items_to_process_stage.append(item_data)
        # If already processed, ensure it's in stage_results_map (e.g. from previous partial run)
        elif item_id not in stage_results_map:
             stage_results_map[item_id] = item_data # Add it if it was processed but not in current results list

    if not items_to_process_stage:
        logger.info(f"No new items to process for {stage_name} stage.")
        return list(stage_results_map.values()) # Return all items (old + potentially new if logic changes)

    logger.info(f"Found {len(items_to_process_stage)} items for {stage_name} stage.")

    tasks_submitted_count = 0
    successfully_processed_this_run = 0

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def sem_task_wrapper(item_data_for_task):
            item_id_for_task = item_data_for_task.get("original_scenario_id") if stage_name == "Plausibility Scoring" else item_data_for_task.get("id")
            endpoint = endpoints[tasks.index(asyncio.current_task()) % len(endpoints)] # Simple round robin
            async with semaphore:
                return await processing_function(session, endpoint, item_data_for_task, item_id_for_task)

        tasks = [sem_task_wrapper(item) for item in items_to_process_stage]
        logger.info(f"Submitting {len(tasks)} tasks for {stage_name}...")

        for future in asyncio.as_completed(tasks):
            item_id_processed, processed_data = await future # processed_data is the output_item or updated_brief_item

            if processed_data: # If processing_function returned something (even if it's the original item with error flags)
                stage_results_map[item_id_processed] = processed_data # Update or add to our map of results
                
                # Check if it was a genuinely successful processing for *this stage*
                # For refinement, success means refined_executive_brief is present
                # For plausibility, success means llm_assigned_plausibility is present
                is_successful_stage_processing = False
                if stage_name == "Refinement" and processed_data.get("refined_executive_brief"):
                    is_successful_stage_processing = True
                elif stage_name == "Plausibility Scoring" and processed_data.get("llm_assigned_plausibility") is not None:
                    is_successful_stage_processing = True
                
                if is_successful_stage_processing and item_id_processed not in processed_ids_stage:
                    processed_ids_stage.append(item_id_processed) # Add to this stage's processed list
                    successfully_processed_this_run += 1
            
            tasks_submitted_count +=1 # Counts completed tasks from the submitted batch

            if successfully_processed_this_run > 0 and (tasks_submitted_count % SAVE_INTERVAL == 0):
                logger.info(f"\n--- Periodic Save ({stage_name}): Iterated {tasks_submitted_count} tasks. Saving progress and results. ---")
                save_json_file(list(stage_results_map.values()), output_file)
                save_json_file(processed_ids_stage, progress_file) # Save list of IDs for this stage
                logger.info(f"--- Periodic Save ({stage_name}) complete ---\n")
    
    logger.info(f"{stage_name} stage finished. {successfully_processed_this_run} items newly processed.")
    if successfully_processed_this_run > 0:
        logger.info(f"Performing final save for {stage_name} stage...")
        save_json_file(list(stage_results_map.values()), output_file)
        save_json_file(processed_ids_stage, progress_file)
        logger.info(f"Final save for {stage_name} complete.")
    
    return list(stage_results_map.values())


async def main_async_pipeline():
    # --- Load initial input data ---
    input_scenarios = load_json_file(INPUT_JSON_FILE, [])
    if not input_scenarios:
        logger.error(f"Input file {INPUT_JSON_FILE} is empty or could not be loaded. Exiting.")
        # Create dummy data if file is missing, for testing
        if not os.path.exists(INPUT_JSON_FILE):
            logger.info(f"Input file {INPUT_JSON_FILE} not found. Creating a dummy file for testing.")
            dummy_data = [
                {
                    "id": "dummy_1", "timeline": "2030-2040", "key_events": ["AI sentient"], 
                    "technological_factors": [], "social_factors": [], "economic_factors": [], 
                    "target_probability_synthetic": 0.5
                },
                {
                    "id": "dummy_2", "timeline": "2040-2050", "key_events": ["Mars colony"],
                    "technological_factors": [], "social_factors": [], "economic_factors": [],
                    "target_probability_synthetic": 0.8
                }
            ]
            save_json_file(dummy_data, INPUT_JSON_FILE)
            input_scenarios = dummy_data
        else:
            return

    if not isinstance(input_scenarios, list):
        logger.error(f"Input file {INPUT_JSON_FILE} must contain a list of scenarios.")
        return

    # --- Stage 1: Scenario Refinement ---
    refined_briefs = await run_processing_stage(
        stage_name="Refinement",
        input_list=input_scenarios, # Original scenarios
        progress_file=PROGRESS_FILE_REFINEMENT,
        output_file=REFINED_BRIEFS_INTERMEDIATE_FILE,
        processing_function=process_scenario_for_refinement,
        endpoints=ENDPOINTS,
        concurrency=CONCURRENCY
    )

    if not refined_briefs:
        logger.error("Refinement stage yielded no results. Exiting.")
        return

    # Filter out items where refinement might have failed (output_item was None)
    successfully_refined_briefs = [item for item in refined_briefs if item.get("refined_executive_brief")]
    if not successfully_refined_briefs:
        logger.error("No scenarios were successfully refined. Exiting plausibility stage.")
        return
    
    logger.info(f"{len(successfully_refined_briefs)} briefs successfully refined, proceeding to plausibility scoring.")

    # --- Stage 2: Plausibility Scoring ---
    final_results_with_plausibility = await run_processing_stage(
        stage_name="Plausibility Scoring",
        input_list=successfully_refined_briefs, # Feed refined briefs to this stage
        progress_file=PROGRESS_FILE_PLAUSIBILITY,
        output_file=OUTPUT_JSON_FILE_WITH_PLAUSIBILITY,
        processing_function=process_brief_for_plausibility,
        endpoints=ENDPOINTS, # Can use same or different endpoints/models
        concurrency=CONCURRENCY
    )
    
    logger.info("All processing stages complete.")
    if final_results_with_plausibility:
        logger.info(f"Final output with plausibility scores saved to {OUTPUT_JSON_FILE_WITH_PLAUSIBILITY}")


def main():
    asyncio.run(main_async_pipeline())

if __name__ == "__main__":
    main()