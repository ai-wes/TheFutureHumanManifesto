# src/narrative/narrative_generator.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama # Using Ollama as specified in README
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# It's good practice to manage model names and configurations centrally
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemma:2b-instruct" # As per README example

# --- Setup Vector Store (Retriever) --- 
# In a real application, the vector store would be built from the Knowledge Graph
# or a curated set of documents and persisted. This is a simplified example.

def get_retriever(texts=None, embeddings_model_name=EMBEDDINGS_MODEL_NAME, k_results=2):
    """
    Creates and returns a FAISS vector store retriever.
    If texts are provided, it builds a new store. Otherwise, it expects a pre-built store.
    (For this example, we'll always build from a dummy set of texts if none provided).
    """
    if texts is None:
        texts = [
            "In 2023, AlphaDev, a reinforcement learning agent, discovered faster sorting algorithms, hinting at AI's growing capability in scientific discovery.",
            "Longevity escape velocity (LEV) is a hypothetical future state where life expectancy increases by more than one year for every calendar year that passes, a key goal for some transhumanists.",
            "Neuralink continues its work on brain-machine interfaces (BMIs), aiming to restore sensory and motor function and eventually expand human capabilities.",
            "Transhumanism as a philosophy explores the ethical use of advanced technologies to enhance human intellectual, physical, and psychological capacities.",
            "The development of Artificial General Intelligence (AGI) is a major point of discussion, with potential timelines varying widely among experts.",
            "CRISPR gene editing technology has opened new avenues for treating genetic diseases, but also raises ethical questions about human enhancement."
        ]
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": k_results})
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        print("Ensure embedding models are downloaded and FAISS is correctly installed.")
        # Fallback to a dummy retriever that returns no context
        class DummyRetriever:
            def invoke(self, query):
                return []
        return DummyRetriever()

# Initialize retriever globally or pass it around as needed.
# For this script, we'll initialize it when the module loads.
retriever = get_retriever()

# --- Initialize LLM --- 
# Ensure Ollama service is running and the specified model (e.g., gemma:2b-instruct) is pulled.
def get_llm(model_name=LLM_MODEL_NAME, temperature=0.7):
    """
    Initializes and returns the LLM.
    """
    try:
        llm = Ollama(model=model_name, temperature=temperature)
        # You might add a check here to see if the LLM is responsive, if possible.
        # For example, try a very simple prompt.
        llm.invoke("Respond with 'OK'.") # Simple check
        print(f"LLM ({model_name}) initialized and responsive.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM ({model_name}): {e}")
        print("Ensure Ollama is running and the model is available (e.g., 'ollama pull gemma:2b-instruct').")
        # Fallback to a dummy LLM that returns a placeholder message
        class DummyLLM:
            def invoke(self, prompt_val):
                return f"Error: LLM not available. Prompt was: {prompt_val[:100]}..."
        return DummyLLM()

llm = get_llm()

# --- Narrative Generation Logic --- 
NARRATIVE_TEMPLATE_STRING = """
As a futurist and speculative fiction writer, craft an engaging narrative vignette (around 200-300 words) based on the following predicted future scenario. Your story should be plausible, grounded in the scenario details, and creatively explore the human experience within this future. Weave in the provided contextual information naturally. If the scenario mentions specific years, try to incorporate them. Avoid simply listing facts; create a compelling story.

**Scenario Details:**
{scenario_details_str}

**Relevant Context from Knowledge Base:**
{context}

**Narrative Vignette:**
"""

prompt_template = ChatPromptTemplate.from_template(NARRATIVE_TEMPLATE_STRING)

def format_scenario_for_prompt(scenario: dict) -> str:
    """Converts scenario dictionary to a string for the prompt."""
    lines = [f"Scenario ID: {scenario.get('scenario_id', 'N/A')}"]
    # lines.append(f"Overall Probability Estimate: {scenario.get('probability_estimate', 'N/A'):.4f}") # May not be useful for narrative
    
    events = scenario.get("events")
    if isinstance(events, dict):
        for event_name, event_data in events.items():
            lines.append(f"\n  Milestone Focus: {event_name.replace('_', ' ')}")
            if isinstance(event_data, dict):
                for year_key, val in sorted(event_data.items()): # Sort by year for readability
                    lines.append(f"    - {year_key.replace('_', ' ')}: {val:.2f}")
            else: # If event_data is simpler (e.g., just a predicted year or single value)
                lines.append(f"    - Predicted outcome/value: {event_data}")
    else:
        lines.append("  No detailed event data provided in this scenario structure.")
    return "\n".join(lines)

# Define the RAG (Retrieval Augmented Generation) chain
def get_rag_chain(current_llm, current_retriever):
    """
    Constructs and returns the RAG chain.
    """
    def retrieve_context_for_scenario(scenario_details_str: str):
        """Dynamically retrieves context based on the scenario string."""
        retrieved_docs = current_retriever.invoke(scenario_details_str)
        return "\n".join([doc.page_content for doc in retrieved_docs])

    chain = (
        {
            "scenario_details_str": RunnablePassthrough(), # Passes the input (formatted scenario string)
            "context": RunnableLambda(retrieve_context_for_scenario) # Dynamically gets context based on scenario string
        }
        | prompt_template
        | current_llm
        | StrOutputParser()
    )
    return chain

# Initialize the chain globally or create it on demand
rag_chain = get_rag_chain(llm, retriever)

def generate_narrative_for_scenario(scenario_data: dict) -> str:
    """Generates a narrative for a given scenario dictionary."""
    if not isinstance(scenario_data, dict):
        return "Error: Input scenario_data must be a dictionary."
    
    formatted_scenario_str = format_scenario_for_prompt(scenario_data)
    
    # Invoke the RAG chain with the formatted scenario string
    try:
        narrative = rag_chain.invoke(formatted_scenario_str)
        return narrative
    except Exception as e:
        print(f"Error during narrative generation: {e}")
        return f"Error generating narrative for scenario {scenario_data.get('scenario_id', 'N/A')}. Details: {e}"

if __name__ == "__main__":
    # Example scenario (similar to the one from scenario_generator.py output)
    example_scenario_data = {
        'scenario_id': 'scenario_example_001',
        'events': {
            'AGI_Achievement_Year_Prediction': {
                'year_2030': 2032.50,
                'year_2031': 2031.00,
                'year_2032': 2030.50 # Example: AGI predicted to be achieved by mid-2030
            },
            'Longevity_Escape_Velocity_Year_Prediction': {
                'year_2035': 2038.00,
                'year_2036': 2037.50
            }
        },
        'probability_estimate': 0.001 # Included for completeness, though not used in this prompt
    }
    
    print(f"--- Generating narrative for Scenario: {example_scenario_data['scenario_id']} ---")
    print(f"Input Scenario Details:\n{format_scenario_for_prompt(example_scenario_data)}")
    
    narrative_text = generate_narrative_for_scenario(example_scenario_data)
    
    print("\n--- Generated Narrative ---")
    print(narrative_text)

    # Another example with simpler event structure
    example_scenario_simple = {
        'scenario_id': 'scenario_simple_002',
        'events': {
            'Universal_Basic_Income_Implemented_Year': 2035 
        }
    }
    print(f"\n--- Generating narrative for Scenario: {example_scenario_simple['scenario_id']} ---")
    print(f"Input Scenario Details:\n{format_scenario_for_prompt(example_scenario_simple)}")
    narrative_text_simple = generate_narrative_for_scenario(example_scenario_simple)
    print("\n--- Generated Narrative (Simple Scenario) ---")
    print(narrative_text_simple)
