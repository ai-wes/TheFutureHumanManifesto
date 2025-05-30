# src/dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np # For dummy data generation

# --- Placeholder Data Loading Functions ---
# In a real application, these would interact with your backend services,
# databases, or file storage where scenarios, narratives, and forecasts are stored.

# Placeholder: Load forecast data (e.g., from STSModel outputs)
@st.cache_data # Cache data to avoid reloading on every interaction
def load_forecast_data(milestone_name: str):
    """Simulates loading forecast data for a given milestone."""
    st.info(f"Simulating data load for milestone: {milestone_name}")
    # Dummy data mimicking STSModel output structure
    if milestone_name == "AGI_Achievement_Year_Prediction":
        years = np.arange(2025, 2035)
        mean_forecast = np.linspace(2035, 2028, len(years)) - np.random.rand(len(years)) * 2
        stddev_forecast = np.linspace(1, 3, len(years)) + np.random.rand(len(years)) * 0.5
        return {
            "forecast_years": years,
            "mean": mean_forecast,
            "stddev": stddev_forecast,
            "samples": np.random.normal(loc=mean_forecast, scale=stddev_forecast, size=(100, len(years)))
        }
    elif milestone_name == "Longevity_Escape_Velocity_Year_Prediction":
        years = np.arange(2025, 2040)
        mean_forecast = np.linspace(2040, 2035, len(years)) - np.random.rand(len(years)) * 3
        stddev_forecast = np.linspace(2, 4, len(years)) + np.random.rand(len(years)) * 1
        return {
            "forecast_years": years,
            "mean": mean_forecast,
            "stddev": stddev_forecast,
            "samples": np.random.normal(loc=mean_forecast, scale=stddev_forecast, size=(100, len(years)))
        }
    return None

# Placeholder: Load scenarios (e.g., from ScenarioGenerator outputs)
@st.cache_data
def load_scenarios_from_db(limit=10):
    """Simulates loading a list of scenarios."""
    st.info(f"Simulating scenario load (limit: {limit})")
    scenarios = []
    for i in range(limit):
        base_agi_year = 2030 + np.random.randint(-3, 3)
        base_lev_year = 2038 + np.random.randint(-5, 5)
        scenarios.append({
            "scenario_id": f"scenario_dashboard_{i+1}",
            "events": {
                "AGI_Achievement_Year_Prediction": {
                    f"year_{base_agi_year-1}": base_agi_year + 0.5 + np.random.randn(),
                    f"year_{base_agi_year}": base_agi_year + np.random.randn(),
                    f"year_{base_agi_year+1}": base_agi_year - 0.5 + np.random.randn(),
                },
                "Longevity_Escape_Velocity_Year_Prediction": {
                    f"year_{base_lev_year-2}": base_lev_year + 1 + np.random.randn(),
                    f"year_{base_lev_year}": base_lev_year + np.random.randn(),
                    f"year_{base_lev_year+2}": base_lev_year - 1 + np.random.randn(),
                }
            },
            "probability_estimate": np.random.rand() * 0.01 # low probability for specific complex scenarios
        })
    return scenarios

# Placeholder: Load narratives for a scenario (e.g., from NarrativeGenerator outputs)
@st.cache_data
def load_narratives_for_scenario(scenario_id: str):
    """Simulates loading narratives for a given scenario ID."""
    st.info(f"Simulating narrative load for scenario: {scenario_id}")
    # Dummy narratives
    return [
        f"Narrative Alpha for {scenario_id}: The world changed rapidly when AGI emerged around the predicted timeframe. Innovations in longevity soon followed, challenging societal norms...",
        f"Narrative Beta for {scenario_id}: A more cautious tale unfolds. While progress towards AGI was evident, unforeseen hurdles delayed its full impact. Meanwhile, longevity research offered incremental but meaningful gains..."
    ]

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="GAPS 2.0 Dashboard")
st.title("🔮 GAPS 2.0 - Transhumanist Futures Dashboard")
st.caption("Exploring Probabilistic Forecasts, Scenarios, and Narratives")

# --- Sidebar for Navigation/Filters ---
st.sidebar.header("Dashboard Controls")

milestone_options = [
    "AGI_Achievement_Year_Prediction", 
    "Longevity_Escape_Velocity_Year_Prediction", 
    # Add more milestone names as defined in your system
]
selected_milestone = st.sidebar.selectbox(
    "Select Milestone for Detailed Forecast", 
    milestone_options,
    index=0
)

num_scenarios_to_display = st.sidebar.slider("Number of Scenarios to Display", 5, 50, 10)

# --- Main Content Area ---

# Tab 1: Forecasts
tab1, tab2, tab3 = st.tabs(["📈 Forecasts", "🎭 Scenarios & Narratives", "ℹ️ System Overview"])

with tab1:
    st.header(f"Detailed Forecast: {selected_milestone}")
    forecast_data = load_forecast_data(selected_milestone)
    
    if forecast_data and forecast_data["forecast_years"] is not None and len(forecast_data["forecast_years"]) > 0:
        df_forecast = pd.DataFrame({
            'Year': forecast_data['forecast_years'],
            'Mean Forecast': forecast_data['mean'],
            'StdDev_Upper': forecast_data['mean'] + forecast_data['stddev'],
            'StdDev_Lower': forecast_data['mean'] - forecast_data['stddev'],
        })
        
        fig = px.line(df_forecast, x='Year', y='Mean Forecast', title=f"Probabilistic Forecast for {selected_milestone}")
        fig.add_scatter(x=df_forecast['Year'], y=df_forecast['StdDev_Upper'], fill=None, mode='lines', line_color='lightgrey', name='Mean +/- StdDev')
        fig.add_scatter(x=df_forecast['Year'], y=df_forecast['StdDev_Lower'], fill='tonexty', mode='lines', line_color='lightgrey', name='StdDev Range')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Forecast Distribution (Sample Paths)")
        if "samples" in forecast_data and forecast_data["samples"].shape[0] > 0:
            sample_df_list = []
            for i in range(min(20, forecast_data["samples"].shape[0])): # Plot up to 20 sample paths
                temp_df = pd.DataFrame({
                    'Year': forecast_data['forecast_years'],
                    'SampleValue': forecast_data["samples"][i, :],
                    'Sample': f'Sample {i+1}'
                })
                sample_df_list.append(temp_df)
            
            if sample_df_list:
                full_sample_df = pd.concat(sample_df_list)
                fig_samples = px.line(full_sample_df, x='Year', y='SampleValue', color='Sample', 
                                      title="Forecast Sample Paths", labels={'SampleValue': 'Forecasted Value'})
                fig_samples.update_layout(showlegend=False) # Hide legend if too many samples
                st.plotly_chart(fig_samples, use_container_width=True)
            else:
                st.write("No sample paths to display.")
        else:
            st.write("No forecast samples available for this milestone.")

        with st.expander("Raw Forecast Data"):
            st.json({
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in forecast_data.items()
            })
    else:
        st.warning(f"No forecast data available or data is empty for {selected_milestone}.")

# Tab 2: Scenarios and Narratives
with tab2:
    st.header("Generated Scenarios & Associated Narratives")
    scenarios = load_scenarios_from_db(limit=num_scenarios_to_display)
    
    if scenarios:
        for i, scenario in enumerate(scenarios):
            st.subheader(f"{i+1}. Scenario ID: {scenario['scenario_id']}")
            col1, col2 = st.columns([0.4, 0.6]) # Ratio for columns
            with col1:
                st.metric(label="Estimated Probability", value=f"{scenario.get('probability_estimate',0):.5f}")
                with st.expander("Scenario Event Details (JSON)", expanded=False):
                    st.json(scenario.get('events', {}))
            
            with col2:
                st.markdown("**Generated Narratives:**")
                narratives = load_narratives_for_scenario(scenario['scenario_id'])
                if narratives:
                    for j, narrative_text in enumerate(narratives):
                        st.markdown(f"**Variant {j+1}:**")
                        st.markdown(f"> {narrative_text}")
                        if st.button("Mark for Editorial Review", key=f"review_{scenario['scenario_id']}_{j}", help="Flag this narrative for deeper review."):
                            st.toast(f"Narrative '{scenario['scenario_id']} - Variant {j+1}' flagged for review!", icon="🚩")
                        st.markdown("---")
                else:
                    st.write("No narratives generated yet for this scenario or an error occurred.")
            st.divider()
    else:
        st.info("No scenarios loaded. Adjust controls or check data sources.")

# Tab 3: System Overview
with tab3:
    st.header("System Overview & Methodology")
    st.markdown("""
    The Generative Assistive Prediction System (GAPS 2.0) integrates several components to forecast 
    and narrate potential transhumanist futures:

    1.  **Data Ingestion**: Real-time collection of news, research, patents, etc., via Kafka.
    2.  **Knowledge Graph**: Neo4j stores structured information and relationships.
    3.  **Probabilistic Forecasting**: TensorFlow Probability (STS) models predict milestones.
    4.  **Scenario Generation**: Monte Carlo methods create plausible future timelines from forecasts.
    5.  **Narrative Synthesis**: Large Language Models (via LangChain) transform scenarios into readable stories, 
        augmented by knowledge graph context (RAG).
    6.  **Dashboard**: This Streamlit application provides an interface to explore the system's outputs.

    **Note**: Data displayed in this dashboard is currently simulated for demonstration purposes.
    """)
    st.subheader("Data Flow (Conceptual)")
    st.graphviz_chart("""
        digraph {
            rankdir="LR";
            node [shape=box, style=rounded];
            DataSources [label="News, Papers, Patents, etc."];
            Kafka [label="Kafka Bus"];
            StreamProc [label="Stream Processing (Quix)"];
            Neo4j [label="Neo4j Knowledge Graph"];
            STSModel [label="STS Forecasting (TFP)"];
            ScenarioGen [label="Scenario Generator"];
            NarrativeGen [label="Narrative Generator (LLM)"];
            Dashboard [label="Streamlit Dashboard"];

            DataSources -> Kafka;
            Kafka -> StreamProc;
            StreamProc -> Kafka [label="processed data"];
            Kafka -> Neo4j [label="to KG"];
            Neo4j -> STSModel [label="features"];
            STSModel -> ScenarioGen [label="forecasts"];
            ScenarioGen -> NarrativeGen [label="scenarios"];
            NarrativeGen -> Dashboard [label="narratives"];
            ScenarioGen -> Dashboard [label="scenarios"];
            STSModel -> Dashboard [label="forecasts"];
        }
    """)

# To run this app: streamlit run src/dashboard/app.py
if __name__ == "__main__":
    # This block is mainly for direct execution if needed, though Streamlit handles it.
    # Can add any specific setup for direct run if necessary.
    st.sidebar.info("Dashboard loaded. Select controls to explore.")
    print("Streamlit dashboard app.py is ready. Run with 'streamlit run src/dashboard/app.py'")
