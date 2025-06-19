import re

def plot_key_event_timeline_distribution(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return

    event_years = []
    for item in data_list:
        for event_str in item.get('key_events', []):
            # Simple regex to find 4-digit years (e.g., "Year 2035:", "(2040)")
            found_years = re.findall(r'\b(20[2-9]\d)\b', event_str) 
            if found_years:
                event_years.append(int(found_years[0])) # Take the first year found

    if not event_years:
        print("No event years found to plot.")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(event_years, kde=False, bins=range(min(event_years), max(event_years) + 2, 1), discrete=True) # Bins per year
    plt.title('Distribution of Key Event Years in Synthetic Scenarios')
    plt.xlabel('Year')
    plt.ylabel('Number of Key Events')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# You would call this function after your train_model function completes,
# passing lists of recorded train and validation losses.
def plot_loss_curves(train_losses: List[float], val_losses: Optional[List[float]] = None, num_epochs: int = 0):
    epochs = range(1, len(train_losses) + 1)
    if num_epochs > 0 and len(epochs) != num_epochs and len(train_losses) < num_epochs:
         print(f"Warning: Plotting {len(train_losses)} epochs, but {num_epochs} were expected (possibly due to early stopping).")


    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    if val_losses and len(val_losses) == len(train_losses):
        plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    elif val_losses:
        print(f"Warning: Mismatch in length of train_losses ({len(train_losses)}) and val_losses ({len(val_losses)}). Not plotting validation loss.")

    plt.title('Training and Validation Loss (NLL)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# --- Inside your train_model function in train_probabilistic_nn.py ---
# def train_model(...):
#     ...
#     all_train_losses = []
#     all_val_losses = []
#     ...
#     for epoch in range(num_epochs):
#         ...
#         avg_train_loss = train_loss_accum / len(train_loader)
#         all_train_losses.append(avg_train_loss)
#         ...
#         if val_loader and len(val_loader) > 0:
#             ...
#             current_val_loss = val_loss_accum / len(val_loader)
#             all_val_losses.append(current_val_loss)
#         ...
#     logger.info("Training finished.")
#     return all_train_losses, all_val_losses # Return the losses

# --- In the __main__ block of train_probabilistic_nn.py, after calling train_model ---
# if len(train_loader) > 0:
#     train_losses_history, val_losses_history = train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device, 
#                                                            patience=EARLY_STOPPING_PATIENCE, best_model_path=BEST_MODEL_SAVE_PATH)
#     plot_loss_curves(train_losses_history, val_losses_history, num_epochs=NUM_EPOCHS)
#     ...


# Example usage:
# plot_key_event_timeline_distribution("data/synthetic_scenarios_generated.json")



# Assume 'predictions' is a list of dicts from forecaster.predict_scenario_probability()
# and 'test_genomes' is the corresponding list of ScenarioGenome objects
def plot_probability_vs_uncertainty(predictions: List[Dict[str, Any]], test_genomes: List[ScenarioGenome]):
    if not predictions:
        print("No predictions to plot.")
        return

    probs = [p.get('probability', 0.5) for p in predictions]
    std_devs = [p.get('std_dev', 0.1) for p in predictions]
    # Optionally, get another metric like number of key events for point size/color
    num_key_events = [len(g.key_events or []) for g in test_genomes]

    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(probs, std_devs, c=num_key_events, cmap='viridis', alpha=0.7, s=[(n+1)*20 for n in num_key_events])
    
    plt.title('Predicted Scenario Probability vs. Uncertainty (Std Dev)')
    plt.xlabel('Predicted Probability of Scenario')
    plt.ylabel('Predicted Standard Deviation (Uncertainty)')
    cbar = plt.colorbar(scatter, label='Number of Key Events')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate some points (e.g., top 3 most probable)
    # sorted_preds = sorted(zip(probs, std_devs, [g.id for g in test_genomes]), key=lambda x: x[0], reverse=True)
    # for i, (p, s, gid) in enumerate(sorted_preds[:3]):
    #     plt.annotate(f"ID: {gid[:8]}", (p, s), textcoords="offset points", xytext=(0,10), ha='center')

    plt.show()

# Example usage (you'd call this after getting predictions):
# forecaster = HybridProbabilisticForecaster(config_loader_instance=config)
# test_scenarios_genomes = [...] # Your list of ScenarioGenomes to test
# forecast_results = [forecaster.predict_scenario_probability(scn) for scn in test_scenarios_genomes]
# plot_probability_vs_uncertainty(forecast_results, test_scenarios_genomes)


import plotly.express as px
import pandas as pd

def plot_interactive_scenario_space(predictions: List[Dict[str, Any]], test_genomes: List[ScenarioGenome]):
    if not predictions or len(predictions) != len(test_genomes):
        print("Prediction and genome lists are empty or mismatched.")
        return

    data_for_df = []
    for genome, pred in zip(test_genomes, predictions):
        data_for_df.append({
            'id': genome.id,
            'probability': pred.get('probability'),
            'std_dev': pred.get('std_dev'),
            # Assume consistency_score is added to prediction dict or fetched separately
            'consistency': pred.get('consistency_score', np.random.rand()), # Dummy consistency
            'num_tech_factors': len(genome.technological_factors or []),
            'num_key_events': len(genome.key_events or []),
            'primary_domain': (genome.domains_focused[0] if genome.domains_focused else "N/A"),
            'timeline_duration': int(genome.timeline.split('-')[-1]) - int(genome.timeline.split('-')[0]) if '-' in genome.timeline and genome.timeline.split('-')[0].isdigit() and genome.timeline.split('-')[-1].isdigit() else 20
        })
    
    df = pd.DataFrame(data_for_df)
    df.dropna(subset=['probability', 'std_dev', 'consistency'], inplace=True) # Drop rows if key metrics are missing

    if df.empty:
        print("DataFrame is empty after processing, cannot create plot.")
        return

    fig = px.scatter_3d(df, 
                        x='probability', 
                        y='std_dev', 
                        z='consistency',
                        color='primary_domain', 
                        size='num_key_events',
                        hover_data=['id', 'timeline_duration', 'num_tech_factors'],
                        title="Interactive Scenario Space Explorer")
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    fig.show()

# Example usage (after getting predictions and consistency scores):
# plot_interactive_scenario_space(detailed_forecast_results, test_scenarios_genomes)



def plot_simulation_run_variables(history: List[Dict[str, Any]], variables_to_plot: List[str]):
    if not history:
        print("No simulation history to plot.")
        return
        
    df_data = {}
    df_data['year'] = [h['year'] for h in history]
    for var_name in variables_to_plot:
        if var_name in history[0]['variables']: # Check if var exists
             df_data[var_name] = [h['variables'].get(var_name) for h in history]
        else:
            print(f"Warning: Variable '{var_name}' not found in simulation history.")


    df = pd.DataFrame(df_data)
    if df.empty or len(df.columns) <=1 : # Need year + at least one var
        print("Not enough data to plot simulation variables.")
        return

    plt.figure(figsize=(14, 7))
    for var_name in variables_to_plot:
        if var_name in df.columns:
            plt.plot(df['year'], df[var_name], label=var_name, marker='o', linestyle='-', markersize=4)
    
    plt.title('Evolution of Key Simulation Variables Over Time (Single Run)')
    plt.xlabel('Year')
    plt.ylabel('Variable Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage (after a simulation run):
# simulator = SyntheticScenarioSimulator(config_loader_instance=config)
# final_state, sim_history = simulator.run_single_simulation(2025, 2050)
# vars_to_plot = ["AGI_Capability", "Public_Acceptance_RadicalTech", "Environmental_Stability_Index", "Funding_FutureTech_Level"]
# plot_simulation_run_variables(sim_history, vars_to_plot)