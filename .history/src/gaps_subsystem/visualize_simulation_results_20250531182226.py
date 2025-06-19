import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Optional

def plot_key_event_timeline_distribution(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return None

    event_years = []
    for item in data_list:
        for event_str in item.get('key_events', []):
            found_years = re.findall(r'\b(20[2-9]\d)\b', event_str)
            if found_years:
                event_years.append(int(found_years[0]))
    if not event_years:
        print("No event years found to plot.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(event_years, kde=False, bins=range(min(event_years), max(event_years) + 2, 1), discrete=True, ax=ax)
    ax.set_title('Distribution of Key Event Years in Synthetic Scenarios')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Key Events')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    return fig

def plot_loss_curves(train_losses: List[float], val_losses: Optional[List[float]] = None, num_epochs: int = 0):
    epochs = range(1, len(train_losses) + 1)
    if num_epochs > 0 and len(epochs) != num_epochs and len(train_losses) < num_epochs:
        print(f"Warning: Plotting {len(train_losses)} epochs, but {num_epochs} were expected (possibly due to early stopping).")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'bo-', label='Training Loss')
    if val_losses and len(val_losses) == len(train_losses):
        ax.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    elif val_losses:
        print(f"Warning: Mismatch in length of train_losses ({len(train_losses)}) and val_losses ({len(val_losses)}). Not plotting validation loss.")

    ax.set_title('Training and Validation Loss (NLL)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_probability_vs_uncertainty(predictions: List[Dict[str, Any]], test_genomes: List[Any]):
    if not predictions:
        print("No predictions to plot.")
        return None

    probs = [p.get('probability', 0.5) for p in predictions]
    std_devs = [p.get('std_dev', 0.1) for p in predictions]
    num_key_events = [len(getattr(g, 'key_events', []) or []) for g in test_genomes]

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(probs, std_devs, c=num_key_events, cmap='viridis', alpha=0.7, s=[(n+1)*20 for n in num_key_events])
    ax.set_title('Predicted Scenario Probability vs. Uncertainty (Std Dev)')
    ax.set_xlabel('Predicted Probability of Scenario')
    ax.set_ylabel('Predicted Standard Deviation (Uncertainty)')
    cbar = plt.colorbar(scatter, ax=ax, label='Number of Key Events')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_interactive_scenario_space(predictions: List[Dict[str, Any]], test_genomes: List[Any]):
    if not predictions or len(predictions) != len(test_genomes):
        print("Prediction and genome lists are empty or mismatched.")
        return None

    data_for_df = []
    for genome, pred in zip(test_genomes, predictions):
        data_for_df.append({
            'id': getattr(genome, 'id', None),
            'probability': pred.get('probability'),
            'std_dev': pred.get('std_dev'),
            'consistency': pred.get('consistency_score', np.random.rand()),
            'num_tech_factors': len(getattr(genome, 'technological_factors', []) or []),
            'num_key_events': len(getattr(genome, 'key_events', []) or []),
            'primary_domain': (getattr(genome, 'domains_focused', ["N/A"])[0] if getattr(genome, 'domains_focused', None) else "N/A"),
            'timeline_duration': (
                int(getattr(genome, 'timeline', '0-0').split('-')[-1]) - int(getattr(genome, 'timeline', '0-0').split('-')[0])
                if '-' in getattr(genome, 'timeline', '0-0') and
                   getattr(genome, 'timeline', '0-0').split('-')[0].isdigit() and
                   getattr(genome, 'timeline', '0-0').split('-')[-1].isdigit()
                else 20
            )
        })
    df = pd.DataFrame(data_for_df)
    df.dropna(subset=['probability', 'std_dev', 'consistency'], inplace=True)
    if df.empty:
        print("DataFrame is empty after processing, cannot create plot.")
        return None

    fig = px.scatter_3d(
        df,
        x='probability',
        y='std_dev',
        z='consistency',
        color='primary_domain',
        size='num_key_events',
        hover_data=['id', 'timeline_duration', 'num_tech_factors'],
        title="Interactive Scenario Space Explorer"
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_simulation_run_variables(history: List[Dict[str, Any]], variables_to_plot: List[str]):
    if not history:
        print("No simulation history to plot.")
        return None

    df_data = {}
    df_data['year'] = [h['year'] for h in history]
    for var_name in variables_to_plot:
        if var_name in history[0]['variables']:
            df_data[var_name] = [h['variables'].get(var_name) for h in history]
        else:
            print(f"Warning: Variable '{var_name}' not found in simulation history.")

    df = pd.DataFrame(df_data)
    if df.empty or len(df.columns) <= 1:
        print("Not enough data to plot simulation variables.")
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    for var_name in variables_to_plot:
        if var_name in df.columns:
            ax.plot(df['year'], df[var_name], label=var_name, marker='o', linestyle='-', markersize=4)
    ax.set_title('Evolution of Key Simulation Variables Over Time (Single Run)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Variable Value')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def visualize_all_simulation_results(
    key_event_json_path: str = None,
    train_losses: List[float] = None,
    val_losses: Optional[List[float]] = None,
    num_epochs: int = 0,
    predictions: List[Dict[str, Any]] = None,
    test_genomes: List[Any] = None,
    sim_history: List[Dict[str, Any]] = None,
    variables_to_plot: List[str] = None
):
    """
    Calls all visualization routines and displays the plots at the end.
    Only calls each plot if the relevant data is provided.
    """
    figs = []

    # Key event timeline distribution
    if key_event_json_path:
        fig = plot_key_event_timeline_distribution(key_event_json_path)
        if fig: figs.append(fig)
        plt.show() if fig else None

    # Loss curves
    if train_losses is not None:
        fig = plot_loss_curves(train_losses, val_losses, num_epochs)
        if fig: figs.append(fig)
        plt.show() if fig else None

    # Probability vs. uncertainty
    if predictions is not None and test_genomes is not None:
        fig = plot_probability_vs_uncertainty(predictions, test_genomes)
        if fig: figs.append(fig)
        plt.show() if fig else None

    # Interactive scenario space (plotly)
    if predictions is not None and test_genomes is not None:
        fig = plot_interactive_scenario_space(predictions, test_genomes)
        if fig:
            fig.show()

    # Simulation run variables
    if sim_history is not None and variables_to_plot is not None:
        fig = plot_simulation_run_variables(sim_history, variables_to_plot)
        if fig: figs.append(fig)
        plt.show() if fig else None

    return figs