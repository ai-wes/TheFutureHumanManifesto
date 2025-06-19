import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

def parse_12_pillars_data(file_path):
    """
    Parses the 12 pillars data from the markdown file.
    It specifically looks for a markdown table with 'Pillar' and 'Current Level' columns.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

    data = {}
    # Find the table header
    header_index = -1
    for i, line in enumerate(lines):
        if '| Pillar' in line and '| Current Level' in line:
            header_index = i
            break
    
    if header_index == -1:
        print("Could not find the 12 Pillars data table in the file.")
        return None

    # Start parsing from after the header and separator
    for line in lines[header_index + 2:]:
        line = line.strip()
        if not line.startswith('|'):
            # Stop if we're no longer in the table
            break
        
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 2:
            pillar = parts[0]
            level_str = parts[1].replace('%', '')
            if level_str.isdigit():
                data[pillar] = int(level_str)

    if not data:
        print("Found table but could not parse any data rows.")
        return None
        
    return data

def create_styled_spider_graph(data, output_path):
    """
    Creates a styled spider graph from the given data and saves it to a file.
    The style is inspired by a dark, futuristic aesthetic.
    """
    labels = list(data.keys())
    values = list(data.values())
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    values += values[:1]
    angles += angles[:1]

    # --- Styling ---
    bg_color = '#1a1a1a'
    line_color = '#00ffff'
    fill_color = '#00ffff'
    text_color = '#f0f0f0'
    grid_color = '#404040'

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color=line_color, zorder=3)
    ax.fill(angles, values, color=fill_color, alpha=0.2, zorder=2)

    # Draw one axe per variable + add labels. Adjusted for more labels.
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, size=11, color=text_color, zorder=3)

    # Set the grid and tick properties
    ax.tick_params(axis='x', which='major', pad=25)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_rlabel_position(90)

    # Style grid lines
    ax.yaxis.grid(color=grid_color, linestyle='--', linewidth=1, zorder=1)
    ax.xaxis.grid(color=grid_color, linestyle='--', linewidth=1, zorder=1)

    # Style tick labels (the radial values)
    y_ticks = [25, 50, 75, 100]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y}%" for y in y_ticks], color=grid_color)
    plt.ylim(0, 100)

    # Add a title
    plt.title('12 Pillars: Current State Assessment', size=24, color=text_color, y=1.1, weight='bold')

    # Remove outer spine
    ax.spines['polar'].set_color('none')
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor=bg_color)
    print(f"Styled spider graph saved to {output_path}")

if __name__ == "__main__":
    # Assumes the script is run from the root of TheFutureHumanManifesto project
    project_root = Path(__file__).resolve().parent.parent
    md_file = project_root / "docs" / "12-pillars.md"
    output_file = project_root / "docs" / "12_pillars_spider_graph.png"
    
    graph_data = parse_12_pillars_data(md_file)
    if graph_data:
        print("Successfully parsed data:", graph_data)
        try:
            create_styled_spider_graph(graph_data, output_file)
        except Exception as e:
            print(f"An error occurred while creating the graph: {e}")
            print("Please ensure you have 'matplotlib' installed. You can install it with: pip install matplotlib")
    else:
        print("Failed to parse data from the markdown file. Cannot generate graph.") 