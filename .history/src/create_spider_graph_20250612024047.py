import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

def parse_spider_graph_data(file_path):
    """
    Parses the spider graph data from the given markdown file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find label and value.
    pattern = r"^(.*?)\s*Progress towards breakthrough:\s*(\d+)%"
    matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)

    # Clean up labels and convert values to int
    data = {label.strip(): int(value) for label, value in matches}
    return data

def create_spider_graph(data, output_path):
    """
    Creates a spider graph from the given data and saves it to a file.
    """
    labels = list(data.keys())
    values = list(data.values())
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles_closed, values_closed, linewidth=2, linestyle='solid', label='Progress')
    ax.fill(angles_closed, values_closed, alpha=0.25)

    # Draw one axe per variable + add labels
    ax.set_thetagrids(np.degrees(angles), labels, size=12)

    # Draw ylabels
    ax.set_rlabel_position(30)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=10)
    plt.ylim(0, 100)

    # Add a title
    plt.title('Progress Towards Breakthrough', size=20, color='black', y=1.1)
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Spider graph saved to {output_path}")
    
    # To prevent showing the plot in a non-GUI environment which can cause a crash
    if __name__ == "__main__":
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")


if __name__ == "__main__":
    # Assumes the script is run from the root of TheFutureHumanManifesto project
    project_root = Path(__file__).parent.parent
    md_file = project_root / "docs" / "spider_graph.md"
    output_file = project_root / "docs" / "spider_graph.png"
    
    if not md_file.exists():
        print(f"Error: {md_file} not found.")
        print("Please make sure the script is in the 'src' directory and 'docs/spider_graph.md' exists.")
    else:
        graph_data = parse_spider_graph_data(md_file)
        if graph_data:
            print("Found data:", graph_data)
            create_spider_graph(graph_data, output_file)
        else:
            print("Could not parse data from the markdown file.") 