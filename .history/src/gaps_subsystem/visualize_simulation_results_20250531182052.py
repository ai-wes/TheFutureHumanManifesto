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

# Example usage:
# plot_key_event_timeline_distribution("data/synthetic_scenarios_generated.json")