import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import os

# Read the CSV file
file_path = r"c:\Users\wes\Downloads\24_Week_Dosage_Schedule.csv"
df = pd.read_csv(file_path)

# Create a formatted text output
output_path = os.path.join(os.path.dirname(file_path), "formatted_dosage_schedule.txt")

# Function to create a nice text table with PrettyTable
def create_text_table(df):
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    
    # Add rows
    for _, row in df.iterrows():
        table.add_row(row.tolist())
    
    # Make it look nicer
    table.align = "l"  # Left align
    table.border = True
    table.hrules = 1  # Show horizontal lines between rows
    
    return table

# Generate text output
with open(output_path, 'w') as f:
    # First summary by week
    f.write("===== DOSAGE SCHEDULE SUMMARY =====\n\n")
    
    # Group by week and medication to summarize
    summary_df = df.groupby(['Week', 'Medication']).agg({
        'Dose_mg': 'first',
        'Day': lambda x: ', '.join(sorted(set(x)))
    }).reset_index()
    
    current_week = None
    for _, row in summary_df.iterrows():
        if current_week != row['Week']:
            f.write(f"\nWEEK {row['Week']}:\n")
            current_week = row['Week']
        
        f.write(f"  - {row['Medication']}: {row['Dose_mg']} mg on {row['Day']}\n")
    
    # Then detailed table
    f.write("\n\n===== DETAILED DOSAGE SCHEDULE =====\n\n")
    table = create_text_table(df)
    f.write(str(table))

print(f"Formatted schedule saved to {output_path}")

# Create a visualization of the dosage changes over time
plt.figure(figsize=(12, 8))

# Plot NAD+ dosage
nad_df = df[df['Medication'] == 'NAD+'].drop_duplicates(['Week', 'Day', 'Dose_mg'])
plt.plot(nad_df['Week'], nad_df['Dose_mg'], 'b-o', label='NAD+ (mg)')

# Plot Tirzepatide dosage
tirzepatide_df = df[df['Medication'] == 'Tirzepatide']
plt.plot(tirzepatide_df['Week'], tirzepatide_df['Dose_mg'], 'r-o', label='Tirzepatide (mg)')

# Add labels and title
plt.xlabel('Week')
plt.ylabel('Dosage (mg)')
plt.title('24-Week Dosage Schedule')
plt.grid(True)
plt.legend()

# Save the plot
plot_path = os.path.join(os.path.dirname(file_path), "dosage_schedule_chart.png")
plt.savefig(plot_path)
plt.close()

print(f"Dosage chart saved to {plot_path}") 