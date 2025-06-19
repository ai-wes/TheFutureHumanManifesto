import csv
import os

# Read the CSV file
file_path = r"c:\Users\wes\Downloads\24_Week_Dosage_Schedule.csv"
output_path = os.path.join(os.path.dirname(file_path), "simple_formatted_dosage.txt")

# Read the data
with open(file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)

# Create a formatted text output
with open(output_path, 'w') as f:
    # Write a title
    f.write("===== 24 WEEK DOSAGE SCHEDULE =====\n\n")
    
    # Group by week
    current_week = None
    for row in data:
        week = row['Week']
        
        if current_week != week:
            f.write(f"\n{'=' * 40}\n")
            f.write(f"WEEK {week}\n")
            f.write(f"{'=' * 40}\n")
            current_week = week
        
        # Format each entry
        f.write(f"Day: {row['Day']}\n")
        f.write(f"Medication: {row['Medication']}\n")
        f.write(f"Dose: {row['Dose_mg']} mg ({row['Units']} units)\n")
        f.write(f"Injection Site: {row['Injection_Site']}\n")
        f.write(f"{'-' * 40}\n")
    
    # Write a summary section
    f.write("\n\n===== DOSAGE PROGRESSION SUMMARY =====\n\n")
    
    # Summarize NAD+ progression
    f.write("NAD+ Dosage Progression:\n")
    nad_doses = {}
    for row in data:
        if row['Medication'] == 'NAD+':
            week = row['Week']
            if week not in nad_doses:
                nad_doses[week] = float(row['Dose_mg'])
    
    for week, dose in sorted(nad_doses.items()):
        f.write(f"  Week {week}: {dose} mg\n")
    
    # Summarize Tirzepatide progression
    f.write("\nTirzepatide Dosage Progression:\n")
    tirzepatide_doses = {}
    for row in data:
        if row['Medication'] == 'Tirzepatide':
            week = row['Week']
            tirzepatide_doses[week] = float(row['Dose_mg'])
    
    for week, dose in sorted(tirzepatide_doses.items()):
        f.write(f"  Week {week}: {dose} mg\n")

print(f"Simple formatted schedule saved to {output_path}") 