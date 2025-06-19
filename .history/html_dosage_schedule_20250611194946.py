import csv
import os
import webbrowser

# Read the CSV file
file_path = r"c:\Users\wes\Downloads\24_Week_Dosage_Schedule.csv"
output_path = os.path.join(os.path.dirname(file_path), "dosage_schedule.html")

# Read the data
with open(file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)

# Create HTML output
with open(output_path, 'w') as f:
    # Write HTML header
    f.write('''<!DOCTYPE html>
<html>
<head>
    <title>24-Week Dosage Schedule</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 { 
            color: #333; 
            text-align: center;
        }
        .week-container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 15px;
        }
        .week-header {
            background-color: #2c3e50;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .medication-row {
            display: flex;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .medication-row:last-child {
            border-bottom: none;
        }
        .medication-cell {
            flex: 1;
            padding: 8px;
        }
        .medication-header {
            font-weight: bold;
            background-color: #f0f0f0;
        }
        .summary-section {
            margin-top: 40px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .summary-card {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .nad-dose {
            color: #2980b9;
        }
        .tirzepatide-dose {
            color: #c0392b;
        }
    </style>
</head>
<body>
    <h1>24-Week Dosage Schedule</h1>
''')

    # Group by week
    weeks = {}
    for row in data:
        week = row['Week']
        if week not in weeks:
            weeks[week] = []
        weeks[week].append(row)

    # Write week sections
    for week, entries in sorted(weeks.items(), key=lambda x: int(x[0])):
        f.write(f'''
    <div class="week-container">
        <div class="week-header">
            <h2>Week {week}</h2>
        </div>
        
        <div class="medication-row medication-header">
            <div class="medication-cell">Day</div>
            <div class="medication-cell">Medication</div>
            <div class="medication-cell">Dose (mg)</div>
            <div class="medication-cell">Units</div>
            <div class="medication-cell">Injection Site</div>
        </div>
''')
        
        for entry in entries:
            f.write(f'''
        <div class="medication-row">
            <div class="medication-cell">{entry['Day']}</div>
            <div class="medication-cell">{entry['Medication']}</div>
            <div class="medication-cell">{entry['Dose_mg']}</div>
            <div class="medication-cell">{entry['Units']}</div>
            <div class="medication-cell">{entry['Injection_Site']}</div>
        </div>
''')
        
        f.write('    </div>\n')

    # Create summary section
    f.write('''
    <div class="summary-section">
        <h2>Dosage Progression Summary</h2>
        
        <div class="summary-card">
            <h3>NAD+ Dosage Progression</h3>
''')
    
    # NAD+ progression
    nad_doses = {}
    for row in data:
        if row['Medication'] == 'NAD+':
            week = row['Week']
            day = row['Day']
            key = f"{week}-{day}"
            if key not in nad_doses:
                nad_doses[key] = {
                    'week': week,
                    'day': day,
                    'dose': float(row['Dose_mg'])
                }
    
    # Sort by week and day
    sorted_nad = sorted(nad_doses.values(), key=lambda x: (int(x['week']), x['day']))
    last_week = None
    for entry in sorted_nad:
        if last_week != entry['week']:
            if last_week is not None:
                f.write('            </ul>\n')
            f.write(f'            <p><strong>Week {entry["week"]}:</strong></p>\n')
            f.write('            <ul>\n')
            last_week = entry['week']
        
        f.write(f'                <li>{entry["day"]}: <span class="nad-dose">{entry["dose"]} mg</span></li>\n')
    
    f.write('            </ul>\n')
    
    # Tirzepatide progression
    f.write('''
        </div>
        
        <div class="summary-card">
            <h3>Tirzepatide Dosage Progression</h3>
''')
    
    tirzepatide_doses = {}
    for row in data:
        if row['Medication'] == 'Tirzepatide':
            week = row['Week']
            tirzepatide_doses[week] = float(row['Dose_mg'])
    
    for week, dose in sorted(tirzepatide_doses.items(), key=lambda x: int(x[0])):
        f.write(f'            <p><strong>Week {week}:</strong> <span class="tirzepatide-dose">{dose} mg</span></p>\n')
    
    # Close HTML
    f.write('''
        </div>
    </div>
</body>
</html>
''')

print(f"HTML schedule saved to {output_path}")

# Open the HTML file in the default browser
webbrowser.open('file://' + os.path.realpath(output_path)) 