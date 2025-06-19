import json

# Paths to your files
file1 = "data/refined_scenarios_briefs_intermediate.json"
file2 = "data/refined_briefs_with_plausibility.json"

def extract_ids(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["original_scenario_id"] for item in data if "original_scenario_id" in item}

ids1 = extract_ids(file1)
ids2 = extract_ids(file2)

overlap = ids1 & ids2

print(f"IDs in {file1}: {len(ids1)}")
print(f"IDs in {file2}: {len(ids2)}")
print(f"Overlapping IDs: {len(overlap)}")
print("Overlapping scenario IDs:")
for oid in overlap:
    print(oid)