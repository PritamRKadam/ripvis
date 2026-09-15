import csv

csv_path = "/home/cvl_project_ss26_dumitriu/Desktop/Videos/dataset_annotations.csv"
valid_count = 0
outliers = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fluid_regime = row.get("fluid_regime", "")
        if "non_water" in fluid_regime:
            outliers.append((row["id"], row["filename"], fluid_regime))
        else:
            valid_count += 1

print(f"Total rows in dataset_annotations.csv: {valid_count + len(outliers)}")
print(f"Valid aquatic clips: {valid_count}")
print(f"Flagged non-water outliers: {len(outliers)}")
for o in outliers:
    print(f"  ID {o[0]}: {o[1]} -> {o[2]}")
