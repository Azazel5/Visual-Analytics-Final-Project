# build_manifest.py

import os
import csv
import re

# Configuration
DATA_DIR     = "input_data/News Articles"             
OUTPUT_CSV   = "manifest.csv"   
DATE_PATTERN = re.compile(r"^PUBLISHED:\s*(\d{4}/\d{2}/\d{2})")

rows = []

for source in os.listdir(DATA_DIR):
    source_dir = os.path.join(DATA_DIR, source)
    if not os.path.isdir(source_dir):
        continue

    for fname in os.listdir(source_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(source_dir, fname)
        
        # Read the first non‐empty line
        with open(path, encoding="latin-1") as f:
            for i, raw in enumerate(f):
                line = raw.strip()
                if not line:
                    continue                     

                m = DATE_PATTERN.search(line)   
                if m:
                    date = m.group(1)
                    print("  → date:", date)
                    break                       

                if i >= 5:
                    break
        
        
        rows.append({
            "source": source,
            "filename": fname,
            "date": date
        })


with open(OUTPUT_CSV, "w", newline="", encoding="utf8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["source","filename","date"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Wrote {len(rows)} entries to {OUTPUT_CSV}")