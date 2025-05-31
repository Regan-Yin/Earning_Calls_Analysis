# ISOM 3350 Final Project 
# Updating metadata.json
# Assist Analyze Calls Python code
# Author: Regan Yin
# Date: 2025-05-11

import os
import json
from datetime import datetime

# Define paths
transcript_dir = "transcript"
metadata_path = "metadata.json"

entries = []

# Gather entries from files
for filename in os.listdir(transcript_dir):
    if filename.endswith(".json"):
        try:
            parts = filename[:-5].split("_")
            if len(parts) < 3:
                raise ValueError("Filename format too short.")

            ticker = parts[0]
            event_date_raw = parts[-1]
            quarter_id = "_".join(parts[1:])

            # Validate and format date
            dt = datetime.strptime(event_date_raw, "%Y%m%d")
            event_date_formatted = dt.strftime("%Y-%m-%d")

            key = f"{ticker}_{quarter_id}"
            entry = {
                "key": key,
                "ticker": ticker,
                "event_date": event_date_formatted,
                "dt": dt  # Keep datetime object for sorting
            }

            entries.append(entry)

        except Exception as e:
            print(f"Error parsing {filename}: {e}")

# Sort entries by date
entries.sort(key=lambda x: x["dt"])

# Construct sorted metadata dictionary
metadata = {
    "_comment": [
        "ISOM 3350 Final Project",
        "Storing the analyzed companies info",
        "Assist Analyze Calls Python code",
        "Author: Regan Yin",
        "Date: 2025-05-11"
    ]
}
metadata.update({
    entry["key"]: {
        "ticker": entry["ticker"],
        "event_date": entry["event_date"]
    } for entry in entries
})

# Save to JSON
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("✅ metadata.json updated and sorted by date.")
