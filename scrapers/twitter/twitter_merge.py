import csv
from pathlib import Path

INPUT_DIR = Path(__file__).parent / "twitter_daily_runs"
OUTPUT_FILE = Path(__file__).parent / "twitter_daily_merged_deduped.csv"

all_rows = []
seen_ids = set()

csv_files = sorted(INPUT_DIR.glob("*.csv"))

for file in csv_files:
    print(f"Reading {file.name}")
    with file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("id", "").strip()
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            all_rows.append({
                "id": tid,
                "author": row.get("author", "").strip(),
                "text": row.get("text", "").strip(),
                "url": row.get("url", "").strip(),
                "query": row.get("query", "").strip(),
            })

with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "author", "text", "url", "query"])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nMerged files: {len(csv_files)}")
print(f"Unique tweets: {len(all_rows)}")
print(f"Saved to: {OUTPUT_FILE}")