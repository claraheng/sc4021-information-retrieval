import csv
import json
from pathlib import Path

INPUT_FILE = Path("/Users/claraheng/Documents/GitHub/sc4021-information-retrieval/scrapers/twitter_daily_merged_deduped.csv")  # change if needed
OUTPUT_FILE = Path("/Users/claraheng/Documents/GitHub/sc4021-information-retrieval/scrapers/twitter.json")


def convert_csv_to_json(input_path, output_path):
    data = []

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = row.get("text", "").strip()

            if not text:
                continue

            data.append({
                "platform": "twitter",
                "text": text
            })

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(data)} rows")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    convert_csv_to_json(INPUT_FILE, OUTPUT_FILE)