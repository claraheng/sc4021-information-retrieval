import csv
import json

csv_file = "/Users/claraheng/Documents/GitHub/sc4021-information-retrieval/scrapers/twitter_data_final.csv"
jsonl_file = "/Users/claraheng/Documents/GitHub/sc4021-information-retrieval/scrapers/twitter_data_final.jsonl"

with open(csv_file, "r", encoding="utf-8") as csvf, open(jsonl_file, "w", encoding="utf-8") as jsonf:
    reader = csv.DictReader(csvf)
    
    for row in reader:
        jsonf.write(json.dumps(row) + "\n")

print("CSV converted to JSONL.")