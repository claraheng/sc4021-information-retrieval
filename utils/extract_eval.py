import os
import json
import random
import pandas as pd
from textblob import TextBlob

MASTER_JSON_PATH = "data/output/master_corpus.json"
EVAL_CSV_PATH = "data/output/eval_workspace.csv"

def get_pre_labels(text):
    """
    Uses TextBlob to guess subjectivity and polarity.
    TextBlob returns:
    - polarity: -1.0 (Very Negative) to 1.0 (Very Positive)
    - subjectivity: 0.0 (Very Objective/Neutral) to 1.0 (Very Subjective/Opinionated)
    """
    blob = TextBlob(text)
    
    # Subjectivity Rule
    if blob.sentiment.subjectivity > 0.3: # Threshold for being an "opinion"
        subj_label = "Opinionated"
    else:
        subj_label = "Neutral"
        
    # Polarity Rule
    if blob.sentiment.polarity > 0.1:
        pol_label = "Positive"
    elif blob.sentiment.polarity < -0.1:
        pol_label = "Negative"
    else:
        pol_label = "Neutral"
        
    return subj_label, pol_label

def create_eval_set():
    print(f"Loading {MASTER_JSON_PATH}...")
    with open(MASTER_JSON_PATH, "r", encoding="utf-8") as f:
        all_records = json.load(f)
        
    # Ensure dataset is large enough
    if len(all_records) < 1000:
        print(f"Warning: Only {len(all_records)} records found. Sampling all of them.")
        sample_size = len(all_records)
    else:
        sample_size = 1000

    # 1. Randomly sample 1000 records
    # Using a fixed seed (e.g., 42) ensures you get the exact same 1000 records 
    # if you accidentally run the script twice.
    random.seed(42) 
    eval_sample = random.sample(all_records, sample_size)
    eval_ids = {rec["id"] for rec in eval_sample}

    # 2. Update the JSON data with the dataset_split and pre-labels
    eval_export_data =[]
    
    print("Pre-labeling 1,000 records with TextBlob... (This will be fast)")
    for record in all_records:
        if record["id"] in eval_ids:
            record["dataset_split"] = "eval"
            
            # Generate ML Guesses
            subj, pol = get_pre_labels(record["text"])
            
            # Prepare row for CSV export
            eval_export_data.append({
                "id": record["id"],
                "text": record["text"],
                "platform": record["platform"],
                "AI_Guess_Subjectivity": subj,
                "AI_Guess_Polarity": pol,
                # Blank columns for your team to fill out on Google Sheets!
                "Annotator_1_Subj": "",
                "Annotator_1_Pol": "",
                "Annotator_2_Subj": "",
                "Annotator_2_Pol": "",
            })
        else:
            record["dataset_split"] = "corpus"

    # 3. Save the updated JSON back to disk (to update Solr later)
    with open(MASTER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4)
        
    # 4. Export to CSV for Google Sheets
    df = pd.DataFrame(eval_export_data)
    df.to_csv(EVAL_CSV_PATH, index=False, encoding="utf-8-sig") # utf-8-sig helps Excel read emojis perfectly
    
    print(f"\n✅ Success!")
    print(f"Updated JSON saved with 'eval' and 'corpus' tags.")
    print(f"Exported {len(df)} records to {EVAL_CSV_PATH}")
    print("\nNext Steps:")
    print("1. Upload 'eval_workspace.csv' to Google Sheets.")
    print("2. Tell Annotator 1 and Annotator 2 to read the text. If the AI_Guess is correct, copy it to their column. If it's wrong, write the correct one!")

if __name__ == "__main__":
    create_eval_set()