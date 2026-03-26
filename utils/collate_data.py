import os
import json
import hashlib
import re

# Directories
INPUT_DIR = "data"
OUTPUT_FILE = "data/output/master_corpus.json"

def normalize_microtext(text):
    """
    Skeleton for microtext normalization.
    You will expand this to handle SG slang, emojis, and abbreviations.
    """
    if not isinstance(text, str):
        return ""

    # 1. Expand standard contractions
    text = re.sub(r"\bcan't\b", "cannot", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdon't\b", "do not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bit's\b", "it is", text, flags=re.IGNORECASE)
    
    # 2. Domain-specific normalization (EVs)
    text = re.sub(r"\bev\b", "electric vehicle", text, flags=re.IGNORECASE)
    text = re.sub(r"\bice\b", "internal combustion engine", text, flags=re.IGNORECASE) # Crucial for car context!
    
    # 3. Singaporean context (Optional but scores creativity points)
    text = re.sub(r"\bcoe\b", "certificate of entitlement", text, flags=re.IGNORECASE)
    
    # 4. Emoji translation (Skeleton: you can use the 'emoji' library later)
    text = text.replace("🔥", " excellent ")
    text = text.replace("🔋", " battery ")
    text = text.replace("🗑️", " trash ")

    # 5. Noise Removal (URLs and mentions)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r"\@\w+", "", text) # Remove @mentions

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def generate_hash_id(text, platform):
    """
    Generates a unique SHA-256 hash based on the text and platform.
    This guarantees that exact duplicate texts get the exact same ID.
    """
    unique_string = f"{platform}_{text.lower()}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:16] # 16 chars is plenty

def compile_and_preprocess_datasets():
    all_records = {} # Using a dictionary to automatically handle duplicates via Hash ID
    files_processed = 0
    duplicates_dropped = 0
    too_short_dropped = 0
    
    print(f"Scanning '{INPUT_DIR}' for JSON files...\n")
    
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".json"):
            continue
            
        filepath = os.path.join(INPUT_DIR, filename)
        files_processed += 1
        
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {filename}. Skipping.")
                continue
                
            for row in data:
                raw_text = row.get("text", "")
                platform = row.get("platform", "Unknown")
                
                # 1. Normalize and Clean
                clean_text = normalize_microtext(raw_text)
                
                # 2. Length Filter (Must be > 3 words)
                if len(clean_text.split()) < 4:
                    too_short_dropped += 1
                    continue
                
                # 3. Generate Content-Based Hash ID
                doc_id = generate_hash_id(clean_text, platform)
                
                # 4. Check for duplicates
                if doc_id in all_records:
                    duplicates_dropped += 1
                    continue
                
                # 5. Build the standardized record
                # We extract the core fields, and let any platform-specific fields (stars, upvotes) ride along
                standardized_record = {
                    "id": doc_id,
                    "text": clean_text,
                    "platform": platform,
                    "source_target": row.get("source_target", "General"),
                    "dataset_split": "corpus" # Default tag for later
                }
                
                # Add platform-specific nullable fields dynamically
                for key, value in row.items():
                    if key not in["id", "doc_id", "text", "platform", "source_target"]:
                        standardized_record[key] = value
                
                # Save to dictionary
                all_records[doc_id] = standardized_record

    # Convert dictionary back to list for JSON export
    final_dataset = list(all_records.values())
    
    # Calculate word count to ensure you meet the 100,000 word rubric requirement
    total_words = sum(len(rec["text"].split()) for rec in final_dataset)
    
    # Save Master JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=4)
        
    print("--- Compilation Complete ---")
    print(f"Files processed:     {files_processed}")
    print(f"Duplicates dropped:  {duplicates_dropped}")
    print(f"Too short dropped:   {too_short_dropped}")
    print(f"Total Valid Records: {len(final_dataset)}")
    print(f"Total Word Count:    {total_words}")
    print(f"Saved to:            {OUTPUT_FILE}")
    
    if len(final_dataset) < 10000 or total_words < 100000:
        print("\n⚠️ WARNING: You have not met the assignment minimums yet! Keep scraping.")

if __name__ == "__main__":
    compile_and_preprocess_datasets()