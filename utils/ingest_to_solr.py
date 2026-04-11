import pysolr
import json
import os

# Connect to the Solr core we created in docker-compose
SOLR_URL = 'http://localhost:8983/solr/ev_reviews'
solr = pysolr.Solr(SOLR_URL, always_commit=True)

MASTER_JSON_PATH = os.path.join(os.path.dirname(__file__), '../data/output/labeled_master_corpus_v2.json')

def ingest_data():
    if not os.path.exists(MASTER_JSON_PATH):
        print(f"Error: {MASTER_JSON_PATH} not found!")
        return
        
    with open(MASTER_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} records. Wiping existing Solr index...")
    solr.delete(q='*:*') # Clear the DB before inserting to avoid duplicates
    
    print("Ingesting data to Solr...")
    # Solr will automatically define the schema based on your JSON keys!
    solr.add(data) 
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest_data()