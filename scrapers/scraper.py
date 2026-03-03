import asyncio
import pysolr
from twitter_scraper import scrape_x_to_solr
from reddit_scraper import scrape_reddit_to_solr
# Solr connection setup (assuming your Docker container is running)
SOLR_URL = 'http://localhost:8983/solr/opinion_engine'
solr = pysolr.Solr(SOLR_URL, always_commit=True)

def main():
    topic = "Electric Vehicles"
    all_documents =[]
    
    # 1. Scrape Reddit
    try:
        reddit_docs = scrape_reddit_to_solr(topic, limit=50)
        all_documents.extend(reddit_docs)
    except Exception as e:
        print(f"Reddit scraping failed: {e}")

    # 2. Scrape X (Twitter)
    try:
        # Since scrape_x_to_solr is async, we run it via asyncio
        x_docs = asyncio.run(scrape_x_to_solr(topic, limit=50))
        all_documents.extend(x_docs)
    except Exception as e:
        print(f"X scraping failed: {e}")

    # 3. Push all data to Solr
    if all_documents:
        print(f"Indexing {len(all_documents)} total documents into Solr...")
        solr.add(all_documents)
        print("Ingestion Complete! Data is now searchable in Solr.")
    else:
        print("No documents were scraped. Check your API keys and credentials.")

if __name__ == "__main__":
    main()