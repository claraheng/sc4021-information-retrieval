# SC4021 Information Retrieval Group Project
## About
Our group built an opinion search engine on the topic of ... The system enables users to find relevant opinions about any instance of the topic.

## Main Tasks
- Crawling
- Indexing
- Classification

## How to Run
1. Open Docker application
2. Run `docker compose up -d` in terminal
3. View Solr dashboard at http://localhost:8983/solr/#/
4. Run `python utils/ingest_to_solr.py` to upload data to solr
5. Run `python app.py` to run web app