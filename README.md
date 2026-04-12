# SC4021 Information Retrieval Group Project

## About
This project implements an EV opinion search engine pipeline end-to-end:

- Crawl and collect EV-related user content from multiple platforms
- Normalize and merge data into a master corpus
- Build subjectivity and polarity classifiers
- Index documents into Solr for search and filtering
- Serve an interactive Flask web app for retrieval and analysis

## Project Components
- Crawling: scripts under `scrapers/`
- Data processing and labeling utilities: scripts under `utils/`
- Classification and benchmarking: scripts under `classifiers/`
- Search backend: Apache Solr (Docker)
- Web frontend/API: Flask app in `app.py`

## Repository Structure
- `app.py`: Flask search application (runs on port 8080)
- `docker-compose.yml`: Solr service (`ev_reviews` core)
- `data/`: raw and processed data files
- `data/output/labeled_master_corpus_v2.json`: main labeled corpus currently used for ingestion/benchmarking
- `utils/ingest_to_solr.py`: loads labeled corpus into Solr
- `utils/collate_data.py`: merges and normalizes platform JSON files into `data/output/master_corpus.json`
- `utils/extract_eval.py`: samples evaluation workspace from corpus
- `classifiers/run_pipeline.py`: training + optional corpus prediction workflow
- `classifiers/benchmark_scalability.py`: throughput benchmark and graph generation

## Prerequisites

### System
- Python 3.10+ recommended
- Docker Desktop (or Docker Engine + Compose)

### Python packages
Install the core packages used by the app, utilities, and classifiers:

	pip install flask pysolr pandas numpy scikit-learn matplotlib joblib nltk openpyxl xlrd textblob

Optional packages for scraping scripts (install only if needed):

	pip install praw playwright beautifulsoup4 selenium webdriver-manager googletrans==4.0.0rc1 tqdm DrissionPage

If you use Playwright-based scrapers, also install browser binaries:

	playwright install

## Quick Start (Search App)

### 1) Create and activate a virtual environment

macOS/Linux:

	python3 -m venv .venv
	source .venv/bin/activate

### 2) Start Solr

	docker compose up -d

Solr Admin should be available at:
- http://localhost:8983/solr/#/

The compose file pre-creates the core:
- `ev_reviews`

### 3) Ingest data into Solr

	python utils/ingest_to_solr.py

This script loads:
- `data/output/labeled_master_corpus_v2.json`

### 4) Run the Flask app

	python app.py

Open:
- http://localhost:8080

## Classification Pipeline

### Train models and run corpus prediction

From project root:

	python classifiers/run_pipeline.py --retrain

What it does:
- Builds eval dataframe from `data/output/master_corpus.json` + `data/output/eval.xls`
- Trains subjectivity and polarity models (saved under `classifiers/models/`)
- Runs inference on corpus split (unless `--skip-predict` is provided)

Useful flags:
- `--retrain`: force retraining
- `--skip-predict`: train/evaluate only
- `--output <path>`: custom predictions CSV output path

## Scalability Benchmark

### Combined inference benchmark (subjectivity -> polarity routing)

	python classifiers/benchmark_scalability.py --master-corpus data/output/labeled_master_corpus_v2.json

Default tested record counts are:
- 500,1000,2000,5000,8000,10000

To test larger counts:

	python classifiers/benchmark_scalability.py --master-corpus data/output/labeled_master_corpus_v2.json --sizes 1,2,3,4,5,6

Outputs:
- `classifiers/scalability_results.csv`
- `classifiers/scalability_graph.png`

## Optional Data Preparation Workflow

If you need to rebuild corpus artifacts from raw platform JSON files:

1. Compile and normalize:

	   python utils/collate_data.py

2. Generate eval sampling workspace:

	   python utils/extract_eval.py

3. Perform manual annotation and export final labeled data

4. Re-ingest into Solr:

	   python utils/ingest_to_solr.py

## Credits

- YouTube data scraping support was adapted with reference to [yt-comments-extractor by vijaykumarpeta](https://github.com/vijaykumarpeta/yt-comments-extractor#).
- Please refer to the source repository for upstream usage details and license terms.

## Troubleshooting

- Solr connection errors:
  - Ensure Docker is running and `docker compose up -d` completed successfully.
  - Check Solr core exists: `ev_reviews`.

- NLTK resource downloads on first run:
  - `classifiers/preprocessing.py` downloads required corpora automatically.

- Model loading/version mismatch:
  - `classifiers/benchmark_scalability.py` includes fallback retraining behavior if serialized model loading fails.

## Team Notes

Recommended day-to-day run order:

1. `docker compose up -d`
2. `python utils/ingest_to_solr.py`
3. `python app.py`

For ML experiments:

1. `python classifiers/run_pipeline.py --retrain`
2. `python classifiers/benchmark_scalability.py --master-corpus data/output/labeled_master_corpus.json`