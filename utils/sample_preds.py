import pandas as pd
from pathlib import Path

# Paths
UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "output"
PREDICTIONS_FILE = DATA_DIR / "predictions.csv"
SAMPLE_FILE = DATA_DIR / "sample_preds.csv"

def sample_predictions(n=100):
    if not PREDICTIONS_FILE.exists():
        print(f"❌ Error: {PREDICTIONS_FILE} not found. Run the pipeline first!")
        return

    print(f"Reading predictions from {PREDICTIONS_FILE}...")
    df = pd.read_csv(PREDICTIONS_FILE)

    if len(df) <= n:
        print(f"⚠️  Dataset size ({len(df)}) is less than or equal to sample size ({n}). Saving all records.")
        sample_df = df
    else:
        print(f"Sampling {n} random records...")
        sample_df = df.sample(n=n, random_state=42)

    sample_df.to_csv(SAMPLE_FILE, index=False)
    print(f"✅ Sample saved to {SAMPLE_FILE}")
    
    print("\nPreview of sampled records:")
    print(sample_df[["id", "subjectivity", "polarity"]].head())

if __name__ == "__main__":
    sample_predictions()
