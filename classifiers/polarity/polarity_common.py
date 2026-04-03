from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_data(path: str) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            # In this project, eval_sample.xls is tab-separated text.
            df = pd.read_csv(input_path, sep="\t", encoding="latin1")
        except Exception as exc:
            raise ValueError(
                f"Could not read '{input_path}' as tab-separated text. If it is comma-separated, rename it to .csv."
            ) from exc
    else:
        df = pd.read_csv(input_path)

    required_cols = ["text", "subjectivity", "polarity"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["subjectivity"] = df["subjectivity"].fillna("").astype(str).str.strip().str.lower()
    df["polarity"] = df["polarity"].fillna("").astype(str).str.strip().str.lower()
    df = df[df["text"] != ""].copy()

    return df


def prepare_opinionated_data(df: pd.DataFrame) -> pd.DataFrame:
    opinion_df = df[df["subjectivity"] == "opinionated"].copy()
    valid_labels = ["positive", "negative", "neutral"]
    opinion_df = opinion_df[opinion_df["polarity"].isin(valid_labels)].copy()

    if opinion_df.empty:
        raise ValueError("No opinionated rows with valid polarity labels found.")

    return opinion_df


def build_model() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10000,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])