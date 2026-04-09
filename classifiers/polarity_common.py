from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import preprocess_text

def load_data(path: str) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            engine = "xlrd" if input_path.suffix.lower() == ".xls" else "openpyxl"
            df = pd.read_excel(input_path, engine=engine)
        except Exception:
            try:
                df = pd.read_csv(input_path, sep="\t", encoding="latin1")
            except Exception:
                try:
                    df = pd.read_csv(input_path, encoding="latin1")
                except Exception as exc:
                    raise ValueError(
                        f"Could not read '{input_path}' as Excel or delimited text."
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

    opinion_df["clean_text"] = opinion_df["text"].apply(preprocess_text)
    return opinion_df


def build_model() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])