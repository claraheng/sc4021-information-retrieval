from pathlib import Path

import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


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


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def prepare_subjectivity_data(df: pd.DataFrame) -> pd.DataFrame:
    subjectivity_df = df[df["subjectivity"].isin(["neutral", "opinionated"])].copy()
    if subjectivity_df.empty:
        raise ValueError("No rows with valid subjectivity labels found.")

    subjectivity_df["clean_text"] = subjectivity_df["text"].apply(preprocess_text)
    return subjectivity_df


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(random_state=42)),
    ])