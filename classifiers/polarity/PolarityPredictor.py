import argparse
from pathlib import Path

import joblib
import pandas as pd


CLASSIFIERS_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = Path(__file__).with_name("polarity_model.joblib")
DEFAULT_INPUT_TEXTS_PATH = CLASSIFIERS_DIR / "data" / "predictions.csv"
DEFAULT_OUTPUT_PATH = CLASSIFIERS_DIR / "data" / "predictions.csv"


def load_prediction_table(input_csv: str) -> pd.DataFrame:
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Input text file not found: {path}")

    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="latin1")
        except Exception:
            df = pd.read_csv(path, sep="\t", encoding="latin1")

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")
    if "subjectivity" not in df.columns:
        raise ValueError("Input CSV must contain a 'subjectivity' column.")
    if "subjectivity_confidence" not in df.columns:
        df["subjectivity_confidence"] = ""
    if "polarity" not in df.columns:
        df["polarity"] = ""
    if "polarity_confidence" not in df.columns:
        df["polarity_confidence"] = ""

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["subjectivity"] = df["subjectivity"].fillna("").astype(str).str.strip().str.lower()
    df["polarity"] = df["polarity"].fillna("").astype(str).str.strip().str.lower()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a trained polarity model and fill polarity labels for opinionated rows."
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained model (.joblib)",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_TEXTS_PATH),
        help="CSV file containing 'text', 'subjectivity', and 'polarity' columns",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to save predictions",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run PolarityClassifier.py first to train and save one."
        )

    model = joblib.load(model_path)
    df = load_prediction_table(args.input_csv)

    opinionated_mask = df["subjectivity"] == "opinionated"
    opinionated_texts = df.loc[opinionated_mask, "text"]

    if not opinionated_texts.empty:
        predictions = model.predict(opinionated_texts)
        probabilities = model.predict_proba(opinionated_texts)
        confidences = probabilities.max(axis=1).round(4)
        df.loc[opinionated_mask, "polarity"] = predictions
        df.loc[opinionated_mask, "polarity_confidence"] = confidences

    output_df = df[[
        "text",
        "subjectivity",
        "subjectivity_confidence",
        "polarity",
        "polarity_confidence",
    ]].copy()
    output_df.to_csv(args.output_csv, index=False)

    print(f"Saved predictions to '{args.output_csv}'")
    print(f"Opinionated rows predicted: {int(opinionated_mask.sum())}")
    print(output_df.head(10).to_string(index=False))
