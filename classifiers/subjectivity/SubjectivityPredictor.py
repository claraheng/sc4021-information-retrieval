import argparse
from pathlib import Path

import joblib
import pandas as pd

from subjectivity_common import preprocess_text


CLASSIFIERS_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = Path(__file__).with_name("subjectivity_model.joblib")
DEFAULT_INPUT_TEXTS_PATH = CLASSIFIERS_DIR / "data" / "texts_to_predict.csv"
DEFAULT_OUTPUT_PATH = CLASSIFIERS_DIR / "data" / "predictions.csv"


def load_input_texts(input_csv: str) -> pd.Series:
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

    return df["text"].fillna("").astype(str).str.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a trained subjectivity model and predict labels for new texts."
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained model (.joblib)",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_TEXTS_PATH),
        help="CSV file containing a 'text' column",
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
            f"Model file not found: {model_path}. Run SubjectivityClassifier.py first to train and save one."
        )

    model = joblib.load(model_path)
    texts = load_input_texts(args.input_csv)
    clean_texts = texts.apply(preprocess_text)

    predictions = model.predict(clean_texts)
    probabilities = model.predict_proba(clean_texts)
    confidences = probabilities.max(axis=1).round(4)
    output_df = pd.DataFrame(
        {
            "text": texts,
            "subjectivity": predictions,
            "subjectivity_confidence": confidences,
            "polarity": "",
            "polarity_confidence": "",
        }
    )
    output_df.to_csv(args.output_csv, index=False)

    print(f"Saved predictions to '{args.output_csv}'")
    print(output_df.head(10).to_string(index=False))
