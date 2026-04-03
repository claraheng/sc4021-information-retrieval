import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

from polarity_common import load_data, prepare_opinionated_data, build_model


CLASSIFIERS_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = CLASSIFIERS_DIR / "data" / "eval_sample.xls"
DEFAULT_MODEL_PATH = Path(__file__).with_name("polarity_model.joblib")


def train_polarity_model(df: pd.DataFrame):
    X = df["text"]
    y = df["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    # Measure scalability: predictions on test set
    start_time = time.time()
    preds = model.predict(X_test)
    prediction_time = time.time() - start_time

    print("\n=== Polarity Detection Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Scalability Metric: Records classified per second
    num_records = len(X_test)
    records_per_second = num_records / prediction_time
    print(f"\n=== Scalability Metrics ===")
    print(f"Total records classified: {num_records}")
    print(f"Classification time: {prediction_time:.4f} seconds")
    print(f"Records classified per second: {records_per_second:.2f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a polarity classifier from a labeled file with text, subjectivity, and polarity columns."
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_DATA_PATH),
        help="Path to the labelled file (.xls/.xlsx tab text, or .csv)"
    )
    parser.add_argument(
        "--model-out",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save trained model (.joblib)"
    )
    args = parser.parse_args()

    # 1. Load full dataset
    df = load_data(args.input_csv)

    # 2. Keep only opinionated rows
    opinion_df = prepare_opinionated_data(df)

    print("Total rows in dataset:", len(df))
    print("Opinionated rows used for polarity training:", len(opinion_df))
    print("\nPolarity label distribution:")
    print(opinion_df["polarity"].value_counts())

    # 3. Train model
    model = train_polarity_model(opinion_df)

    joblib.dump(model, args.model_out)
    print(f"\nSaved model to '{args.model_out}'")