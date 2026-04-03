import argparse
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from subjectivity_common import build_model, load_data, prepare_subjectivity_data


CLASSIFIERS_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = CLASSIFIERS_DIR / "data" / "eval_sample.xls"
DEFAULT_MODEL_PATH = Path(__file__).with_name("subjectivity_model.joblib")


def train_subjectivity_model(df: pd.DataFrame):
    X = df["clean_text"]
    y = df["subjectivity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    pipeline = build_model()
    pipeline.fit(X_train, y_train)

    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    classification_time = time.time() - start_time

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    num_records = len(X_test)
    records_per_second = num_records / classification_time
    print("\n=== Scalability Metrics ===")
    print(f"Total records classified: {num_records}")
    print(f"Classification time: {classification_time:.4f} seconds")
    print(f"Records classified per second: {records_per_second:.2f}")

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a subjectivity classifier from labeled data."
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to labeled input (.xls/.xlsx as tab text, or .csv)",
    )
    parser.add_argument(
        "--model-out",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save trained model (.joblib)",
    )
    args = parser.parse_args()

    df = load_data(args.input_file)
    subjectivity_df = prepare_subjectivity_data(df)

    print("Dataset preview:")
    print(subjectivity_df.head(10))
    print(f"\nDataset shape: {subjectivity_df.shape}")
    print("\nSubjectivity label distribution:")
    print(subjectivity_df["subjectivity"].value_counts())

    model = train_subjectivity_model(subjectivity_df)
    joblib.dump(model, args.model_out)
    print(f"\nSaved model to '{args.model_out}'")
