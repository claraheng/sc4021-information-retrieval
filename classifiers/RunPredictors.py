import argparse
import subprocess
import sys
from pathlib import Path
import re
import time
from datetime import datetime

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
CLASSIFIERS_DIR = ROOT_DIR

SUBJECTIVITY_PREDICTOR = CLASSIFIERS_DIR / "SubjectivityPredictor.py"
POLARITY_PREDICTOR = CLASSIFIERS_DIR / "PolarityPredictor.py"
REPORT_PATH = CLASSIFIERS_DIR / "report.md"


def run_command(command):
    print("\n$", " ".join(str(c) for c in command))
    started = time.time()
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    elapsed = time.time() - started

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)

    return result, elapsed


def _extract_first_int(pattern: str, text: str):
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _append_prediction_report(input_csv: str, predictions_csv: str, subj_stdout: str, pol_stdout: str, subj_elapsed: float, pol_elapsed: float):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    opinionated_predicted = _extract_first_int(r"Opinionated rows predicted:\s*(\d+)", pol_stdout)

    pred_path = Path(predictions_csv)
    total_rows = None
    opinionated_rows = None

    if pred_path.exists():
        df = pd.read_csv(pred_path)
        total_rows = len(df)

        if "subjectivity" in df.columns:
            subj = df["subjectivity"].fillna("").astype(str).str.strip().str.lower()
            opinionated_rows = int((subj == "opinionated").sum())

    lines = [
        "",
        "## RunPredictors",
        f"- Timestamp: {now}",
        f"- Input text CSV: {input_csv}",
        f"- Output predictions CSV: {predictions_csv}",
        "",
        "### Prediction Pipeline Summary",
        "- Stage 1: Subjectivity prediction for all input rows.",
        "- Stage 2: Polarity prediction only for rows predicted as opinionated.",
        "",
        "### Prediction Process Metrics",
        f"- End-to-end subjectivity predictor runtime: {subj_elapsed:.4f} s",
        f"- End-to-end polarity predictor runtime: {pol_elapsed:.4f} s",
        f"- Total rows in predictions file: {total_rows if total_rows is not None else 'N/A'}",
        f"- Opinionated rows in predictions file: {opinionated_rows if opinionated_rows is not None else 'N/A'}",
        f"- Opinionated rows predicted by polarity stage: {opinionated_predicted if opinionated_predicted is not None else 'N/A'}",
        "",
        "### Notes",
        "- Prediction run does not compute precision/recall/F1 by itself because labels are not available for new unlabeled input rows.",
        "- Use the training section above (RunClassifiers) for model evaluation metrics on held-out labeled data.",
    ]

    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nUpdated report: {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run both subjectivity and polarity predictors."
    )
    parser.add_argument(
        "--input-csv",
        default=str(CLASSIFIERS_DIR / "texts_to_predict.csv"),
        help="Path to input CSV containing a 'text' column",
    )
    parser.add_argument(
        "--predictions-csv",
        default=str(CLASSIFIERS_DIR / "predictions.csv"),
        help="Path to shared predictions CSV",
    )
    args = parser.parse_args()

    subj_result, subj_elapsed = run_command([
        sys.executable,
        str(SUBJECTIVITY_PREDICTOR),
        "--input-csv",
        args.input_csv,
        "--output-csv",
        args.predictions_csv,
    ])

    pol_result, pol_elapsed = run_command([
        sys.executable,
        str(POLARITY_PREDICTOR),
        "--input-csv",
        args.predictions_csv,
        "--output-csv",
        args.predictions_csv,
    ])

    _append_prediction_report(
        input_csv=args.input_csv,
        predictions_csv=args.predictions_csv,
        subj_stdout=subj_result.stdout or "",
        pol_stdout=pol_result.stdout or "",
        subj_elapsed=subj_elapsed,
        pol_elapsed=pol_elapsed,
    )

    print("\nAll predictors finished successfully.")
