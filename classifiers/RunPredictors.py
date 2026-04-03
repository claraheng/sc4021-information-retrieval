import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CLASSIFIERS_DIR = ROOT_DIR

SUBJECTIVITY_PREDICTOR = CLASSIFIERS_DIR / "subjectivity" / "SubjectivityPredictor.py"
POLARITY_PREDICTOR = CLASSIFIERS_DIR / "polarity" / "PolarityPredictor.py"


def run_command(command):
    print("\n$", " ".join(str(c) for c in command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run both subjectivity and polarity predictors."
    )
    parser.add_argument(
        "--input-csv",
        default=str(CLASSIFIERS_DIR / "data" / "texts_to_predict.csv"),
        help="Path to input CSV containing a 'text' column",
    )
    parser.add_argument(
        "--predictions-csv",
        default=str(CLASSIFIERS_DIR / "data" / "predictions.csv"),
        help="Path to shared predictions CSV",
    )
    args = parser.parse_args()

    run_command([
        sys.executable,
        str(SUBJECTIVITY_PREDICTOR),
        "--input-csv",
        args.input_csv,
        "--output-csv",
        args.predictions_csv,
    ])

    run_command([
        sys.executable,
        str(POLARITY_PREDICTOR),
        "--input-csv",
        args.predictions_csv,
        "--output-csv",
        args.predictions_csv,
    ])

    print("\nAll predictors finished successfully.")
