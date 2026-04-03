import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CLASSIFIERS_DIR = ROOT_DIR

SUBJECTIVITY_TRAINER = CLASSIFIERS_DIR / "subjectivity" / "SubjectivityClassifier.py"
POLARITY_TRAINER = CLASSIFIERS_DIR / "polarity" / "PolarityClassifier.py"


def run_command(command):
    print("\n$", " ".join(str(c) for c in command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train both subjectivity and polarity classifiers."
    )
    parser.add_argument(
        "--input-file",
        default=str(CLASSIFIERS_DIR / "data" / "eval.xls"),
        help="Path to the shared labelled input file (.xls/.xlsx tab text, or .csv)",
    )
    args = parser.parse_args()

    run_command([
        sys.executable,
        str(SUBJECTIVITY_TRAINER),
        "--input-file",
        args.input_file,
    ])

    run_command([
        sys.executable,
        str(POLARITY_TRAINER),
        "--input-csv",
        args.input_file,
    ])

    print("\nAll classifiers finished successfully.")
