import argparse
import subprocess
import sys
from pathlib import Path
import re
import time
from datetime import datetime


ROOT_DIR = Path(__file__).resolve().parent
CLASSIFIERS_DIR = ROOT_DIR

SUBJECTIVITY_TRAINER = CLASSIFIERS_DIR / "SubjectivityClassifier.py"
POLARITY_TRAINER = CLASSIFIERS_DIR / "PolarityClassifier.py"
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


def _extract_first_float(pattern: str, text: str):
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_first_int(pattern: str, text: str):
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_block_between(text: str, start_marker: str, end_marker: str):
    start = text.find(start_marker)
    if start == -1:
        return ""

    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


def _append_training_report(input_file: str, subj_stdout: str, pol_stdout: str, subj_elapsed: float, pol_elapsed: float):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subj_accuracy = _extract_first_float(r"Accuracy:\s*([0-9]*\.?[0-9]+)", subj_stdout)
    subj_auc = _extract_first_float(r"ROC-AUC:\s*([0-9]*\.?[0-9]+)", subj_stdout)
    subj_rps = _extract_first_float(r"Records classified per second:\s*([0-9]*\.?[0-9]+)", subj_stdout)
    subj_eval_size = _extract_first_int(r"Total records classified:\s*(\d+)", subj_stdout)
    subj_report = _extract_block_between(subj_stdout, "Classification Report:", "Confusion Matrix:")

    pol_accuracy = _extract_first_float(r"Accuracy:\s*([0-9]*\.?[0-9]+)", pol_stdout)
    pol_auc = _extract_first_float(r"ROC-AUC:\s*([0-9]*\.?[0-9]+)", pol_stdout)
    pol_rps = _extract_first_float(r"Records classified per second:\s*([0-9]*\.?[0-9]+)", pol_stdout)
    pol_eval_size = _extract_first_int(r"Total records classified:\s*(\d+)", pol_stdout)
    pol_report = _extract_block_between(pol_stdout, "Classification Report:", "Confusion Matrix:")

    discussion_lines = []
    if subj_accuracy is not None:
        discussion_lines.append(
            f"- Subjectivity holdout accuracy is {subj_accuracy:.4f}; this reflects generalization to unseen rows from the same dataset distribution."
        )
    if pol_accuracy is not None:
        discussion_lines.append(
            f"- Polarity holdout accuracy is {pol_accuracy:.4f}; polarity is typically harder because class boundaries are less distinct and labels are more imbalanced."
        )
    if subj_auc is not None and pol_auc is not None:
        discussion_lines.append(
            f"- ROC-AUC is {subj_auc:.4f} for subjectivity and {pol_auc:.4f} for polarity, indicating ranking quality beyond a single decision threshold."
        )
    if subj_rps is not None and pol_rps is not None:
        discussion_lines.append(
            f"- Throughput indicates good short-run scalability on this machine ({subj_rps:.2f} rec/s for subjectivity, {pol_rps:.2f} rec/s for polarity)."
        )
    discussion_lines.append(
        "- Precision/recall/F1 should be interpreted per class, especially when minority classes have lower support."
    )

    lines = [
        "",
        "## RunClassifiers",
        f"- Timestamp: {now}",
        f"- Input labelled dataset: {input_file}",
        "",
        "### Evaluation Summary",
        "- Method: Random holdout evaluation using stratified train/test splits.",
        "- Subjectivity split: 70% train / 30% test (random_state=42, stratified).",
        "- Polarity split: 80% train / 20% test (random_state=42, stratified).",
        "- This holdout test is your random accuracy test on the rest of the data: performance is measured on unseen test rows, not training rows.",
        "",
        "### Subjectivity Model Metrics",
        f"- Accuracy: {subj_accuracy if subj_accuracy is not None else 'N/A'}",
        f"- ROC-AUC: {subj_auc if subj_auc is not None else 'N/A'}",
        f"- Evaluated records (test set): {subj_eval_size if subj_eval_size is not None else 'N/A'}",
        f"- Records classified per second: {subj_rps if subj_rps is not None else 'N/A'}",
        f"- End-to-end trainer runtime: {subj_elapsed:.4f} s",
        "",
        "Precision/Recall/F1 report:",
        "```text",
        subj_report if subj_report else "(classification report not found in stdout)",
        "```",
        "",
        "### Polarity Model Metrics",
        f"- Accuracy: {pol_accuracy if pol_accuracy is not None else 'N/A'}",
        f"- ROC-AUC: {pol_auc if pol_auc is not None else 'N/A'}",
        f"- Evaluated records (test set): {pol_eval_size if pol_eval_size is not None else 'N/A'}",
        f"- Records classified per second: {pol_rps if pol_rps is not None else 'N/A'}",
        f"- End-to-end trainer runtime: {pol_elapsed:.4f} s",
        "",
        "Precision/Recall/F1 report:",
        "```text",
        pol_report if pol_report else "(classification report not found in stdout)",
        "```",
        "",
        "### Scalability Notes",
        "- Throughput is reported as records classified per second from the held-out test set.",
        "- As dataset size grows, expected prediction time scales approximately linearly with record count for fixed model and hardware.",
        "- End-to-end runtime also includes data loading and preprocessing overhead, not only model inference.",
        "",
        "### Discussion",
        *discussion_lines,
    ]

    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nUpdated report: {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train both subjectivity and polarity classifiers."
    )
    parser.add_argument(
        "--input-file",
        default=str(CLASSIFIERS_DIR / "eval.xls"),
        help="Path to the shared labelled input file (.xls/.xlsx tab text, or .csv)",
    )
    args = parser.parse_args()

    subj_result, subj_elapsed = run_command([
        sys.executable,
        str(SUBJECTIVITY_TRAINER),
        "--input-file",
        args.input_file,
    ])

    pol_result, pol_elapsed = run_command([
        sys.executable,
        str(POLARITY_TRAINER),
        "--input-csv",
        args.input_file,
    ])

    _append_training_report(
        input_file=args.input_file,
        subj_stdout=subj_result.stdout or "",
        pol_stdout=pol_result.stdout or "",
        subj_elapsed=subj_elapsed,
        pol_elapsed=pol_elapsed,
    )

    print("\nAll classifiers finished successfully.")
