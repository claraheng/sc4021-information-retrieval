"""
run_pipeline.py — Master script to train classifiers and run predictions.

Data source: data/output/master_corpus.json
  - Records with dataset_split == "eval"   → labeled, used for training & evaluation
  - Records with dataset_split == "corpus" → unlabeled, used for bulk inference

Labels come from data/output/eval.xls (joined on text to the eval split).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from SubjectivityClassifier import SubjectivityClassifier
from PolarityClassifier import PolarityClassifier
from preprocessing import preprocess_text

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "output"
MASTER_CORPUS_PATH = DATA_DIR / "master_corpus.json"
EVAL_LABELS_PATH = DATA_DIR / "eval.xls"
MODELS_DIR = Path(__file__).resolve().parent / "models"
SUBJECTIVITY_MODEL_PATH = MODELS_DIR / "subjectivity_model.joblib"
POLARITY_MODEL_PATH = MODELS_DIR / "polarity_model.joblib"
PREDICTIONS_OUTPUT_PATH = DATA_DIR / "predictions.csv"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_master_corpus() -> pd.DataFrame:
    """Load master_corpus.json and return a DataFrame."""
    with open(MASTER_CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded master corpus: {len(df)} records")
    print(f"  Splits: {df['dataset_split'].value_counts().to_dict()}")
    return df


def load_eval_labels() -> pd.DataFrame:
    """Load the manually labeled eval.xls and normalise columns."""
    
    # Use openpyxl because Google Sheets downloads as .xlsx
    try:
        df = pd.read_excel(EVAL_LABELS_PATH, engine="openpyxl")
    except Exception:
        # Fallback just in case you saved it as a strict 97-2003 .xls
        df = pd.read_excel(EVAL_LABELS_PATH, engine="xlrd")

    # CRITICAL: Strip hidden spaces from column headers (e.g. "id " -> "id")
    df.columns = df.columns.str.strip()

    # Safety check to ensure you added the final columns in Google Sheets
    required_cols = ["id", "subjectivity", "polarity"]
    missing =[col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Your eval.xls is missing these columns: {missing}. Add them to Google Sheets and re-download!")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["subjectivity"] = df["subjectivity"].fillna("").astype(str).str.strip().str.lower()
    df["polarity"] = df["polarity"].fillna("").astype(str).str.strip().str.lower()
    
    df = df[df["text"] != ""].copy()
    return df


def build_eval_dataframe() -> pd.DataFrame:
    """
    Merge eval-split texts from master_corpus.json with labels from eval.xls.
    Also preprocess the text column.
    """
    corpus = load_master_corpus()
    labels = load_eval_labels()

    eval_texts = corpus[corpus["dataset_split"] == "eval"][["id", "text"]].copy()
    eval_texts["text"] = eval_texts["text"].fillna("").astype(str).str.strip()

    # Merge labels onto eval texts
    # Join on the immutable ID instead of the fragile text
    merged = eval_texts.merge(labels[["id", "subjectivity", "polarity"]], on="id", how="inner")
    print(f"  Eval records matched with labels: {len(merged)}")

    # Preprocess
    merged["clean_text"] = merged["text"].apply(preprocess_text)
    return merged


def build_corpus_dataframe(corpus_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the unlabeled corpus split, preprocessed."""
    if corpus_df is None:
        corpus_df = load_master_corpus()
    df = corpus_df[corpus_df["dataset_split"] == "corpus"].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    df["clean_text"] = df["text"].apply(preprocess_text)
    return df


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_auc(model, X_test, y_test):
    """Compute ROC-AUC, handling binary and multiclass cases."""
    y_proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    if len(classes) == 2:
        pos_label = classes[1]
        pos_idx = classes.index(pos_label)
        y_binary = (y_test == pos_label).astype(int)
        return roc_auc_score(y_binary, y_proba[:, pos_idx])
    return roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")


def evaluate_model(model_name, y_test, y_pred, auc_score, pred_time):
    """Print a full metrics report for a trained model."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    num_records = len(y_test)
    rps = num_records / pred_time if pred_time > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}  (weighted)")
    print(f"  Recall:     {rec:.4f}  (weighted)")
    print(f"  F1 Score:   {f1:.4f}  (weighted)")
    print(f"  ROC-AUC:    {auc_score:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Scalability:")
    print(f"    Records evaluated:  {num_records}")
    print(f"    Inference time:     {pred_time:.4f}s")
    print(f"    Records/second:     {rps:.2f}")


# ── Training ─────────────────────────────────────────────────────────────────

def train_subjectivity(df_eval: pd.DataFrame):
    """Train and evaluate the subjectivity model."""
    valid = df_eval[df_eval["subjectivity"].isin(["neutral", "opinionated"])].copy()
    if valid.empty:
        raise ValueError("No rows with valid subjectivity labels (neutral / opinionated).")

    X = valid["clean_text"]
    y = valid["subjectivity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n[Subjectivity] Training on {len(X_train)} rows, testing on {len(X_test)} rows")
    print(f"  Label distribution (train): {y_train.value_counts().to_dict()}")

    clf = SubjectivityClassifier()
    clf.fit(X_train, y_train)
    clf.save(SUBJECTIVITY_MODEL_PATH)
    print(f"  Model saved → {SUBJECTIVITY_MODEL_PATH}")

    t0 = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - t0

    auc = compute_auc(clf, X_test, y_test)
    evaluate_model("Subjectivity", y_test, y_pred, auc, pred_time)
    return clf


def train_polarity(df_eval: pd.DataFrame):
    """Train and evaluate the polarity model (opinionated rows only)."""
    opinionated = df_eval[df_eval["subjectivity"] == "opinionated"].copy()
    valid_labels = ["positive", "negative", "neutral"]
    opinionated = opinionated[opinionated["polarity"].isin(valid_labels)].copy()

    if opinionated.empty:
        raise ValueError("No opinionated rows with valid polarity labels.")

    X = opinionated["clean_text"]
    y = opinionated["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[Polarity] Training on {len(X_train)} rows, testing on {len(X_test)} rows")
    print(f"  Label distribution (train): {y_train.value_counts().to_dict()}")

    clf = PolarityClassifier()
    clf.fit(X_train, y_train)
    clf.save(POLARITY_MODEL_PATH)
    print(f"  Model saved → {POLARITY_MODEL_PATH}")

    t0 = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - t0

    auc = compute_auc(clf, X_test, y_test)
    evaluate_model("Polarity", y_test, y_pred, auc, pred_time)
    return clf


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_corpus(subj_model, pol_model, df_corpus: pd.DataFrame) -> pd.DataFrame:
    """Run both models on the unlabeled corpus split."""
    print(f"\n{'=' * 60}")
    print(f"  Running Inference on {len(df_corpus)} corpus records")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Subjectivity
    subj_preds = subj_model.predict(df_corpus["clean_text"])
    df_corpus["subjectivity"] = subj_preds

    # Polarity (only for opinionated)
    df_corpus["polarity"] = ""
    mask = df_corpus["subjectivity"] == "opinionated"
    if mask.any():
        pol_preds = pol_model.predict(df_corpus.loc[mask, "clean_text"])
        df_corpus.loc[mask, "polarity"] = pol_preds

    elapsed = time.time() - t0
    rps = len(df_corpus) / elapsed if elapsed > 0 else 0

    print(f"  Subjectivity distribution: {df_corpus['subjectivity'].value_counts().to_dict()}")
    print(f"  Polarity distribution:     {df_corpus[df_corpus['polarity'] != '']['polarity'].value_counts().to_dict()}")
    print(f"\n  Scalability:")
    print(f"    Total records:      {len(df_corpus)}")
    print(f"    Inference time:     {elapsed:.4f}s")
    print(f"    Records/second:     {rps:.2f}")

    return df_corpus


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train subjectivity & polarity classifiers and predict on the corpus."
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retraining even if model files already exist.",
    )
    parser.add_argument(
        "--skip-predict", action="store_true",
        help="Skip bulk inference on the corpus split.",
    )
    parser.add_argument(
        "--output", default=str(PREDICTIONS_OUTPUT_PATH),
        help="Path to save corpus predictions CSV.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Build labeled eval dataframe ──────────────────────────────────
    df_eval = build_eval_dataframe()

    # ── 2. Subjectivity ─────────────────────────────────────────────────
    if args.retrain or not SUBJECTIVITY_MODEL_PATH.exists():
        subj_model = train_subjectivity(df_eval)
    else:
        print(f"\nLoading existing subjectivity model from {SUBJECTIVITY_MODEL_PATH}")
        subj_model = SubjectivityClassifier.load(SUBJECTIVITY_MODEL_PATH)

    # ── 3. Polarity ──────────────────────────────────────────────────────
    if args.retrain or not POLARITY_MODEL_PATH.exists():
        pol_model = train_polarity(df_eval)
    else:
        print(f"\nLoading existing polarity model from {POLARITY_MODEL_PATH}")
        pol_model = PolarityClassifier.load(POLARITY_MODEL_PATH)

    # ── 4. Predict on corpus ─────────────────────────────────────────────
    if not args.skip_predict:
        df_corpus = build_corpus_dataframe()
        df_results = predict_corpus(subj_model, pol_model, df_corpus)

        out = Path(args.output)
        df_results[["id", "text", "platform", "source_target", "subjectivity", "polarity"]].to_csv(
            out, index=False
        )
        print(f"\n  Predictions saved → {out}")


if __name__ == "__main__":
    main()
