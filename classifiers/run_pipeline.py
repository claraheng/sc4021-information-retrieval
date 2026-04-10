"""
run_pipeline.py — Master script to train classifiers and run predictions.

Data source: data/output/master_corpus.json
  - Records with dataset_split == "eval"   → labeled, used for training & evaluation
  - Records with dataset_split == "corpus" → unlabeled, used for bulk inference

Labels come from data/output/eval.xls (joined on id to the eval split).
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
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
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.utils import resample

from SubjectivityClassifier import SubjectivityClassifier
from PolarityClassifier import PolarityClassifier, LexiconScorer
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
    with open(MASTER_CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded master corpus: {len(df)} records")
    print(f"  Splits: {df['dataset_split'].value_counts().to_dict()}")
    return df


def load_eval_labels() -> pd.DataFrame:
    try:
        df = pd.read_excel(EVAL_LABELS_PATH, engine="openpyxl")
    except Exception:
        df = pd.read_excel(EVAL_LABELS_PATH, engine="xlrd")

    df.columns = df.columns.str.strip()

    required_cols = ["id", "subjectivity", "polarity"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"❌ eval.xls is missing columns: {missing}")

    df["text"]         = df["text"].fillna("").astype(str).str.strip()
    df["subjectivity"] = df["subjectivity"].fillna("").astype(str).str.strip().str.lower()
    df["polarity"]     = df["polarity"].fillna("").astype(str).str.strip().str.lower()
    return df[df["text"] != ""].copy()


def build_eval_dataframe() -> pd.DataFrame:
    corpus = load_master_corpus()
    labels = load_eval_labels()
    eval_texts = corpus[corpus["dataset_split"] == "eval"][["id", "text"]].copy()
    eval_texts["text"] = eval_texts["text"].fillna("").astype(str).str.strip()
    merged = eval_texts.merge(labels[["id", "subjectivity", "polarity"]], on="id", how="inner")
    print(f"  Eval records matched with labels: {len(merged)}")
    merged["clean_text"] = merged["text"].apply(preprocess_text)
    return merged


def build_corpus_dataframe(corpus_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if corpus_df is None:
        corpus_df = load_master_corpus()
    df = corpus_df[corpus_df["dataset_split"] == "corpus"].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    df["clean_text"] = df["text"].apply(preprocess_text)
    return df


# ── Class balancing ───────────────────────────────────────────────────────────

def balance_training_data(X_train: pd.Series, y_train: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Undersample the majority class (neutral) to 2× the size of the smallest
    minority class. Minority classes are left untouched.

    Keeps the eval set completely unmodified — only the training split is
    rebalanced, so evaluation metrics remain honest.

    Example — before: neutral=494, negative=129, positive=117
              after:  neutral=234, negative=129, positive=117
    """
    df = pd.concat([X_train, y_train], axis=1)
    df.columns = ["text", "label"]

    counts = df["label"].value_counts()
    minority_size = counts.min()
    majority_label = counts.idxmax()
    target_majority = min(counts[majority_label], 2 * minority_size)

    parts = []
    for label, group in df.groupby("label"):
        if label == majority_label and len(group) > target_majority:
            group = resample(group, n_samples=target_majority, random_state=42, replace=False)
        parts.append(group)

    balanced = pd.concat(parts).sample(frac=1, random_state=42)  # shuffle
    print(f"  [Undersampling] {counts.to_dict()} → {balanced['label'].value_counts().to_dict()}")
    return balanced["text"], balanced["label"]


# ── Threshold calibration ─────────────────────────────────────────────────────

def calibrate_thresholds(
    model: PolarityClassifier,
    X_train: pd.Series,
    y_train: pd.Series,
) -> dict[str, float]:
    """
    Find per-class probability thresholds that maximise macro F1 on the
    training set via 5-fold cross-validation.

    How it works:
      1. Get out-of-fold predicted probabilities on X_train (no data leakage).
      2. Grid-search threshold combinations for each class.
      3. Pick the combo with the highest macro F1.

    The returned dict maps class name → threshold. Use apply_thresholds()
    at prediction time instead of model.predict().

    Why cross-val instead of a held-out val set?
      With only ~700 training rows we can't afford to split further. OOF
      predictions give an unbiased estimate of calibration without losing data.
    """
    classes = list(model.classes_)
    print(f"\n  [Threshold calibration] Running 5-fold CV on {len(X_train)} training rows…")

    proba_oof = cross_val_predict(
        model.model, X_train, y_train, cv=5, method="predict_proba"
    )

    grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_f1, best_thresholds = 0.0, {c: 0.5 for c in classes}

    for combo in product(grid, repeat=len(classes)):
        thresh_vec = np.array(combo)
        # Subtract threshold from each class probability, then argmax
        adjusted = proba_oof - thresh_vec
        preds = np.array(classes)[np.argmax(adjusted, axis=1)]
        score = f1_score(y_train, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresholds = dict(zip(classes, combo))

    print(f"  [Threshold calibration] Best thresholds: {best_thresholds}  (CV macro F1: {best_f1:.4f})")
    return best_thresholds


def apply_thresholds(
    model: PolarityClassifier,
    X: pd.Series,
    thresholds: dict[str, float],
) -> np.ndarray:
    """
    Predict class labels using per-class probability thresholds instead of
    the default argmax (which implicitly uses 0.5 for all classes).
    """
    classes = list(model.classes_)
    proba   = model.predict_proba(X)
    thresh_vec = np.array([thresholds[c] for c in classes])
    return np.array(classes)[np.argmax(proba - thresh_vec, axis=1)]


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_auc(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    if len(classes) == 2:
        pos_idx = 1
        y_binary = (y_test == classes[1]).astype(int)
        return roc_auc_score(y_binary, y_proba[:, pos_idx])
    return roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")


def evaluate_model(model_name, y_test, y_pred, auc_score, pred_time):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    rps  = len(y_test) / pred_time if pred_time > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Precision:       {prec:.4f}  (weighted)")
    print(f"  Recall:          {rec:.4f}  (weighted)")
    print(f"  F1 Score:        {f1_w:.4f}  (weighted)")
    print(f"  Macro F1:        {f1_m:.4f}  ← primary metric")
    print(f"  ROC-AUC:         {auc_score:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Scalability:")
    print(f"    Records evaluated:  {len(y_test)}")
    print(f"    Inference time:     {pred_time:.4f}s")
    print(f"    Records/second:     {rps:.2f}")


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    model_output_path: Path,
    use_ensemble: bool = False,
    use_lexicon: bool = False,
    use_undersampling: bool = False,
    use_calibration: bool = False,
) -> dict:
    """
    Train a PolarityClassifier with any combination of innovations, evaluate
    on the held-out test set, and return a metrics dict for ablation_study().

    Parameters
    ----------
    use_undersampling : downsample neutral class before fitting
    use_calibration   : tune per-class thresholds via CV on training data
    """
    # Optionally rebalance training data before fitting
    X_tr, y_tr = (balance_training_data(X_train, y_train)
                  if use_undersampling else (X_train, y_train))

    model = PolarityClassifier(use_ensemble=use_ensemble, use_lexicon=use_lexicon)
    model.fit(X_tr, y_tr)
    model.save(model_output_path)

    # Optionally calibrate thresholds on the (possibly rebalanced) training set
    thresholds = None
    if use_calibration:
        thresholds = calibrate_thresholds(model, X_tr, y_tr)

    # Evaluate
    t0 = time.time()
    y_pred = (apply_thresholds(model, X_test, thresholds)
              if thresholds else model.predict(X_test))
    pred_time = time.time() - t0

    try:
        auc = compute_auc(model, X_test, y_test)
    except Exception as e:
        print(f"⚠️  AUC computation failed: {e}")
        auc = 0.0

    evaluate_model(model_name, y_test, y_pred, auc, pred_time)

    return {
        "name":       model_name,
        "accuracy":   accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro":   f1_score(y_test, y_pred, average="macro", zero_division=0),
        "auc":        auc,
        "pred_time":  pred_time,
        "model":      model,
        "thresholds": thresholds,
    }


def ablation_study(m1: dict, m2: dict, output_path: Path):
    """Print a side-by-side comparison and save to CSV."""
    def boost(a, b): return f"+{(b - a) * 100:.2f}%"

    print("\n" + "=" * 70)
    print("🚀 ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Metric':<22} | {m1['name']:<20} | {m2['name']:<20} | Boost")
    print("-" * 70)
    print(f"{'Accuracy':<22} | {m1['accuracy']:<20.4f} | {m2['accuracy']:<20.4f} | {boost(m1['accuracy'], m2['accuracy'])}")
    print(f"{'F1 (weighted)':<22} | {m1['f1_weighted']:<20.4f} | {m2['f1_weighted']:<20.4f} | {boost(m1['f1_weighted'], m2['f1_weighted'])}")
    print(f"{'F1 (macro) ★':<22} | {m1['f1_macro']:<20.4f} | {m2['f1_macro']:<20.4f} | {boost(m1['f1_macro'], m2['f1_macro'])}")
    print(f"{'ROC-AUC':<22} | {m1['auc']:<20.4f} | {m2['auc']:<20.4f} | {boost(m1['auc'], m2['auc'])}")
    print("=" * 70)

    pd.DataFrame([
        {"Model": m["name"], "Accuracy": m["accuracy"],
         "F1_Weighted": m["f1_weighted"], "F1_Macro": m["f1_macro"],
         "ROC_AUC": m["auc"], "Inference_Time_s": m["pred_time"]}
        for m in (m1, m2)
    ]).to_csv(output_path, index=False)
    print(f"📊 Saved → {output_path}")


def train_subjectivity(df_eval: pd.DataFrame):
    valid = df_eval[df_eval["subjectivity"].isin(["neutral", "opinionated"])].copy()
    if valid.empty:
        raise ValueError("No rows with valid subjectivity labels.")

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

    t0 = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - t0
    auc = compute_auc(clf, X_test, y_test)
    evaluate_model("Subjectivity", y_test, y_pred, auc, pred_time)
    return clf


def train_polarity(df_eval: pd.DataFrame):
    """
    Train all polarity model variants and run ablation studies.

    Ablation ladder (each adds one innovation on top of base):
        1. Base            — TF-IDF + LR
        2. + Ensemble      — swap LR for soft-voting (LR+RF+NB)
        3. + Lexicon       — add EV domain lexicon features
        4. + Undersample   — balance neutral class at training time
        5. + Calibration   — tune per-class probability thresholds
        6. Full Hybrid     — undersampling + calibration together (best model)
    """
    opinionated = df_eval[df_eval["subjectivity"] == "opinionated"].copy()
    opinionated = opinionated[opinionated["polarity"].isin(["positive", "negative", "neutral"])].copy()
    if opinionated.empty:
        raise ValueError("No opinionated rows with valid polarity labels.")

    X = opinionated["clean_text"]
    y = opinionated["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[Polarity] Training on {len(X_train)} rows, testing on {len(X_test)} rows")
    print(f"  Label distribution (train): {y_train.value_counts().to_dict()}")

    # Lexicon diagnostic
    scorer = LexiconScorer()
    scores = scorer.transform(X_train)
    print(f"  Lexicon hit rate: {(scores[:, 1] > 0).mean():.1%}")

    # ── 1. Base ───────────────────────────────────────────────────────────
    print("\n--- [1/6] BASE ---")
    base = train_model(
        "Base (TF-IDF + LR)",
        X_train, y_train, X_test, y_test,
        POLARITY_MODEL_PATH,
    )

    # ── 2. + Ensemble ─────────────────────────────────────────────────────
    print("\n--- [2/6] + ENSEMBLE ---")
    ensemble = train_model(
        "+ Ensemble",
        X_train, y_train, X_test, y_test,
        MODELS_DIR / "polarity_model_ensemble.joblib",
        use_ensemble=True,
    )

    # ── 3. + Lexicon ──────────────────────────────────────────────────────
    print("\n--- [3/6] + LEXICON ---")
    lexicon = train_model(
        "+ Lexicon",
        X_train, y_train, X_test, y_test,
        MODELS_DIR / "polarity_model_lexicon.joblib",
        use_lexicon=True,
    )

    # ── 4. + Undersample ─────────────────────────────────────────────────
    print("\n--- [4/6] + UNDERSAMPLE ---")
    undersample = train_model(
        "+ Undersample",
        X_train, y_train, X_test, y_test,
        MODELS_DIR / "polarity_model_undersample.joblib",
        use_undersampling=True,
    )

    # ── 5. + Calibration ─────────────────────────────────────────────────
    print("\n--- [5/6] + CALIBRATION ---")
    calibration = train_model(
        "+ Calibration",
        X_train, y_train, X_test, y_test,
        MODELS_DIR / "polarity_model_calibration.joblib",
        use_calibration=True,
    )

    # ── 6. Full Hybrid (undersample + calibration) ────────────────────────
    print("\n--- [6/6] FULL HYBRID (Undersample + Calibration) ---")
    full_hybrid = train_model(
        "Full Hybrid",
        X_train, y_train, X_test, y_test,
        MODELS_DIR / "polarity_model_full_hybrid.joblib",
        use_undersampling=True,
        use_calibration=True,
    )

    # ── Ablation studies (each vs base) ───────────────────────────────────
    ablation_study(base, ensemble,    DATA_DIR / "ablation_ensemble.csv")
    ablation_study(base, lexicon,     DATA_DIR / "ablation_lexicon.csv")
    ablation_study(base, undersample, DATA_DIR / "ablation_undersample.csv")
    ablation_study(base, calibration, DATA_DIR / "ablation_calibration.csv")
    ablation_study(base, full_hybrid, DATA_DIR / "ablation_full_hybrid.csv")

    # ── Grand summary table ───────────────────────────────────────────────
    all_models = [base, ensemble, lexicon, undersample, calibration, full_hybrid]
    print("\n" + "=" * 80)
    print("📋 GRAND SUMMARY")
    print("=" * 80)
    print(f"{'Model':<35} | {'Macro F1':>8} | {'W-F1':>8} | {'AUC':>7} | {'Acc':>6}")
    print("-" * 80)
    for m in all_models:
        marker = " ★" if m["f1_macro"] == max(x["f1_macro"] for x in all_models) else ""
        print(f"  {m['name']:<33} | {m['f1_macro']:>8.4f} | {m['f1_weighted']:>8.4f} | {m['auc']:>7.4f} | {m['accuracy']:>6.4f}{marker}")
    print("=" * 80)

    # Save grand summary
    pd.DataFrame([
        {"Model": m["name"], "Macro_F1": m["f1_macro"], "Weighted_F1": m["f1_weighted"],
         "ROC_AUC": m["auc"], "Accuracy": m["accuracy"], "Inference_Time_s": m["pred_time"]}
        for m in all_models
    ]).to_csv(DATA_DIR / "ablation_grand_summary.csv", index=False)
    print(f"📊 Grand summary saved → {DATA_DIR / 'ablation_grand_summary.csv'}")

    # Return full hybrid as the production model for corpus inference
    return full_hybrid["model"], full_hybrid.get("thresholds")


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_corpus(
    subj_model,
    pol_model,
    pol_thresholds: dict | None,
    df_corpus: pd.DataFrame,
) -> pd.DataFrame:
    print(f"\n{'=' * 60}")
    print(f"  Running Inference on {len(df_corpus)} corpus records")
    print(f"{'=' * 60}")

    t0 = time.time()

    subj_preds = subj_model.predict(df_corpus["clean_text"])
    df_corpus["subjectivity"] = subj_preds

    df_corpus["polarity"] = ""
    mask = df_corpus["subjectivity"] == "opinionated"
    if mask.any():
        X_corpus = df_corpus.loc[mask, "clean_text"]
        pol_preds = (
            apply_thresholds(pol_model, X_corpus, pol_thresholds)
            if pol_thresholds else pol_model.predict(X_corpus)
        )
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
    parser.add_argument("--retrain", action="store_true",
                        help="Force retraining even if model files already exist.")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip bulk inference on the corpus split.")
    parser.add_argument("--output", default=str(PREDICTIONS_OUTPUT_PATH),
                        help="Path to save corpus predictions CSV.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_eval = build_eval_dataframe()

    if args.retrain or not SUBJECTIVITY_MODEL_PATH.exists():
        subj_model = train_subjectivity(df_eval)
    else:
        print(f"\nLoading subjectivity model from {SUBJECTIVITY_MODEL_PATH}")
        subj_model = SubjectivityClassifier.load(SUBJECTIVITY_MODEL_PATH)

    pol_thresholds = None
    if args.retrain or not POLARITY_MODEL_PATH.exists():
        pol_model, pol_thresholds = train_polarity(df_eval)
    else:
        print(f"\nLoading polarity model from {POLARITY_MODEL_PATH}")
        pol_model = PolarityClassifier.load(POLARITY_MODEL_PATH)

    if not args.skip_predict:
        df_corpus = build_corpus_dataframe()
        df_results = predict_corpus(subj_model, pol_model, pol_thresholds, df_corpus)
        out = Path(args.output)
        df_results[["id", "text", "platform", "source_target", "subjectivity", "polarity"]].to_csv(
            out, index=False
        )
        print(f"\n  Predictions saved → {out}")


if __name__ == "__main__":
    main()
