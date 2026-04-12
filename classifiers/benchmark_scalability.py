import argparse
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from PolarityClassifier import PolarityClassifier
from SubjectivityClassifier import SubjectivityClassifier
from preprocessing import preprocess_text
from subjectivity_common import load_data as load_labeled_data
from subjectivity_common import prepare_subjectivity_data


CLASSIFIERS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = CLASSIFIERS_DIR.parent / "data" / "output" / "eval.xls"
DEFAULT_SUBJECTIVITY_MODEL = CLASSIFIERS_DIR / "models" / "subjectivity_model.joblib"
DEFAULT_POLARITY_MODEL = CLASSIFIERS_DIR / "models" / "polarity_model_full_hybrid.joblib"
DEFAULT_RESULTS_CSV = CLASSIFIERS_DIR / "scalability_results.csv"
DEFAULT_OUTPUT_PNG = CLASSIFIERS_DIR / "scalability_graph.png"
DEFAULT_MASTER_CORPUS_PATH = CLASSIFIERS_DIR.parent / "data" / "output" / "labeled_master_corpus_v2.json"


class ConstantPredictor:
    def __init__(self, label: str):
        self.label = label

    def predict(self, X):
        return [self.label] * len(X)


def parse_sizes(sizes_text: str) -> list[int]:
    values = []
    for raw in sizes_text.split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("At least one dataset size is required.")
    return sorted(set(values))


def benchmark_model(model, texts: pd.Series, model_name: str, sizes: list[int], repeats: int) -> list[dict]:
    records = []
    max_len = len(texts)

    for size in sizes:
        if size > max_len:
            continue

        run_times = []
        for _ in range(repeats):
            sample = texts.sample(n=size, replace=False, random_state=42).reset_index(drop=True)
            start = time.perf_counter()
            _ = model.predict(sample)
            elapsed = time.perf_counter() - start
            run_times.append(elapsed)

        avg_time = sum(run_times) / len(run_times)
        std_time = math.sqrt(sum((t - avg_time) ** 2 for t in run_times) / len(run_times))
        records_per_second = size / avg_time if avg_time > 0 else float("inf")

        records.append(
            {
                "model": model_name,
                "record_count": size,
                "avg_prediction_time_s": avg_time,
                "std_prediction_time_s": std_time,
                "records_per_second": records_per_second,
            }
        )

    return records


def load_corpus_texts(master_corpus_path: Path) -> pd.Series:
    if not master_corpus_path.exists():
        raise FileNotFoundError(f"Master corpus file not found: {master_corpus_path}")

    with master_corpus_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unsupported JSON format for master corpus")

    texts = []
    for row in records:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        split = str(row.get("dataset_split", "")).strip().lower()
        # Focus on the real inference target split when available.
        if split and split != "corpus":
            continue
        texts.append(text)

    if not texts:
        raise ValueError("No corpus texts found in master_corpus.json")

    return pd.Series(texts, name="text")


def load_labeled_from_master(master_corpus_path: Path) -> pd.DataFrame:
    if not master_corpus_path.exists():
        raise FileNotFoundError(f"Master corpus file not found: {master_corpus_path}")

    with master_corpus_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unsupported JSON format for master corpus")

    rows = []
    for row in records:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "")).strip()
        subjectivity = str(row.get("subjectivity", "")).strip().lower()
        polarity = str(row.get("polarity", "")).strip().lower()
        if not text:
            continue
        if subjectivity not in {"neutral", "opinionated"}:
            continue
        if polarity not in {"positive", "negative", "neutral"}:
            continue
        rows.append({"text": text, "subjectivity": subjectivity, "polarity": polarity})

    if not rows:
        raise ValueError("No labeled rows found in master corpus for fallback training.")

    df = pd.DataFrame(rows)
    df["clean_text"] = df["text"].apply(preprocess_text)
    return df


def benchmark_combined_inference(
    subjectivity_model,
    polarity_model,
    texts: pd.Series,
    sizes: list[int],
    repeats: int,
) -> list[dict]:
    rows = []
    max_len = len(texts)

    for size in sizes:
        if size > max_len:
            continue

        run_times = []
        opinionated_ratios = []
        for rep in range(repeats):
            sample = texts.sample(n=size, replace=False, random_state=42 + rep).reset_index(drop=True)
            sample_clean = sample.apply(preprocess_text)

            start = time.perf_counter()
            subj_pred = subjectivity_model.predict(sample_clean)
            subj_series = pd.Series(subj_pred).fillna("").astype(str).str.strip().str.lower()
            opinionated_mask = subj_series == "opinionated"

            if opinionated_mask.any():
                _ = polarity_model.predict(sample_clean[opinionated_mask.values])

            elapsed = time.perf_counter() - start
            run_times.append(elapsed)
            opinionated_ratios.append(float(opinionated_mask.mean()))

        avg_time = sum(run_times) / len(run_times)
        std_time = math.sqrt(sum((t - avg_time) ** 2 for t in run_times) / len(run_times))
        records_per_second = size / avg_time if avg_time > 0 else float("inf")

        rows.append(
            {
                "model": "combined_inference",
                "record_count": size,
                "avg_prediction_time_s": avg_time,
                "std_prediction_time_s": std_time,
                "records_per_second": records_per_second,
                "avg_opinionated_ratio": sum(opinionated_ratios) / len(opinionated_ratios),
            }
        )

    return rows


def load_models_with_fallback(
    subj_model_path: Path,
    pol_model_path: Path,
    labeled_input_path: Path,
    master_corpus_path: Path,
):
    try:
        subjectivity_model = joblib.load(subj_model_path)
        polarity_model = joblib.load(pol_model_path)
        print(f"Loaded subjectivity model from {subj_model_path}")
        print(f"Loaded polarity model from {pol_model_path}")
        return subjectivity_model, polarity_model
    except Exception as exc:
        print("Model load failed; falling back to in-environment retraining.")
        print(f"Reason: {exc}")

        if not labeled_input_path.exists():
            raise FileNotFoundError(
                f"Could not load models and fallback training input does not exist: {labeled_input_path}"
            )

        try:
            labeled_df = load_labeled_data(str(labeled_input_path))
        except Exception as eval_exc:
            print("Fallback training could not use eval input; trying labeled master corpus JSON.")
            print(f"Reason: {eval_exc}")
            labeled_df = load_labeled_from_master(master_corpus_path)
        subjectivity_df = prepare_subjectivity_data(labeled_df)
        polarity_df = labeled_df[
            (labeled_df["subjectivity"] == "opinionated")
            & (labeled_df["polarity"].isin(["positive", "negative", "neutral"]))
        ].copy()
        if polarity_df.empty:
            raise ValueError("Fallback training failed: no valid polarity rows found in labeled input.")

        subj_classes = subjectivity_df["subjectivity"].dropna().astype(str).str.strip().str.lower().unique()
        if len(subj_classes) < 2:
            subjectivity_model = ConstantPredictor(subj_classes[0])
            print(f"Using constant subjectivity predictor: {subj_classes[0]}")
        else:
            subjectivity_model = SubjectivityClassifier().fit(
                subjectivity_df["clean_text"], subjectivity_df["subjectivity"]
            )

        pol_classes = polarity_df["polarity"].dropna().astype(str).str.strip().str.lower().unique()
        if len(pol_classes) < 2:
            polarity_model = ConstantPredictor(pol_classes[0])
            print(f"Using constant polarity predictor: {pol_classes[0]}")
        else:
            polarity_model = PolarityClassifier().fit(
                polarity_df["clean_text"], polarity_df["polarity"]
            )
        return subjectivity_model, polarity_model


def assess_scalability(model_df: pd.DataFrame) -> str:
    if model_df.empty or len(model_df) < 2:
        return "Insufficient points to assess trend."

    first_rps = float(model_df.iloc[0]["records_per_second"])
    last_rps = float(model_df.iloc[-1]["records_per_second"])
    change_pct = ((last_rps - first_rps) / first_rps) * 100 if first_rps > 0 else 0.0

    if change_pct >= -10:
        return f"Near-linear scaling (records/sec change {change_pct:.2f}% from smallest to largest batch)."
    if change_pct >= -25:
        return f"Moderate scaling degradation (records/sec change {change_pct:.2f}%)."
    return f"Significant scaling degradation (records/sec change {change_pct:.2f}%)."


def build_plot(df: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for model_name, group in df.groupby("model"):
        g = group.sort_values("record_count")
        ax.plot(g["record_count"], g["records_per_second"], marker="o", label=model_name)

    ax.set_title("Records per Second vs Record Count")
    ax.set_xlabel("Record count")
    ax.set_ylabel("Records classified / second")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark prediction scalability and generate graph/report-ready outputs."
    )
    parser.add_argument(
        "--mode",
        choices=["separate", "combined"],
        default="combined",
        help="Benchmark separate model inference or full combined inference pipeline",
    )
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE), help="Path to labeled source data")
    parser.add_argument(
        "--master-corpus",
        default=str(DEFAULT_MASTER_CORPUS_PATH),
        help="Path to master_corpus.json for combined inference benchmarking",
    )
    parser.add_argument(
        "--sizes",
        default="500,1000,2000,5000,8000,10000",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of runs per size")
    parser.add_argument(
        "--subjectivity-model",
        default=str(DEFAULT_SUBJECTIVITY_MODEL),
        help="Path to subjectivity model",
    )
    parser.add_argument(
        "--polarity-model",
        default=str(DEFAULT_POLARITY_MODEL),
        help="Path to polarity model",
    )
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV), help="Output CSV path")
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG), help="Output graph image path")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    master_corpus_path = Path(args.master_corpus)
    subj_model_path = Path(args.subjectivity_model)
    pol_model_path = Path(args.polarity_model)
    results_csv = Path(args.results_csv)
    output_png = Path(args.output_png)

    sizes = parse_sizes(args.sizes)

    subjectivity_model, polarity_model = load_models_with_fallback(
        subj_model_path=subj_model_path,
        pol_model_path=pol_model_path,
        labeled_input_path=input_path,
        master_corpus_path=master_corpus_path,
    )

    if args.mode == "separate":
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        labeled_df = load_labeled_data(str(input_path))
        subjectivity_df = prepare_subjectivity_data(labeled_df)
        polarity_df = labeled_df[labeled_df["subjectivity"] == "opinionated"].copy()

        subjectivity_texts = subjectivity_df["clean_text"].reset_index(drop=True)
        if "clean_text" in polarity_df.columns:
            polarity_texts = polarity_df["clean_text"].reset_index(drop=True)
        else:
            polarity_texts = polarity_df["text"].apply(preprocess_text).reset_index(drop=True)

        all_rows = []
        all_rows.extend(benchmark_model(subjectivity_model, subjectivity_texts, "subjectivity", sizes, args.repeats))
        all_rows.extend(benchmark_model(polarity_model, polarity_texts, "polarity", sizes, args.repeats))
    else:
        corpus_texts = load_corpus_texts(master_corpus_path)
        all_rows = benchmark_combined_inference(
            subjectivity_model=subjectivity_model,
            polarity_model=polarity_model,
            texts=corpus_texts,
            sizes=sizes,
            repeats=args.repeats,
        )

    if not all_rows:
        raise ValueError("No benchmark rows were generated. Check dataset size and --sizes values.")

    results_df = pd.DataFrame(all_rows).sort_values(["model", "record_count"]).reset_index(drop=True)
    results_df.to_csv(results_csv, index=False)
    build_plot(results_df, output_png)

    print("\n=== Scalability Benchmark Results ===")
    for model_name, group in results_df.groupby("model"):
        print(f"\nModel: {model_name}")
        print(group[["record_count", "avg_prediction_time_s", "records_per_second"]].to_string(index=False))
        print("Assessment:", assess_scalability(group.sort_values("record_count")))

    print(f"\nSaved CSV: {results_csv}")
    print(f"Saved graph: {output_png}")


if __name__ == "__main__":
    main()
