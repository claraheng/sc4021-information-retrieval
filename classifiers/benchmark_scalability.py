import argparse
import math
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from polarity_common import load_data as load_polarity_source
from polarity_common import prepare_opinionated_data
from subjectivity_common import load_data as load_subjectivity_source
from subjectivity_common import prepare_subjectivity_data


CLASSIFIERS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = CLASSIFIERS_DIR / "eval.xls"
DEFAULT_SUBJECTIVITY_MODEL = CLASSIFIERS_DIR / "subjectivity_model.joblib"
DEFAULT_POLARITY_MODEL = CLASSIFIERS_DIR / "polarity_model.joblib"
DEFAULT_RESULTS_CSV = CLASSIFIERS_DIR / "scalability_results.csv"
DEFAULT_OUTPUT_PNG = CLASSIFIERS_DIR / "scalability_graph.png"


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
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE), help="Path to labeled source data")
    parser.add_argument(
        "--sizes",
        default="50,100,200,300,500,800",
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
    subj_model_path = Path(args.subjectivity_model)
    pol_model_path = Path(args.polarity_model)
    results_csv = Path(args.results_csv)
    output_png = Path(args.output_png)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not subj_model_path.exists():
        raise FileNotFoundError(f"Subjectivity model not found: {subj_model_path}")
    if not pol_model_path.exists():
        raise FileNotFoundError(f"Polarity model not found: {pol_model_path}")

    sizes = parse_sizes(args.sizes)

    subjectivity_df = prepare_subjectivity_data(load_subjectivity_source(str(input_path)))
    polarity_df = prepare_opinionated_data(load_polarity_source(str(input_path)))

    subjectivity_texts = subjectivity_df["clean_text"].reset_index(drop=True)
    polarity_texts = polarity_df["text"].reset_index(drop=True)

    subjectivity_model = joblib.load(subj_model_path)
    polarity_model = joblib.load(pol_model_path)

    all_rows = []
    all_rows.extend(benchmark_model(subjectivity_model, subjectivity_texts, "subjectivity", sizes, args.repeats))
    all_rows.extend(benchmark_model(polarity_model, polarity_texts, "polarity", sizes, args.repeats))

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
