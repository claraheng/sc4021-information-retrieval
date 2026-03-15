#!/usr/bin/env python3
"""Vertically merge (concatenate) CSV files.

Usage examples:
  python3 merge_csv_vertical.py files -o files/merged.csv
  python3 merge_csv_vertical.py file1.csv file2.csv -o merged.csv

Behavior:
- Uses union of all columns across files.
- Preserves row order by input file order.
- Fills missing columns with empty values.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

COMMENTS_FILE_RE = re.compile(r"^(\d+)_comments\.csv$")


def collect_csv_files(inputs: list[str]) -> tuple[list[Path], int]:
    paths: list[Path] = []
    quantity_sum = 0

    if len(inputs) == 1:
        p = Path(inputs[0])
        if p.is_dir():
            for csv_file in sorted(p.glob("*_comments.csv")):
                match = COMMENTS_FILE_RE.match(csv_file.name)
                if not match:
                    continue
                paths.append(csv_file)
                quantity_sum += int(match.group(1))
        elif p.is_file():
            match = COMMENTS_FILE_RE.match(p.name)
            if not match:
                raise FileNotFoundError(
                    f"Input file does not match [Quantity]_comments.csv: {p}"
                )
            paths = [p]
            quantity_sum = int(match.group(1))
        else:
            raise FileNotFoundError(f"Input not found: {p}")
    else:
        for item in inputs:
            p = Path(item)
            match = COMMENTS_FILE_RE.match(p.name)
            if not p.is_file() or not match:
                raise FileNotFoundError(
                    f"Input not found or not [Quantity]_comments.csv: {p}"
                )
            paths.append(p)
            quantity_sum += int(match.group(1))

    if not paths:
        raise FileNotFoundError("No [Quantity]_comments.csv files found.")

    return paths, quantity_sum


def merge_csv_vertical(input_files: list[Path], output_file: Path) -> tuple[int, int]:
    all_rows: list[dict[str, str]] = []
    all_columns: list[str] = []
    seen_columns: set[str] = set()

    for csv_file in input_files:
        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue

            for col in reader.fieldnames:
                if col not in seen_columns:
                    seen_columns.add(col)
                    all_columns.append(col)

            for row in reader:
                all_rows.append(dict(row))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    return len(input_files), len(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertically merge CSV files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files named [Quantity]_comments.csv, or one directory containing them",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output merged CSV file path (default: <sum>_comments.csv in input directory)",
    )
    args = parser.parse_args()

    input_files, quantity_sum = collect_csv_files(args.inputs)
    if args.output:
        output_file = Path(args.output)
    else:
        if len(args.inputs) == 1 and Path(args.inputs[0]).is_dir():
            output_dir = Path(args.inputs[0])
        else:
            output_dir = input_files[0].parent
        output_file = output_dir / f"{quantity_sum}_comments.csv"

    file_count, row_count = merge_csv_vertical(input_files, output_file)
    print(f"Merged {file_count} file(s), total rows: {row_count}")
    print(f"Quantity sum: {quantity_sum}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
