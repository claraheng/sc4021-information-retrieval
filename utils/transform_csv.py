#!/usr/bin/env python3
"""Transform comment CSV files into a normalized opinion dataset.

Output columns:
- Text
- Sentiment (positive/negative/neutral)
- Concept
- Subjective/objective
- Country

Assumes input CSVs follow the same general schema as files in /files
(e.g., contain a text column like "Comment Text").
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

RE_WORD = re.compile(r"[a-zA-Z']+")

TEXT_COLUMN_CANDIDATES = [
    "Comment Text",
    "text",
    "Text",
    "comment",
    "Comment",
    "content",
    "Content",
]

CONCEPT_RULES = {
    "Cost": [
        "cost",
        "price",
        "expensive",
        "cheap",
        "saving",
        "savings",
        "insurance",
        "afford",
        "petrol",
        "diesel",
        "fuel",
        "kwh",
        "tariff",
    ],
    "Charging": [
        "charge",
        "charging",
        "charger",
        "battery",
        "range",
        "fast charge",
        "home charging",
        "plug",
    ],
    "Performance": [
        "power",
        "fast",
        "acceleration",
        "drive",
        "driving",
        "smooth",
        "quiet",
        "handling",
    ],
    "Infrastructure": [
        "station",
        "infrastructure",
        "grid",
        "public charging",
        "network",
    ],
    "Reliability": [
        "reliable",
        "reliability",
        "break",
        "broken",
        "issue",
        "problem",
        "fault",
    ],
    "Environment": [
        "environment",
        "climate",
        "emission",
        "green",
        "pollution",
        "sustainable",
        "co2",
    ],
    "Policy": [
        "government",
        "policy",
        "subsidy",
        "tax",
        "ban",
        "regulation",
    ],
    "Brand": [
        "tesla",
        "byd",
        "vw",
        "hyundai",
        "kia",
        "bmw",
        "mercedes",
        "toyota",
        "ford",
        "nissan",
    ],
}

POSITIVE_WORDS = {
    "love",
    "great",
    "good",
    "excellent",
    "amazing",
    "better",
    "best",
    "quiet",
    "smooth",
    "save",
    "saving",
    "savings",
    "recommend",
    "happy",
}

NEGATIVE_WORDS = {
    "bad",
    "worse",
    "worst",
    "hate",
    "awful",
    "terrible",
    "problem",
    "issues",
    "issue",
    "broken",
    "expensive",
    "annoying",
    "slow",
    "poor",
}

EV_CONTEXT_MARKERS = {
    "ev",
    "evs",
    "electric",
    "electric vehicle",
    "electric vehicles",
    "battery",
    "charging",
    "charger",
    "range",
    "ice",
    "petrol",
    "diesel",
    "hybrid",
}

EV_POSITIVE_PHRASES = [
    "love my ev",
    "never going back",
    "went with the ev",
    "home charging",
    "cheap to run",
    "saving money",
    "saves money",
    "lower running cost",
]

EV_NEGATIVE_PHRASES = [
    "range anxiety",
    "too expensive",
    "charging is slow",
    "no charger",
    "not enough chargers",
    "battery issue",
    "battery problem",
    "would never buy",
    "not practical",
]

SUBJECTIVE_CUES = {
    "i",
    "my",
    "me",
    "we",
    "our",
    "feel",
    "think",
    "believe",
    "prefer",
    "love",
    "hate",
}

COUNTRY_RULES = {
    "United Kingdom": ["uk", "britain", "england", "scotland", "wales", "quid"],
    "United States": ["usa", "us", "america", "dollar"],
    "Canada": ["canada", "cad"],
    "Australia": ["australia", "aud"],
    "Singapore": ["singapore", "sg", "sgd"],
    "India": ["india", "inr", "rupee"],
    "Malaysia": ["malaysia", "myr", "ringgit"],
    "Germany": ["germany", "deutschland"],
    "France": ["france"],
}

SYMBOL_COUNTRY_RULES = {
    "United Kingdom": ["\u00a3"],
    "United States": ["$"],
}


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in RE_WORD.findall(text)]


def infer_concept(text: str) -> str:
    t = text.lower()
    scores: dict[str, int] = {}
    for concept, keywords in CONCEPT_RULES.items():
        score = sum(1 for k in keywords if k in t)
        if score:
            scores[concept] = score
    if not scores:
        return "General"
    return max(scores, key=scores.get)


def infer_subjectivity(text: str) -> str:
    words = set(tokenize(text))
    if words & SUBJECTIVE_CUES:
        return "subjective"
    # Weak heuristic: objective if statement looks informational and lacks first-person cues.
    has_numeric_fact = bool(re.search(r"\b\d+[\d.,]*\b", text))
    return "objective" if has_numeric_fact else "subjective"


def infer_sentiment(text: str, subjectivity: str) -> str:
    if subjectivity == "objective":
        return "neutral"

    lowered = text.lower()
    if not any(marker in lowered for marker in EV_CONTEXT_MARKERS):
        return "neutral"

    words = tokenize(text)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    pos += sum(1 for phrase in EV_POSITIVE_PHRASES if phrase in lowered)
    neg += sum(1 for phrase in EV_NEGATIVE_PHRASES if phrase in lowered)

    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def infer_country(text: str) -> str:
    tokens = set(tokenize(text))
    for country, symbols in SYMBOL_COUNTRY_RULES.items():
        if any(symbol in text for symbol in symbols):
            return country

    for country, markers in COUNTRY_RULES.items():
        if any(marker in tokens for marker in markers):
            return country
    return "Unknown"


def detect_text_column(fieldnames: Iterable[str]) -> str:
    fields = list(fieldnames)
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in fields:
            return candidate
    raise ValueError(
        "Could not find a text column. Expected one of: "
        + ", ".join(TEXT_COLUMN_CANDIDATES)
    )


def transform_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header: {input_path}")
        text_col = detect_text_column(reader.fieldnames)

        rows = []
        for row in reader:
            text = (row.get(text_col) or "").strip()
            if not text:
                continue

            subjectivity = infer_subjectivity(text)
            existing_concept = (row.get("Concept") or "").strip()
            sentiment = infer_sentiment(text, subjectivity)

            concept = existing_concept if existing_concept else infer_concept(text)
            country = infer_country(text)

            rows.append(
                {
                    "Text": text,
                    "Sentiment": sentiment,
                    "Concept": concept,
                    "Subjective/objective": subjectivity,
                    "Country": country,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "Text",
                "Sentiment",
                "Concept",
                "Subjective/objective",
                "Country",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform CSV comments into normalized sentiment/concept format."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="files",
        help="Input CSV file or directory (default: files)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="files/transformed",
        help="Directory for transformed CSV outputs (default: files/transformed)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if input_path.is_file():
        out_file = output_dir / f"{input_path.stem}_transformed.csv"
        transform_file(input_path, out_file)
        print(f"Transformed: {input_path} -> {out_file}")
        return

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
        for csv_file in csv_files:
            out_file = output_dir / f"{csv_file.stem}_transformed.csv"
            transform_file(csv_file, out_file)
            print(f"Transformed: {csv_file} -> {out_file}")
        return

    raise FileNotFoundError(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
