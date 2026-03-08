import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# =========================================================
# CONFIG
# =========================================================
SEED_DATA_PATH = "ev_twitter_FINAL_dataset_v9.csv"   # manually reviewed dataset
NEW_DATA_PATH = "new_scraped_tweets.csv"             # new scraped tweets

OUTPUT_DIR = Path("outputs_sklearn")
OUTPUT_DIR.mkdir(exist_ok=True)

LABELED_OUTPUT_PATH = OUTPUT_DIR / "sklearn_labeled_ev_tweets.csv"
REVIEW_OUTPUT_PATH = OUTPUT_DIR / "sklearn_ev_tweets_for_review.csv"
CLEANED_NEW_DATA_PATH = OUTPUT_DIR / "sklearn_cleaned_new_scraped_tweets.csv"


# =========================================================
# FILTERING RULES
# =========================================================
EV_KEYWORDS = [
    "ev", "evs",
    "electric car", "electric cars",
    "electric vehicle", "electric vehicles",
    "tesla", "battery", "batteries",
    "charging", "charger", "charge",
    "range", "autopilot", "supercharger",
    "electric"
]

POLITICAL_KEYWORDS = [
    "government", "policy", "policies", "tariff", "tariffs",
    "tax", "taxes", "democrat", "democrats",
    "republican", "republicans", "biden", "trump",
    "labor", "labour", "subsidy", "subsidies",
    "trade war", "immigration", "geopolitics"
]

GEOPOLITICAL_KEYWORDS = [
    "country", "nation", "economy", "trade", "sanction",
    "import", "export", "china", "india", "zimbabwe",
    "canada", "australia", "britain"
]

PERSONALITY_KEYWORDS = [
    "elon musk", "musk", "president", "prime minister", "ceo"
]

COUNTRY_KEYWORDS: Dict[str, List[str]] = {
    "Australia": ["australia", "australian", "victoria", "victorian"],
    "China": ["china", "chinese"],
    "Canada": ["canada", "canadian"],
    "India": ["india", "indian"],
    "Zimbabwe": ["zimbabwe"],
    "United States": ["united states", "america", "american", "usa", "u.s."],
    "United Kingdom": ["united kingdom", "britain", "british", "uk", "u.k."],
}


# =========================================================
# HELPERS
# =========================================================
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def contains_any(text: str, keywords: List[str]) -> bool:
    text = text.lower()
    return any(keyword.lower() in text for keyword in keywords)


def detect_country(text: str) -> str:
    text = text.lower()
    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return country
    return "Unknown"


def is_probably_political_not_ev_main_topic(text: str) -> bool:
    text = text.lower()

    has_ev = contains_any(text, EV_KEYWORDS)
    if not has_ev:
        return True

    has_political = contains_any(text, POLITICAL_KEYWORDS)
    has_geo = contains_any(text, GEOPOLITICAL_KEYWORDS)
    has_personality = contains_any(text, PERSONALITY_KEYWORDS)

    # If politics/geo/personality is present and there's no EV-specific aspect,
    # likely not EV-main-topic.
    ev_aspects = [
        "battery", "charging", "charger", "range", "torque", "accelerat",
        "maintenance", "autopilot", "infrastructure", "lithium",
        "emission", "emissions", "supercharger", "panel gaps",
        "build quality", "interior", "quiet"
    ]
    has_ev_aspect = contains_any(text, ev_aspects)

    if (has_political or has_geo or has_personality) and not has_ev_aspect:
        return True

    return False


def load_csv_flex(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"'text' column not found in {path}")
    return df


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=3000))
    ])


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print("Loading datasets...")
    seed_df = load_csv_flex(SEED_DATA_PATH)
    new_df = load_csv_flex(NEW_DATA_PATH)

    required_seed_cols = ["text", "sentiment", "concept", "subjective/objective"]
    missing = [c for c in required_seed_cols if c not in seed_df.columns]
    if missing:
        raise ValueError(f"Seed dataset is missing columns: {missing}")

    # Clean text
    seed_df = seed_df.dropna(subset=["text"]).copy()
    new_df = new_df.dropna(subset=["text"]).copy()

    seed_df["clean_text"] = seed_df["text"].apply(clean_text)
    new_df["clean_text"] = new_df["text"].apply(clean_text)

    # Remove duplicates and short tweets
    before = len(new_df)
    new_df = new_df.drop_duplicates(subset=["clean_text"]).copy()
    new_df = new_df[new_df["clean_text"].str.split().str.len() >= 5].copy()
    print(f"After dedupe/short filter: {before} -> {len(new_df)}")

    # EV filter
    before = len(new_df)
    new_df = new_df[new_df["clean_text"].apply(lambda x: contains_any(x, EV_KEYWORDS))].copy()
    print(f"After EV filter: {before} -> {len(new_df)}")

    # Political/geopolitical filter
    before = len(new_df)
    new_df = new_df[~new_df["clean_text"].apply(is_probably_political_not_ev_main_topic)].copy()
    print(f"After political/geopolitical filter: {before} -> {len(new_df)}")

    # Save cleaned new tweets
    new_df.to_csv(CLEANED_NEW_DATA_PATH, index=False)

    # Train models
    print("Training sklearn models...")
    concept_model = build_model()
    sentiment_model = build_model()
    subjectivity_model = build_model()

    concept_model.fit(seed_df["clean_text"], seed_df["concept"])
    sentiment_model.fit(seed_df["clean_text"], seed_df["sentiment"])
    subjectivity_model.fit(seed_df["clean_text"], seed_df["subjective/objective"])

    # Predict labels
    print("Predicting labels...")
    new_df = new_df.reset_index(drop=True)

    pred_concept = concept_model.predict(new_df["clean_text"])
    pred_sentiment = sentiment_model.predict(new_df["clean_text"])
    pred_subjectivity = subjectivity_model.predict(new_df["clean_text"])

    concept_probs = concept_model.predict_proba(new_df["clean_text"])
    sentiment_probs = sentiment_model.predict_proba(new_df["clean_text"])
    subjectivity_probs = subjectivity_model.predict_proba(new_df["clean_text"])

    concept_conf = concept_probs.max(axis=1)
    sentiment_conf = sentiment_probs.max(axis=1)
    subjectivity_conf = subjectivity_probs.max(axis=1)

    overall_conf = (concept_conf + sentiment_conf + subjectivity_conf) / 3.0

    # Build final output
    rows = []
    for i, row in new_df.iterrows():
        if "country" in row and pd.notna(row["country"]):
            country = row["country"]
        else:
            country = detect_country(row["clean_text"])

        review_flag = (
            concept_conf[i] < 0.45
            or sentiment_conf[i] < 0.45
            or subjectivity_conf[i] < 0.45
            or overall_conf[i] < 0.50
        )

        review_reason = []
        if concept_conf[i] < 0.45:
            review_reason.append("low_concept_confidence")
        if sentiment_conf[i] < 0.45:
            review_reason.append("low_sentiment_confidence")
        if subjectivity_conf[i] < 0.45:
            review_reason.append("low_subjectivity_confidence")
        if overall_conf[i] < 0.50:
            review_reason.append("low_overall_confidence")

        rows.append({
            "text": row["text"],
            "clean_text": row["clean_text"],
            "sentiment": pred_sentiment[i],
            "concept": pred_concept[i],
            "subjective/objective": pred_subjectivity[i],
            "country": country,
            "concept_confidence": round(float(concept_conf[i]), 4),
            "sentiment_confidence": round(float(sentiment_conf[i]), 4),
            "subjectivity_confidence": round(float(subjectivity_conf[i]), 4),
            "overall_confidence": round(float(overall_conf[i]), 4),
            "review_flag": review_flag,
            "review_reason": ";".join(review_reason) if review_reason else ""
        })

    labeled_df = pd.DataFrame(rows)
    review_df = labeled_df[labeled_df["review_flag"] == True].copy()

    labeled_df.to_csv(LABELED_OUTPUT_PATH, index=False)
    review_df.to_csv(REVIEW_OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Labeled rows: {len(labeled_df)}")
    print(f"Rows needing review: {len(review_df)}")
    print(f"Saved: {LABELED_OUTPUT_PATH}")
    print(f"Saved: {REVIEW_OUTPUT_PATH}")
    print(f"Saved: {CLEANED_NEW_DATA_PATH}")


if __name__ == "__main__":
    main()