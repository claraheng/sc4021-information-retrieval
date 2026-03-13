import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================
SEED_DATA_PATH = "ev_twitter_FINAL_dataset_v9.csv"   # your manually reviewed seed dataset
NEW_DATA_PATH = "new_scraped_tweets.csv"             # your newly scraped 2k+ tweets

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AUTO_LABELED_PATH = OUTPUT_DIR / "auto_labeled_ev_tweets.csv"
REVIEW_PATH = OUTPUT_DIR / "ev_tweets_for_manual_review.csv"
CLEANED_NEW_DATA_PATH = OUTPUT_DIR / "cleaned_new_scraped_tweets.csv"


# =========================================================
# KEYWORDS / RULES
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

CONCEPT_KEYWORDS: Dict[str, List[str]] = {
    "Range & Battery": [
        "battery", "batteries", "range", "range anxiety", "degrade",
        "degradation", "kwh", "battery life", "battery replacement"
    ],
    "Charging": [
        "charging", "charger", "charge", "supercharger",
        "charging station", "charging stations", "plug",
        "plug in", "fast charge", "home charging"
    ],
    "Performance": [
        "torque", "acceleration", "accelerate", "fast", "faster",
        "speed", "fun to drive", "handling", "0-60", "driving fun"
    ],
    "Software/Tech": [
        "autopilot", "software", "update", "updates", "infotainment",
        "screen", "app", "driverless", "self-driving", "tech"
    ],
    "Cost & Value": [
        "expensive", "cheap", "affordable", "cost", "price", "priced",
        "value", "maintenance", "resale", "insurance", "cheaper"
    ],
    "Build & Comfort": [
        "interior", "build quality", "panel gaps", "comfort", "quiet",
        "noise", "seat", "seats", "ride quality", "fire", "catch fire",
        "futuristic"
    ],
    "Sustainability": [
        "pollution", "emissions", "emission", "carbon", "co2",
        "environment", "green", "sustainable", "lithium mining",
        "oil dependence", "fossil fuel"
    ]
}

POS_WORDS = [
    "good", "great", "amazing", "better", "best", "love", "like",
    "affordable", "cheaper", "improving", "improve", "future",
    "convenient", "fun", "cleaner", "reduce", "reduces", "faster",
    "quiet", "modern", "cool", "impressive", "excellent"
]

NEG_WORDS = [
    "bad", "worse", "worst", "expensive", "costly", "hate", "slow",
    "too long", "problem", "issue", "limited", "rare", "inconvenient",
    "degrade", "degradation", "harmful", "damages", "terrible",
    "unreliable", "cheap", "dangerous"
]

SUBJECTIVE_WORDS = [
    "i think", "i like", "i love", "i hate", "should", "better",
    "worse", "best", "worst", "amazing", "terrible", "fun", "cool",
    "overrated", "future", "expensive", "cheap", "good", "bad",
    "impressive", "not good enough", "too expensive"
]

COUNTRY_KEYWORDS = {
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
    text = re.sub(r"http\S+", " ", text)      # remove URLs
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)         # remove mentions
    text = re.sub(r"#", "", text)             # remove hashtag symbol only
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
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


def label_concept_rule(text: str) -> Tuple[Optional[str], int]:
    scores: Dict[str, int] = {}
    for concept, keywords in CONCEPT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[concept] = score
    if not scores:
        return None, 0
    best = max(scores, key=scores.get)
    return best, scores[best]


def label_sentiment_rule(text: str) -> Tuple[str, int]:
    pos_score = sum(1 for w in POS_WORDS if w in text)
    neg_score = sum(1 for w in NEG_WORDS if w in text)

    if pos_score > neg_score:
        return "positive", pos_score
    if neg_score > pos_score:
        return "negative", neg_score
    if pos_score == 0 and neg_score == 0:
        return "neutral", 0
    return "neutral", max(pos_score, neg_score)


def label_subjectivity_rule(text: str) -> Tuple[str, int]:
    score = sum(1 for w in SUBJECTIVE_WORDS if w in text)
    if score > 0:
        return "subjective", score
    return "objective", 0


def is_probably_political_not_ev_main_topic(text: str) -> bool:
    """
    Remove rows where politics/geopolitics appears to be the main topic.
    Keeps some general EV tweets; only removes obvious cases.
    """
    text = text.lower()

    has_ev = contains_any(text, EV_KEYWORDS)
    if not has_ev:
        return True

    has_political = contains_any(text, POLITICAL_KEYWORDS)
    has_geo = contains_any(text, GEOPOLITICAL_KEYWORDS)
    has_personality = contains_any(text, PERSONALITY_KEYWORDS)

    # Strong EV aspect talk
    concept_label, concept_score = label_concept_rule(text)

    # Heuristic:
    # remove if politics/geo/personality are present and there is no strong EV aspect
    if (has_political or has_geo or has_personality) and concept_score == 0:
        return True

    return False


def load_csv_flex(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"'text' column not found in {path}")
    return df


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print("Loading datasets...")
    seed_df = load_csv_flex(SEED_DATA_PATH)
    new_df = load_csv_flex(NEW_DATA_PATH)

    # Ensure required seed columns exist
    required_seed_cols = ["text", "sentiment", "concept", "subjective/objective"]
    missing = [c for c in required_seed_cols if c not in seed_df.columns]
    if missing:
        raise ValueError(f"Seed dataset is missing columns: {missing}")

    # Clean text
    seed_df = seed_df.dropna(subset=["text"]).copy()
    new_df = new_df.dropna(subset=["text"]).copy()

    seed_df["clean_text"] = seed_df["text"].apply(clean_text)
    new_df["clean_text"] = new_df["text"].apply(clean_text)

    # Remove duplicates and very short tweets
    before = len(new_df)
    new_df = new_df.drop_duplicates(subset=["clean_text"]).copy()
    new_df = new_df[new_df["clean_text"].str.split().str.len() >= 5].copy()
    print(f"After dedupe/short filter: {before} -> {len(new_df)}")

    # Keep EV-related tweets only
    before = len(new_df)
    new_df = new_df[new_df["clean_text"].apply(lambda x: contains_any(x, EV_KEYWORDS))].copy()
    print(f"After EV filter: {before} -> {len(new_df)}")

    # Remove obviously political/geopolitical/personality-first tweets
    before = len(new_df)
    new_df = new_df[~new_df["clean_text"].apply(is_probably_political_not_ev_main_topic)].copy()
    print(f"After political/geopolitical filter: {before} -> {len(new_df)}")

    # Save cleaned new data
    new_df.to_csv(CLEANED_NEW_DATA_PATH, index=False)

    # Vectorize for similarity search
    print("Building TF-IDF similarity index...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    combined_text = pd.concat([seed_df["clean_text"], new_df["clean_text"]], axis=0)
    X_all = vectorizer.fit_transform(combined_text)

    X_seed = X_all[: len(seed_df)]
    X_new = X_all[len(seed_df) :]

    sim_matrix = cosine_similarity(X_new, X_seed)

    # Reset index so sim row index matches dataframe row order
    new_df = new_df.reset_index(drop=True)
    seed_df = seed_df.reset_index(drop=True)

    results = []

    print("Labeling new tweets...")
    for i, row in new_df.iterrows():
        text = row["clean_text"]

        rule_concept, concept_score = label_concept_rule(text)
        rule_sentiment, sent_score = label_sentiment_rule(text)
        rule_subjectivity, subj_score = label_subjectivity_rule(text)

        sim_row = sim_matrix[i]
        top_k = 3
        top_idx = sim_row.argsort()[-top_k:][::-1]
        top_scores = sim_row[top_idx]
        top_rows = seed_df.iloc[top_idx]

        sim_concept = top_rows["concept"].value_counts().idxmax()
        sim_sentiment = top_rows["sentiment"].value_counts().idxmax()
        sim_subjectivity = top_rows["subjective/objective"].value_counts().idxmax()
        sim_score = float(top_scores.mean())

        # Final concept
        if concept_score >= 1:
            final_concept = rule_concept
        else:
            final_concept = sim_concept

        # Final sentiment
        if sent_score >= 1:
            final_sentiment = rule_sentiment
        else:
            final_sentiment = sim_sentiment

        # Final subjectivity
        if subj_score >= 1:
            final_subjectivity = rule_subjectivity
        else:
            final_subjectivity = sim_subjectivity

        # Country
        if "country" in row and pd.notna(row["country"]):
            country = row["country"]
        else:
            country = detect_country(text)

        # Review logic
        review_flag = False
        review_reason = []

        if concept_score == 0:
            review_flag = True
            review_reason.append("no_rule_concept_match")

        if sent_score == 0 and sim_score < 0.25:
            review_flag = True
            review_reason.append("uncertain_sentiment")

        if sim_score < 0.20:
            review_flag = True
            review_reason.append("low_similarity_to_seed")

        if final_concept is None:
            review_flag = True
            review_reason.append("missing_concept")

        results.append({
            "text": row["text"],
            "clean_text": text,
            "sentiment": final_sentiment,
            "concept": final_concept,
            "subjective/objective": final_subjectivity,
            "country": country,
            "rule_concept_score": concept_score,
            "rule_sentiment_score": sent_score,
            "rule_subjectivity_score": subj_score,
            "similarity_score": round(sim_score, 4),
            "review_flag": review_flag,
            "review_reason": ";".join(review_reason) if review_reason else ""
        })

    labeled_df = pd.DataFrame(results)

    # Save outputs
    labeled_df.to_csv(AUTO_LABELED_PATH, index=False)
    review_df = labeled_df[labeled_df["review_flag"] == True].copy()
    review_df.to_csv(REVIEW_PATH, index=False)

    print("\nDone.")
    print(f"Auto-labeled rows: {len(labeled_df)}")
    print(f"Rows needing manual review: {len(review_df)}")
    print(f"Saved: {AUTO_LABELED_PATH}")
    print(f"Saved: {REVIEW_PATH}")
    print(f"Saved: {CLEANED_NEW_DATA_PATH}")


if __name__ == "__main__":
    main()