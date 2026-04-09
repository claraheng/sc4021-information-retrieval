import pandas as pd
from pathlib import Path
from preprocessing import preprocess_text

# Load data relative to this script (works regardless of current working directory)
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_PATH = SCRIPT_DIR / "eval.xls"
df = pd.read_excel(EVAL_PATH, engine="xlrd")

# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# Count words in original text
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

# Measure how different the processed text is
df["change"] = df.apply(
    lambda row: len(set(str(row["text"]).lower().split()) - set(row["clean_text"].split())),
    axis=1
)

# Filter:
# 1. Less than 10 words
# 2. Has some meaningful change
filtered = df[(df["word_count"] <= 10) & (df["change"] >= 2)]

# Get top 10 most interesting ones
top_examples = filtered.sort_values(by="change", ascending=False).head(10)


# Save
top_examples[["text", "clean_text"]].to_excel("classifiers/preprocessed_examples.xlsx", index=False)