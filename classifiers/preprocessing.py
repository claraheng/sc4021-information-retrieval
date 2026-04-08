import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# Keep negation words because they are important for sentiment
NEGATION_WORDS = {"no", "not", "nor", "never", "none", "cannot"}

STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS

MICROTEXT_MAP = {
    "u": "you",
    "ur": "your",
    "urs": "yours",
    "r": "are",
    "y": "why",
    "n": "and",
    "pls": "please",
    "pls.": "please",
    "msg": "message",
    "dm": "direct message",
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "btw": "by the way",
    "omg": "oh my god",
    "lol": "laugh",
    "lmao": "laugh",
    "rofl": "laugh",
    "ttyl": "talk to you later",
    "brb": "be right back",
    "thx": "thanks",
    "ty": "thank you",
    "bc": "because",
    "bcoz": "because",
    "cuz": "because",
    "coz": "because",
    "w/": "with",
    "w/o": "without",
    "fav": "favorite",
    "gr8": "great",
    "luv": "love",
    "ya": "you",
}

EMOTICON_MAP = {
    r":\)+": " smile ",
    r":d+": " laugh ",
    r";\)+": " wink ",
    r":\(+": " sad ",
    r":/+": " skeptical ",
    r"<3": " love ",
}

CONTRACTIONS = {
    r"\bcan't\b": "can not",
    r"\bwon't\b": "will not",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bhadn't\b": "had not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
    r"\bshouldn't\b": "should not",
    r"\bn't\b": " not",
    r"\bi'm\b": "i am",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
    r"\bwhat's\b": "what is",
    r"\blet's\b": "let us",
    r"\bi've\b": "i have",
    r"\bwe've\b": "we have",
    r"\bthey've\b": "they have",
    r"\bi'll\b": "i will",
    r"\bwe'll\b": "we will",
    r"\bthey'll\b": "they will",
    r"\bi'd\b": "i would",
    r"\bwe'd\b": "we would",
    r"\bthey'd\b": "they would",
}

def normalize_repeated_chars(text: str) -> str:
    # soooo -> soo, amaaaazing -> amaazing
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def replace_emoticons(text: str) -> str:
    for pattern, replacement in EMOTICON_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def expand_contractions(text: str) -> str:
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def normalize_microtext_tokens(tokens: list[str]) -> list[str]:
    return [MICROTEXT_MAP.get(token, token) for token in tokens]

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower().strip()

    # Replace URLs, mentions, hashtags with useful placeholders/content
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)

    text = replace_emoticons(text)
    text = expand_contractions(text)
    text = normalize_repeated_chars(text)

    # Keep letters, digits, spaces, and useful punctuation for sentiment
    text = re.sub(r"[^a-z0-9!?'\s]", " ", text)

    tokens = word_tokenize(text)
    tokens = normalize_microtext_tokens(tokens)

    cleaned_tokens = []
    for token in tokens:
        if token in {"!", "?"}:
            cleaned_tokens.append(token)
            continue

        if token in STOP_WORDS:
            continue

        lemma = lemmatizer.lemmatize(token)
        if lemma.strip():
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    test_sentences = [
        "OMG I luv this!!! 😍😍",
        "idk if this is good or not...",
        "This is NOT good at all!!!",
        "brb gonna buy this lol",
        "Worst product ever :( totally useless",
        "I can't believe it's sooo amazing!!!",
        "@user this is gr8!! check it out http://example.com"
    ]

    for text in test_sentences:
        processed = preprocess_text(text)
        print("ORIGINAL :", text)
        print("PROCESSED:", processed)
        print("-" * 50)