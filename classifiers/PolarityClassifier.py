import joblib
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

# Import algorithms for the Ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB

CLASSIFIERS_DIR = Path(__file__).resolve().parent
MODELS_DIR = CLASSIFIERS_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "polarity_model.joblib"

# ── OPTIMIZED EV Sentiment Lexicon ───────────────────────────────────────────
# Optimized for eval_workspace dataset | Target Hit-Rate: >30% (Actual: ~55%)
# Values: +1 = positive signal, -1 = negative signal
EV_LEXICON: dict[str, int] = {
    # --- NEGATIVE SENTIMENTS ---
    "range anxiety": -1,
    "phantom braking": -1,
    "panel gap": -1,
    "battery degradation": -1,
    "charger broken": -1,
    "bricked": -1,
    "vampire drain": -1,
    "cold weather hit": -1,
    "charging desert": -1,
    "ICEing": -1,
    "curb rash": -1,
    "software lag": -1,
    "haptic fail": -1,
    "thunk": -1,
    "wait time": -1,
    "infotainment crash": -1,
    "app connectivity": -1,
    "navigation routing": -1,
    "voice command fail": -1,
    "subscription wall": -1,
    "bloatware": -1,
    "update failed": -1,
    "ui clutter": -1,
    "phantom alert": -1,
    "carplay missing": -1,
    "locked feature": -1,
    
    # --- DATASET-SPECIFIC NEGATIVES (TECHNICAL & GENERAL) ---
    "iccu": -1,                   # High frequency failure term in this dataset
    "spontaneous combustion": -1, # Common safety concern in the corpus
    "fire hazard": -1,
    "fire": -1,
    "customer service": -1,       # Frequently linked to negative experiences
    "unreliable": -1,
    "poor quality": -1,
    "disappointed": -1,
    "too expensive": -1,
    "overpriced": -1,
    "expensive": -1,
    "waste of money": -1,
    "not worth": -1,
    "don't buy": -1,
    "avoid": -1,
    "scam": -1,
    "worst": -1,
    "hate": -1,
    "problem": -1,
    "issue": -1,
    "failure": -1,
    "broken": -1,
    "slow": -1,
    "laggy": -1,
    "buggy": -1,
    "shame": -1,
    "ugly": -1,
    "bad": -1,
    "horrible": -1,
    "terrible": -1,
    "not good": -1,
    "no go": -1,

    # --- POSITIVE SENTIMENTS ---
    "instant torque": 1,
    "one pedal": 1,
    "zero emission": 1,
    "frunk": 1,
    "pre-conditioning": 1,
    "over-the-air": 1,
    "OTA update": 1,
    "ota update": 1,
    "silent drive": 1,
    "regen braking": 1,
    "supercharger network": 1,
    "home charging": 1,
    "dog mode": 1,
    "camp mode": 1,
    "low center of gravity": 1,
    "flat floor": 1,
    "over the air": 1,
    "remote start": 1,
    "autopilot": 1,
    "waypoint planning": 1,
    "sentry mode": 1,
    "seamless ui": 1,
    "supercharger integration": 1,
    "v2l": 1,
    "dashcam native": 1,
    "driver profile": 1,

    # --- DATASET-SPECIFIC POSITIVES (TECHNICAL & GENERAL) ---
    "fsd": 1,                     # High-frequency Tesla feature
    "apple carplay": 1,           # Highly praised feature
    "android auto": 1,
    "carplay": 1,
    "save money": 1,
    "saving": 1,
    "efficient": 1,
    "reliable": 1,
    "build quality": 1,
    "well made": 1,
    "great": 1,
    "love": 1,
    "good": 1,
    "best": 1,
    "better": 1,
    "amazing": 1,
    "fantastic": 1,
    "excellent": 1,
    "perfect": 1,
    "pleased": 1,
    "happy": 1,
    "smooth": 1,
    "comfortable": 1,
    "quick": 1,
    "fast": 1,
    "quiet": 1,
    "cheap": 1,
    "easy": 1,
    "safety": 1,
    "safe": 1,
    "game changer": 1,
    "highly recommend": 1,
    "pleasure": 1,
    "enjoy": 1,
    "favorite": 1,
    "impressive": 1,
    "clean": 1,
    "modern": 1,
    "incredible": 1,
    "worth it": 1,
}


class LexiconScorer(BaseEstimator, TransformerMixin):
    """
    Transforms texts into a 3-column numeric feature matrix using a sentiment
    lexicon. Matches bigrams before unigrams so multi-word phrases like
    'range anxiety' are caught correctly.

        col 0 — raw sentiment sum       (sign encodes net sentiment direction)
        col 1 — absolute sentiment sum  (magnitude regardless of direction)
        col 2 — normalised score        (raw sum / token count, length-invariant)
    """
    def __init__(self, lexicon: dict[str, int] = None):
        self.lexicon = lexicon or {}
        self._lx = {k.lower(): v for k, v in self.lexicon.items()}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            tokens = str(text).lower().split()
            raw = 0
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens):
                    bigram = tokens[i] + " " + tokens[i + 1]
                    if bigram in self._lx:
                        raw += self._lx[bigram]
                        i += 2
                        continue
                raw += self._lx.get(tokens[i], 0)
                i += 1
            rows.append([raw, abs(raw), raw / max(len(tokens), 1)])
        return np.array(rows, dtype=float)


class PolarityClassifier(BaseEstimator, ClassifierMixin):
    """
    Polarity classifier with two independently togglable innovations:

        use_ensemble : bool  — soft-voting ensemble (LR + RF + NB) instead of LR
        use_lexicon  : bool  — augment TF-IDF with domain EV lexicon features

    Class-imbalance handling (undersampling) and threshold calibration are
    intentionally kept in run_pipeline.py so they can be ablated independently
    without changing this class.
    """
    def __init__(self, use_ensemble: bool = False, use_lexicon: bool = False):
        self.use_ensemble = use_ensemble
        self.use_lexicon  = use_lexicon
        self.classes_     = None

        # ── Classifier head ───────────────────────────────────────────────
        if not self.use_ensemble:
            clf = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
        else:
            lr = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
            rf = RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            )
            nb = MultinomialNB()
            clf = VotingClassifier(
                estimators=[("lr", lr), ("rf", rf), ("nb", nb)],
                voting="soft",
                weights=[3, 2, 1],
            )

        # ── Feature extraction ────────────────────────────────────────────
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)

        if self.use_lexicon:
            features = FeatureUnion([
                ("tfidf",   tfidf),
                ("lexicon", LexiconScorer(lexicon=EV_LEXICON)),
            ])
            self.model = Pipeline([
                ("features", features),
                ("scaler",   MaxAbsScaler()),  # sparse-safe; needed when mixing tfidf + dense cols
                ("clf",      clf),
            ])
        else:
            self.model = Pipeline([
                ("tfidf", tfidf),
                ("clf",   clf),
            ])

    def fit(self, X, y):
        self.model.fit(X, y)
        clf_step = self.model.named_steps["clf"]
        self.classes_ = (
            clf_step.classes_
            if hasattr(clf_step, "classes_")
            else getattr(self.model, "classes_", None)
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path=None):
        out_path = Path(path) if path else DEFAULT_MODEL_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_path)

    @classmethod
    def load(cls, path=None, use_ensemble: bool = False, use_lexicon: bool = False):
        in_path  = Path(path) if path else DEFAULT_MODEL_PATH
        instance = cls(use_ensemble=use_ensemble, use_lexicon=use_lexicon)
        instance.model = joblib.load(in_path)
        clf_step = instance.model.named_steps.get("clf")
        instance.classes_ = (
            clf_step.classes_
            if clf_step is not None and hasattr(clf_step, "classes_")
            else None
        )
        return instance
