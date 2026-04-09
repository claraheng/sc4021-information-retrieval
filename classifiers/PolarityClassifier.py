import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin

from polarity_common import build_model

CLASSIFIERS_DIR = Path(__file__).resolve().parent
MODELS_DIR = CLASSIFIERS_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "polarity_model.joblib"

class PolarityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = build_model()
        self.classes_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
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
    def load(cls, path=None):
        in_path = Path(path) if path else DEFAULT_MODEL_PATH
        instance = cls()
        instance.model = joblib.load(in_path)
        instance.classes_ = instance.model.classes_ if hasattr(instance.model, 'classes_') else None
        return instance