import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DailyShapeNormalizer(BaseEstimator, TransformerMixin):
    """Normalize each daily profile by its own mean consumption."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        row_means = values.mean(axis=1, keepdims=True)
        safe_means = np.where(np.abs(row_means) < 1e-12, 1.0, row_means)
        return values / safe_means


def build_preprocessor(mode: str) -> Pipeline:
    if mode not in {"absolute", "shape"}:
        raise ValueError("mode must be 'absolute' or 'shape'")

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if mode == "shape":
        steps.append(("shape_normalizer", DailyShapeNormalizer()))
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)

