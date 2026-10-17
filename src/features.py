"""Feature engineering + the model wrapper used by both train.py and the app."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

CURRENT_YEAR = 2026  # newest listings in the data are from Aug-2026

CATEGORICAL = ["Brand", "Model_Full", "Trim", "Fuel_Type", "Transmission", "Platform", "City"]
NUMERIC = ["Age", "Kms_Driven", "Kms_Per_Year", "Owner", "Listing_Year"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Selling_Price"  # lakhs

LGBM_PARAMS = dict(
    n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=10,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    cat_smooth=5, min_data_per_group=10, cat_l2=10, verbose=-1,
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> model features. Works for the training CSV and for a single app input row."""
    out = df.copy()
    out["Age"] = out["Listing_Year"] - out["Year"]
    out["Kms_Per_Year"] = out["Kms_Driven"] / np.maximum(out["Age"], 0.5)
    out["Model_Full"] = out["Brand"] + "|" + out["Model"]                 # 'City' exists for Honda AND as a city
    out["Trim"] = out["Model_Full"] + "|" + out["Variant_Key"].fillna("base")
    return out


class CarPriceModel:
    """LightGBM on log(price), averaged over a few random seeds for stability.

    Categorical columns are handled natively by LightGBM (no one-hot needed, works with
    thousands of variants). Unknown categories at prediction time simply fall back to
    the model's general behaviour for that feature.
    """

    def __init__(self, seeds=(42, 7, 2026), **params):
        self.seeds = seeds
        self.params = {**LGBM_PARAMS, **params}

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURES].copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        return X

    def fit(self, X: pd.DataFrame, y_lakhs: pd.Series):
        self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in CATEGORICAL}
        Xp, ylog = self._prep(X), np.log(y_lakhs)
        self.models_ = [LGBMRegressor(random_state=s, **self.params).fit(Xp, ylog) for s in self.seeds]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xp = self._prep(X)
        return np.exp(np.mean([m.predict(Xp) for m in self.models_], axis=0))

    def get_params(self, deep=False):  # lets sklearn's cross_validate clone it
        return {"seeds": self.seeds, **self.params}

    def set_params(self, **p):
        self.seeds = p.pop("seeds", self.seeds)
        self.params.update(p)
        return self
