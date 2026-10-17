"""
Train, compare and save the used-car price model.

Usage:  python src/build_dataset.py   (only if data/raw changed)
        python src/train.py           (~3 min)
Outputs: models/car_price_model.joblib, models/metrics.json
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import CATEGORICAL, CURRENT_YEAR, FEATURES, NUMERIC, TARGET, CarPriceModel, add_features  # noqa: E402

SEED = 42


class RidgeLog:
    """Linear baseline: one-hot categories, ridge regression on log(price)."""
    def fit(self, X, y):
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
                                 ("num", StandardScaler(), NUMERIC)])
        self.m = make_pipeline(pre, Ridge(alpha=1.0)).fit(X[FEATURES], np.log(y))
        return self

    def predict(self, X):
        return np.exp(self.m.predict(X[FEATURES]))


class LGBMSubset(CarPriceModel):
    """Same LightGBM, but WITHOUT variant and city — shows how much those two inputs add."""
    def _prep(self, X):
        X = super()._prep(X)
        X["Trim"] = X["Model_Full"]
        X["City"] = pd.Categorical(["Other"] * len(X), categories=self.categories_["City"])
        return X


CANDIDATES = {
    "Ridge (linear baseline)": lambda: RidgeLog(),
    "LightGBM, no variant/city": lambda: LGBMSubset(seeds=(42,)),
    "LightGBM (1 seed)": lambda: CarPriceModel(seeds=(42,)),
    "LightGBM (3-seed average)": lambda: CarPriceModel(),
}


def scores(y_true, y_pred) -> dict:
    ape = np.abs(y_pred - y_true) / y_true
    return {"r2": round(float(r2_score(y_true, y_pred)), 4),
            "mae_lakhs": round(float(mean_absolute_error(y_true, y_pred)), 3),
            "mape_pct": round(float(ape.mean() * 100), 2),
            "median_ape_pct": round(float(np.median(ape) * 100), 2),
            "within_10pct": round(float((ape <= 0.10).mean() * 100), 1),
            "within_20pct": round(float((ape <= 0.20).mean() * 100), 1)}


def cross_val(make, X, y, k=5):
    """Plain k-fold CV; also returns out-of-fold predictions (used for the price range)."""
    oof = np.zeros(len(y))
    for tr, va in KFold(k, shuffle=True, random_state=SEED).split(X):
        oof[va] = make().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    return scores(y.values, oof), oof


def build_catalog(df: pd.DataFrame) -> dict:
    """Brand -> Model -> Variants (+ the usual fuel/gearbox) for the app's dropdowns."""
    cat = {}
    for (b, m), g in df.groupby(["Brand", "Model"]):
        variants = {}
        for vk, gv in g.groupby("Variant_Key"):
            label = gv["Variant"].dropna().mode()
            variants[vk] = {"label": str(label.iloc[0]).strip().upper() if len(label) else vk.upper(),
                            "fuel": gv["Fuel_Type"].mode().iloc[0],
                            "transmission": gv["Transmission"].mode().iloc[0],
                            "count": int(len(gv))}
        cat.setdefault(b, {})[m] = {
            "count": int(len(g)), "variants": variants,
            "years": [int(g["Year"].min()), int(g["Year"].max())],
            "fuel": g["Fuel_Type"].mode().iloc[0], "transmission": g["Transmission"].mode().iloc[0],
        }
    return cat


def main():
    df = add_features(pd.read_csv(ROOT / "data" / "used_cars_latest.csv"))
    print(f"{len(df):,} listings | {df.Brand.nunique()} brands | {df.Model_Full.nunique()} models | "
          f"{df.Trim.nunique():,} variants | listed {df.Listing_Year.min()}-{df.Listing_Year.max()}\n")

    X, y = df, df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    results, oof_best = {}, None
    head = f"{'Model':<28}{'CV MAPE':>9}{'CV R2':>8}{'Test MAPE':>11}{'Test R2':>9}{'MAE':>7}{'±10%':>7}"
    print(head + "\n" + "-" * len(head))
    for name, make in CANDIDATES.items():
        t = time.time()
        cv, oof = cross_val(make, X_tr, y_tr)
        test = scores(y_te.values, make().fit(X_tr, y_tr).predict(X_te))
        results[name] = {"cv": cv, "test": test, "seconds": round(time.time() - t)}
        print(f"{name:<28}{cv['mape_pct']:>8.1f}%{cv['r2']:>8.3f}{test['mape_pct']:>10.1f}%{test['r2']:>9.3f}"
              f"{test['mae_lakhs']:>7.2f}{test['within_10pct']:>6.0f}%")
        if name == "LightGBM (3-seed average)":
            oof_best = oof

    best = min(results, key=lambda k: results[k]["cv"]["mape_pct"])
    print(f"\nBest by cross-validation: {best}")

    # price range: 10th-90th percentile of honest out-of-fold errors (in log space, so it scales with price)
    log_err = np.log(y_tr.values) - np.log(oof_best)
    lo_q, hi_q = np.quantile(log_err, [0.10, 0.90])
    coverage = float(((np.log(y_te) - np.log(CarPriceModel().fit(X_tr, y_tr).predict(X_te))).between(lo_q, hi_q)).mean())
    print(f"80% price range = estimate x [{np.exp(lo_q):.3f}, {np.exp(hi_q):.3f}]  (covers {coverage:.1%} of test cars)")

    final = CANDIDATES[best]().fit(X, y)  # refit on ALL data for the app
    artifact = {
        "model": final, "model_name": best, "current_year": CURRENT_YEAR,
        "interval": [float(np.exp(lo_q)), float(np.exp(hi_q))],
        "catalog": build_catalog(df),
        "cities": sorted(c for c in df.City.unique() if c != "Other") + ["Other"],
        "platforms": sorted(df.Platform.unique()),
        "metrics": results[best]["test"],
        "n_listings": int(len(df)),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(artifact, ROOT / "models" / "car_price_model.joblib", compress=3)
    with open(ROOT / "models" / "metrics.json", "w") as f:
        json.dump({"best_model": best, "rows": int(len(df)), "price_range_coverage_on_test": round(coverage, 3),
                   "models": results}, f, indent=2)
    print("Saved models/car_price_model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
