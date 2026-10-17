"""
Build one clean, up-to-date used-car dataset from two recent sources.

  1. data/raw/cardekho_cars24_spinny_2025_2026.csv
       36,919 listings from CarDekho (2025 + Apr-2026), Cars24 (Jul-2025) and Spinny (Jan-Aug 2025),
       already merged and name-harmonised. Built from four public Kaggle datasets (MIT / CC0 / Apache-2.0).
       Source: github.com/Sidd-0710/car-price-prediction  (data/used_cars_india.csv)
  2. data/raw/cars24_aug_2026.csv
       6,629 Cars24 listings scraped on 30-Aug-2026 (the freshest public snapshot found).
       Source: github.com/akshayks13/BA-Case-Study-Used-Car-Pricing-and-Depreciation

Usage:  python src/build_dataset.py      ->  data/used_cars_latest.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "used_cars_latest.csv"

COLUMNS = ["Source", "Platform", "Listing_Year", "Brand", "Model", "Variant", "Variant_Key",
           "Year", "Kms_Driven", "Fuel_Type", "Transmission", "Owner", "City", "Selling_Price"]

BRAND_FIX = {"maruti": "Maruti Suzuki", "kia": "Kia", "mercedes benz": "Mercedes-Benz",
             "citroen": "Citroen", "mg": "MG"}
MODEL_ALIAS = {"wagon r 10": "wagon r", "wagon r": "wagon r", "redi go": "redigo",
               "i20 n line": "i20 n line", "urban cross": "urban cross"}
VARIANT_NOISE = {"petrol", "diesel", "cng", "bsiv", "bsvi", "bsiii", "bs4", "bs6", "bs", "bsii",
                 "dual", "tone", "dt", "mt", "manual", "edition", "new"}
VARIANT_SYN = {"opt": "o", "option": "o", "optional": "o", "automatic": "at", "auto": "at"}


def _key(text) -> str:
    text = str(text).lower().replace(".", "")
    return " ".join(re.sub(r"[^a-z0-9+]+", " ", text).split())


def model_key(model) -> str:
    k = re.sub(r"^new ", "", _key(model))
    return MODEL_ALIAS.get(k, k)


def variant_key(variant, model="") -> str:
    """'ZXi Plus AMT', 'zxi-plus-amt', 'AMT ZXI+' -> 'amt plus zxi' (order-free, noise removed)."""
    if pd.isna(variant):
        return "base"
    words, out = _key(variant).replace("+", " plus ").split(), []
    skip = set(_key(model).split())
    for w in words:
        w = VARIANT_SYN.get(w, w)
        if w in VARIANT_NOISE or w in skip or re.fullmatch(r"(19|20)\d\d", w) or w in out:
            continue
        out.append(w)
    return " ".join(sorted(out[:6])) or "base"


CITY_ALIAS = {"Delhi Ncr": "Delhi", "New Delhi": "Delhi", "Gurugram": "Gurgaon", "Bengaluru": "Bangalore",
              "Thane West": "Thane", "Dombivli East": "Dombivli", "Ranga Reddy": "Hyderabad",
              "Vadapalani": "Chennai", "Pimpri Chinchwad": "Pune", "Sambhajinagar": "Aurangabad"}
# state names / scraping fragments that are not a city
NOT_A_CITY = {"Pradesh", "Gujarat", "Uttarakhand", "Maharashtra", "Rajasthan", "Haryana", "Jharkhand",
              "Punjab", "Bengal", "Chhattisgarh", "Kerala", "Assam", "Odisha", "Karnataka", "Nadu",
              "Telangana", "Garden", "Nagar", "3", "Nit - 3", "Ap40", "Prakasam", "Unknown"}
MIN_CITY_LISTINGS = 30


def clean_city(city: pd.Series) -> pd.Series:
    c = city.fillna("Other").str.strip().str.title().replace(CITY_ALIAS)
    c = c.where(~c.isin(NOT_A_CITY), "Other")
    counts = c.value_counts()
    return c.where(c.map(counts) >= MIN_CITY_LISTINGS, "Other")


def load_multi_platform() -> pd.DataFrame:
    df = pd.read_csv(RAW / "cardekho_cars24_spinny_2025_2026.csv")
    return df


def load_cars24_2026(canonical: pd.DataFrame) -> pd.DataFrame:
    a = pd.read_csv(RAW / "cars24_aug_2026.csv")
    a["Brand"] = a["brand"].str.lower().map(BRAND_FIX).fillna(a["brand"])

    # use the same display name for a model as the big dataset ("NEXON" -> "Nexon")
    names = (canonical.assign(k=canonical["Model"].map(model_key))
             .groupby(["Brand", "k"])["Model"].agg(lambda s: s.mode().iloc[0]).to_dict())
    a["k"] = a["model"].map(model_key)
    a["Model"] = [names.get((b, k), m.title() if m.isupper() else m)
                  for b, k, m in zip(a["Brand"], a["k"], a["model"])]

    fuel = a["fuel_type"].str.strip().str.lower().map(
        {"petrol": "Petrol", "diesel": "Diesel", "cng": "CNG", "electric": "Electric"})
    out = pd.DataFrame({
        "Source": "cars24_2026", "Platform": "Cars24", "Listing_Year": 2026,
        "Brand": a["Brand"], "Model": a["Model"], "Variant": a["variant"].str.strip(),
        "Year": a["year"], "Kms_Driven": a["km_driven"], "Fuel_Type": fuel,
        "Transmission": a["transmission"].str.strip(),
        "Owner": (a["ownership"] - 1).clip(0, 3),        # previous owners, same scale as source 1
        "City": a["city"].str.strip().str.title(), "Selling_Price": a["asking_price_lakh"],
    })
    return out.dropna(subset=["Fuel_Type"])               # drops 25 hybrids + 3 EVs (too few)


def build() -> pd.DataFrame:
    multi = load_multi_platform()
    c24 = load_cars24_2026(multi)
    df = pd.concat([multi, c24], ignore_index=True)

    df["Variant_Key"] = [variant_key(v, m) for v, m in zip(df["Variant"], df["Model"])]
    # sanity filters
    df = df[df["Selling_Price"].between(0.5, 500)
            & df["Kms_Driven"].between(100, 500_000)
            & (df["Year"] <= df["Listing_Year"])
            & (df["Year"] >= 2000)]

    # the same car re-listed (across cities / snapshots): keep the newest listing
    df = df.sort_values("Listing_Year", ascending=False)
    df = df.drop_duplicates(subset=["Brand", "Model", "Year", "Kms_Driven", "Fuel_Type", "Transmission"])

    df["City"] = clean_city(df["City"])
    df = df[COLUMNS].sort_values(["Brand", "Model", "Year"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build()
    df.to_csv(OUT, index=False)
    print(f"Saved {OUT.relative_to(ROOT)}: {len(df):,} listings, {df.Brand.nunique()} brands, "
          f"{df.Model.nunique()} models, {df.Variant_Key.nunique():,} variants")
    print(df.groupby(["Platform", "Listing_Year"]).size().to_string())
