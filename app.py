"""Used Car Price Predictor (India, 2025-26 prices) — Streamlit app.   Run:  streamlit run app.py"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # so joblib can find CarPriceModel
from features import add_features  # noqa: E402

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load(ROOT / "models" / "car_price_model.joblib")


art = load_artifact()
catalog, YEAR = art["catalog"], art["current_year"]
OWNERS = {0: "1st owner", 1: "2nd owner", 2: "3rd owner", 3: "4th owner or more"}


def fmt(lakhs: float) -> str:
    return f"₹ {lakhs / 100:.2f} Cr" if lakhs >= 100 else f"₹ {lakhs:.2f} L"


def predict(row: dict, platform: str) -> float:
    df = pd.DataFrame([{**row, "Platform": platform, "Listing_Year": YEAR}])
    return float(art["model"].predict(add_features(df))[0])


# ---------------- Header ----------------
st.title("🚗 Used Car Price Predictor")
st.caption(f"Resale price estimate for India, trained on {art['n_listings']:,} CarDekho, Cars24 and Spinny "
           f"listings from 2025–2026.")

# ---------------- Car ----------------
c1, c2 = st.columns(2)
brands = sorted(catalog, key=lambda b: -sum(m["count"] for m in catalog[b].values()))
brand = c1.selectbox("Brand", brands)
models = sorted(catalog[brand], key=lambda m: -catalog[brand][m]["count"])
model = c2.selectbox("Model", models)
info = catalog[brand][model]

variants = sorted(info["variants"], key=lambda v: -info["variants"][v]["count"])
variant = st.selectbox("Variant", variants,
                       format_func=lambda v: f"{info['variants'][v]['label']}  ·  {info['variants'][v]['count']} listings")
vinfo = info["variants"][variant]

c3, c4 = st.columns(2)
fuels = ["Petrol", "Diesel", "CNG", "Electric"]
fuel = c3.selectbox("Fuel", fuels, index=fuels.index(vinfo["fuel"]))
gears = ["Manual", "Automatic"]
trans = c4.selectbox("Transmission", gears, index=gears.index(vinfo["transmission"]))

# ---------------- Condition ----------------
c5, c6 = st.columns(2)
y_lo, y_hi = max(2000, info["years"][0] - 2), YEAR
year = c5.slider("Registration year", y_lo, y_hi, min(max(2019, y_lo), y_hi))
kms = c6.number_input("KMs driven", min_value=0, max_value=500_000, value=min(45_000, (YEAR - year + 1) * 10_000),
                      step=1_000)

c7, c8 = st.columns(2)
owner = c7.selectbox("Ownership", list(OWNERS), format_func=OWNERS.get)
cities = art["cities"]
city = c8.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0,
                    help="Pick 'Other' if your city isn't listed.")

# ---------------- Predict ----------------
if st.button("Estimate price", type="primary", use_container_width=True):
    row = {"Brand": brand, "Model": model, "Variant_Key": variant, "Year": year, "Kms_Driven": max(kms, 100),
           "Owner": owner, "City": city, "Fuel_Type": fuel, "Transmission": trans}
    by_platform = {p: predict(row, p) for p in art["platforms"]}
    price = sum(by_platform.values()) / len(by_platform)
    lo, hi = price * art["interval"][0], price * art["interval"][1]

    st.success(f"### Estimated price: {fmt(price)}")
    st.write(f"80% likely range: **{fmt(lo)} – {fmt(hi)}**")

    cols = st.columns(len(by_platform))
    for col, (p, v) in zip(cols, by_platform.items()):
        col.metric(f"On {p}", fmt(v))
    st.caption("Each platform prices a little differently; the headline figure is their average.")

    if vinfo["count"] < 5:
        st.warning(f"Only {vinfo['count']} listing(s) of this exact variant in the data — the estimate leans on "
                   f"similar {model} variants, so treat it as rough.")
    if year < info["years"][0]:
        st.warning(f"No {model} older than {info['years'][0]} in the data — estimate is an extrapolation.")
