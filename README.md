# 🚗 Used Car Price Predictor (India, 2025–26 prices)

Predicts the resale price of a used car in India from its **brand, model, variant, year, kilometres,
fuel, gearbox, ownership and city**. Trained on **40,320 real listings from CarDekho, Cars24 and Spinny
collected between January 2025 and August 2026**. Comes with a Streamlit web app.

## Results (on 8,064 listings the model never saw)

| Metric | Value |
|---|---|
| Typical error (median) | **7.7%** |
| Average error (MAPE) | **11.3%** |
| Cars predicted within ±10% | **60%** |
| Cars predicted within ±20% | **84%** |
| Mean absolute error | **₹0.93 lakh** |
| R² | 0.85 |
| 80% price range really covers | 81.5% of test cars |

### Model comparison (5-fold cross-validation + hold-out test)

| Model | CV error | Test error | Test R² | MAE (lakh) |
|---|---|---|---|---|
| Ridge (linear baseline) | 14.2% | 13.5% | 0.788 | 1.12 |
| LightGBM without variant & city | 14.6% | 13.9% | 0.852 | 1.09 |
| LightGBM (1 seed) | 12.2% | 11.4% | 0.850 | 0.94 |
| **LightGBM, 3-seed average** ✅ | **12.1%** | **11.3%** | 0.849 | **0.93** |

Adding the **variant** (VXI / ZXI+ / AX7…) and **city** cut the error from 13.9% to 11.3%.

### Compared with the previous version

| | v1 (old 2023 dataset) | v2 (this) |
|---|---|---|
| Listings | 1,105 (after removing duplicates) | **40,320** |
| Prices from | 2023 | **2025 – Aug 2026** |
| Average error | 16.2% | **11.3%** |
| Variant & city used | no | **yes** |

v1 had a higher R² (0.96), but that dataset mixed ₹1-lakh hatchbacks with ₹400-lakh supercars, and that huge
spread pushes R² up on its own. Percentage error is the fair comparison, and v2 is clearly better.

## Data

| File | Rows | Source |
|---|---|---|
| `data/raw/cardekho_cars24_spinny_2025_2026.csv` | 36,919 | CarDekho (2025, Apr-2026), Cars24 (Jul-2025), Spinny (Jan–Aug 2025). Merged from four public Kaggle datasets (MIT / CC0 / Apache-2.0) by [Sidd-0710/car-price-prediction](https://github.com/Sidd-0710/car-price-prediction) |
| `data/raw/cars24_aug_2026.csv` | 6,629 | Cars24, scraped 30-Aug-2026, from [akshayks13/BA-Case-Study-Used-Car-Pricing-and-Depreciation](https://github.com/akshayks13/BA-Case-Study-Used-Car-Pricing-and-Depreciation) |
| `data/used_cars_latest.csv` | **40,320** | Output of `src/build_dataset.py` |

What `build_dataset.py` does:
- Uses the same brand/model names in both sources ("NEXON" → "Nexon", "Maruti" → "Maruti Suzuki", "New Wagon-R" → "Wagon R").
- Turns variants into an order-free key, so "ZXi Plus AMT" and "AMT ZXI+" count as one variant.
- Cleans city names ("Gurugram" → "Gurgaon", "Delhi NCR" → "Delhi"). Cities with fewer than 30 listings become "Other".
- Removes bad rows: prices outside ₹0.5–500 lakh, odometer readings outside 100–5,00,000 km, impossible years, and hybrids/EVs from the Aug-2026 file (too few to learn from).
- Removes duplicates (the same car listed again in another city or snapshot), keeping the newest listing.

> The listings are scraped from those platforms and shared for education and research. Check the platforms'
> terms before any commercial use.

## Project structure

```
car-price-app/
├── app.py                      # Streamlit web app
├── requirements.txt
├── car_price_model.ipynb       # walkthrough notebook
├── data/
│   ├── raw/                    # the two source files
│   └── used_cars_latest.csv    # merged + cleaned dataset
├── src/
│   ├── build_dataset.py        # merge + clean the sources
│   ├── features.py             # feature engineering + model class
│   └── train.py                # compare models, save the best
└── models/
    ├── car_price_model.joblib  # trained model + dropdown catalog (8 MB)
    └── metrics.json            # all scores
```

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py

# to rebuild from scratch
python src/build_dataset.py     # a few seconds
python src/train.py             # about 2–3 minutes
```

## Features the model uses
Brand, model, variant, fuel, transmission, city, platform, car age, km driven, km per year, number of
previous owners, and listing year. The model learns **log(price)**, so a 10% error counts the same for a
₹3-lakh car as for a ₹30-lakh one.

## Limits
- These are **asking prices**, not final deal prices. Condition, accident history and service record aren't in the data.
- Rare variants and luxury cars (only a few listings) get wider errors. The app warns you about these.
