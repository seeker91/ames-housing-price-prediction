# Ames Housing — What Drives House Prices?

A regression project predicting residential sale prices in Ames, Iowa, using
exploratory data analysis, ANOVA, feature engineering, and tuned machine
learning models (Linear Regression, Ridge, Random Forest, XGBoost).

## The question

Which features best explain the sale price of a home in Ames, Iowa, and how
accurately can we predict it?

## Dataset

- **Source:** De Cock (2011), "Ames, Iowa: Alternative to the Boston Housing
  Data as an End of Semester Regression Project", *Journal of Statistics
  Education*, 19(3). The same data as Kaggle's "House Prices: Advanced
  Regression Techniques" competition.
- **File:** `data/AmesHousing.csv` — 2,930 residential sales,
  2006–2010, 82 variables per sale (lot, dwelling, basement, garage,
  location, condition, sale details).
- **Target:** `SalePrice` (median $160,000, mean
  $180,796, range $12,789–$755,000).
  The target is right-skewed (skew 1.74), so models are
  trained on log(price) and predictions converted back to dollars.

## Methods

1. **Cleaning:** dropped ID columns (`Order`, `PID`) and two near-empty
   columns (`PoolQC`, `MiscFeature`); treated missing values explicitly —
   "None" for absent categorical features (no alley, no garage, no
   basement…), 0 for absent numeric quantities, neighborhood-median
   imputation for `LotFrontage`.
2. **EDA:** price distribution, top correlations, price by neighborhood,
   quality, living area, and year built.
3. **Feature engineering:** house age at sale, years since remodel, total
   living area, total bathrooms, total porch/deck area, and a
   quality × living-area interaction feature.
4. **ANOVA** (one-way, on log price) across neighborhood groups, house
   styles, and overall-quality bands.
5. **Modelling:** 80/20 train/test split (`random_state=42`); preprocessing
   (imputation, scaling, one-hot encoding) inside `Pipeline`s to avoid
   leakage. Linear Regression and Ridge baselines; Random Forest and
   XGBoost tuned with `RandomizedSearchCV` (10 candidates, 3-fold CV).
   Metrics reported on the test set in original dollars.

## Key findings

- **Quality and size dominate.** Strongest correlates of price:
  - `OverallQual`: r = 0.799
  - `GrLivArea`: r = 0.707
  - `GarageCars`: r = 0.648
  - `GarageArea`: r = 0.64
  - `TotalBsmtSF`: r = 0.633
- **Location matters.** Median prices range from $88,250
  (MeadowV) to $319,000 (StoneBr).
- **ANOVA:** all three factors are significant — neighborhood
  (F = 135.4, p < 0.001), house style
  (F = 65.3, p < 0.001), overall quality
  (F = 1,157.2, p < 0.001). Quality is the strongest
  group separator.
- **Engineered features helped:** the quality × area interaction
  (r = 0.84) and total living area
  (r = 0.71) rank among the strongest predictors.
- **Best model: XGBoost (tuned)** — test RMSE $22,047,
  MAE $13,468, R² = 0.939, a
  23% RMSE improvement over the Linear Regression baseline.
  For a typical $160,000 home, predictions are typically
  within about $13,468 of the actual price.
- **Top 5 features** for the best tree model (XGBoost):
  1. **Qual_x_Area** (importance 0.127)
  2. **OverallQual** (importance 0.049)
  3. **CentralAir = N** (importance 0.045)
  4. **ExterQual = TA** (importance 0.044)
  5. **CentralAir = Y** (importance 0.032)

## Model comparison (test set, original dollar scale)

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| XGBoost (tuned) | $22,047 | $13,468 | 0.939 |
| Random Forest (tuned) | $25,975 | $15,231 | 0.916 |
| Ridge (alpha=1) | $27,820 | $15,413 | 0.903 |
| Linear Regression | $28,737 | $15,069 | 0.897 |

Best tuned hyperparameters — Random Forest: `n_estimators = 300`, `min_samples_split = 5`, `min_samples_leaf = 1`, `max_features = 0.6`, `max_depth = 20`. XGBoost:
`subsample = 0.8`, `reg_lambda = 1`, `n_estimators = 500`, `max_depth = 4`, `learning_rate = 0.03`, `colsample_bytree = 0.7`.

## Files

- `ames-housing.ipynb` — the full executed notebook.
- `figures/` — all charts as PNGs (also embedded in the notebook).
- `results_summary.json` — key numbers from the run (metrics, ANOVA,
  feature importances).
- `data/AmesHousing.csv` — the dataset.

## Reproducing

Run with the project virtual environment (Python 3.12, scikit-learn 1.9.1,
XGBoost 3.4.1):

```
MPLBACKEND=Agg ../.venv/bin/python -m jupyter nbconvert --to notebook \
    --execute --inplace ames-housing.ipynb
```

Notes on the environment: the venv uses system site-packages, so matplotlib
3.6.3 comes from the OS packages while IPython is 9.x. Two compatibility
workarounds are in place — `MPLBACKEND=Agg` avoids a broken
matplotlib-inline backend, and the notebook's setup cell restores the
`backend2gui` mapping that IPython 9 removed but matplotlib 3.6.3 still
imports on first figure creation. On a setup with a newer matplotlib, both
workarounds are harmless no-ops.

## Caveats

- One city, one period (2006–2010, spanning the housing downturn) — price
  levels do not transfer to other markets.
- Very expensive homes are still under-predicted; the model is most
  reliable near the median.
- ANOVA and correlations describe associations, not causes.
