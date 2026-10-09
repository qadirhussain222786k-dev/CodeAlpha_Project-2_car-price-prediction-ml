"""
Task 3: Car Price Prediction with Machine Learning
--------------------------------------------------
Predicts the selling price of used cars (in lakhs) from features such as
brand goodwill, present (showroom) price, age, mileage, fuel type, etc.

Workflow: load -> clean -> feature engineering -> EDA -> preprocessing
          -> model training & comparison -> tuning -> evaluation -> save results

Run:  python car_price_prediction.py
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "car_data.csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("=" * 70, "\n1. DATASET OVERVIEW\n" + "=" * 70)
print(f"Shape: {df.shape}")
print(df.head(), "\n")
print(df.info(), "\n")
print(df.describe().round(2), "\n")

# ----------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------
print("=" * 70, "\n2. DATA CLEANING\n" + "=" * 70)
print("Missing values:", int(df.isna().sum().sum()))
dups = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed: {dups}  -> new shape {df.shape}")

# Clean text columns
df["Car_Name"] = df["Car_Name"].str.strip().str.lower()

# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------------
print("=" * 70, "\n3. FEATURE ENGINEERING\n" + "=" * 70)
CURRENT_YEAR = df["Year"].max() + 1          # dataset is from 2018 -> reference year 2019
df["Car_Age"] = CURRENT_YEAR - df["Year"]

# Brand extraction. The dataset lists model names (e.g. 'ritz', 'fortuner') and bike names
# (e.g. 'bajaj pulsar 150'), so we map each entry to its manufacturer.
CAR_MODEL_TO_BRAND = {
    "ritz": "Maruti", "sx4": "Maruti", "ciaz": "Maruti", "wagon r": "Maruti", "swift": "Maruti",
    "vitara brezza": "Maruti", "s cross": "Maruti", "alto 800": "Maruti", "alto k10": "Maruti",
    "ertiga": "Maruti", "dzire": "Maruti", "omni": "Maruti", "baleno": "Maruti", "ignis": "Maruti",
    "800": "Maruti",
    "fortuner": "Toyota", "innova": "Toyota", "corolla altis": "Toyota", "corolla": "Toyota",
    "etios cross": "Toyota", "etios g": "Toyota", "etios liva": "Toyota", "etios gd": "Toyota",
    "camry": "Toyota", "land cruiser": "Toyota",
    "creta": "Hyundai", "i20": "Hyundai", "grand i10": "Hyundai", "i10": "Hyundai",
    "eon": "Hyundai", "xcent": "Hyundai", "elantra": "Hyundai", "verna": "Hyundai",
    "city": "Honda", "brio": "Honda", "amaze": "Honda", "jazz": "Honda",
}
TWO_WHEELER_PREFIXES = {
    "bajaj": "Bajaj", "hero": "Hero", "honda": "Honda", "activa": "Honda", "ktm": "KTM",
    "royal enfield": "Royal Enfield", "yamaha": "Yamaha", "tvs": "TVS", "suzuki": "Suzuki",
    "hyosung": "Hyosung", "mahindra": "Mahindra", "um ": "UM",
}

def get_brand(name: str) -> str:
    """Map a vehicle name to its manufacturer."""
    if name in CAR_MODEL_TO_BRAND:
        return CAR_MODEL_TO_BRAND[name]
    for prefix, brand in TWO_WHEELER_PREFIXES.items():
        if name.startswith(prefix):
            return brand
    return "Other"

df["Brand"] = df["Car_Name"].apply(get_brand)
print("Unique vehicle names:", df["Car_Name"].nunique())
print("Brand counts:\n", df["Brand"].value_counts(), "\n")

# Brand goodwill = average *depreciation retention* of the brand, learned ONLY from training data
# (computed after the split to avoid target leakage - see section 5).
# Price retention ratio per row = Selling_Price / Present_Price
df["Retention"] = df["Selling_Price"] / df["Present_Price"]

# Log transform of skewed columns
df["Log_Kms"] = np.log1p(df["Driven_kms"])
df["Log_Present_Price"] = np.log1p(df["Present_Price"])

print("Engineered features: Car_Age, Brand, Log_Kms , Log_Present_Price (+ Brand_Goodwill after split)\n")

# ----------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------
print("=" * 70, "\n4. EXPLORATORY DATA ANALYSIS (plots saved to outputs/)\n" + "=" * 70)

# 4.1 Price distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df["Selling_Price"], bins=30, color="#2a6fb0", edgecolor="white")
ax[0].set(title="Selling Price Distribution", xlabel="Selling Price (lakhs)", ylabel="Count")
ax[1].hist(np.log1p(df["Selling_Price"]), bins=30, color="#e07b39", edgecolor="white")
ax[1].set(title="log(1 + Selling Price)", xlabel="log price", ylabel="Count")
plt.tight_layout(); plt.savefig(f"{OUT}/01_price_distribution.png"); plt.close()

# 4.2 Correlation heatmap
num_cols = ["Selling_Price", "Present_Price", "Driven_kms", "Car_Age", "Owner"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=40, ha="right")
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.grid(False); ax.set_title("Correlation Matrix"); plt.colorbar(im)
plt.tight_layout(); plt.savefig(f"{OUT}/02_correlation_heatmap.png"); plt.close()

# 4.3 Key relationships
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(df["Present_Price"], df["Selling_Price"], alpha=0.6, c="#2a6fb0")
ax[0].set(title="Present vs Selling Price", xlabel="Present Price (lakhs)", ylabel="Selling Price (lakhs)")
ax[1].scatter(df["Car_Age"], df["Selling_Price"], alpha=0.6, c="#e07b39")
ax[1].set(title="Car Age vs Selling Price", xlabel="Age (years)", ylabel="Selling Price (lakhs)")
ax[2].scatter(df["Driven_kms"], df["Selling_Price"], alpha=0.6, c="#3a9d5d")
ax[2].set(title="Mileage vs Selling Price", xlabel="Kilometres driven", ylabel="Selling Price (lakhs)")
plt.tight_layout(); plt.savefig(f"{OUT}/03_key_relationships.png"); plt.close()

# 4.4 Categorical effects
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, col in zip(ax, ["Fuel_Type", "Selling_type", "Transmission", "Owner"]):
    df.groupby(col)["Selling_Price"].mean().plot(kind="bar", ax=a, color="#2a6fb0", edgecolor="white")
    a.set(title=f"Avg Price by {col}", ylabel="Avg price (lakhs)", xlabel="")
    a.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(f"{OUT}/04_categorical_effects.png"); plt.close()

print("Correlation of numeric features with Selling_Price:")
print(corr["Selling_Price"].drop("Selling_Price").round(3), "\n")

# ----------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT  (+ leakage-safe 'Brand Goodwill' feature)
# ----------------------------------------------------------------------
print("=" * 70, "\n5. TRAIN / TEST SPLIT\n" + "=" * 70)
TARGET = "Selling_Price"
train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
train_df, test_df = train_df.copy(), test_df.copy()

# Brand goodwill: mean price-retention of each brand, computed on TRAIN ONLY.
# Brands with few samples are shrunk toward the global mean (smoothing) so rare brands don't overfit.
global_ret = train_df["Retention"].mean()
stats = train_df.groupby("Brand")["Retention"].agg(["mean", "count"])
SMOOTH = 5
stats["goodwill"] = (stats["mean"] * stats["count"] + global_ret * SMOOTH) / (stats["count"] + SMOOTH)
goodwill_map = stats["goodwill"].to_dict()
for part in (train_df, test_df):
    part["Brand_Goodwill"] = part["Brand"].map(goodwill_map).fillna(global_ret)

print("Brand goodwill (price-retention score, higher = holds value better):")
print(stats["goodwill"].sort_values(ascending=False).round(3).head(10), "\n")

NUM_FEATURES = ["Log_Present_Price", "Car_Age", "Log_Kms", "Owner", "Brand_Goodwill"]
CAT_FEATURES = ["Fuel_Type", "Selling_type", "Transmission"]
FEATURES = NUM_FEATURES + CAT_FEATURES

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]
print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# ----------------------------------------------------------------------
# 6. PREPROCESSING PIPELINE
# ----------------------------------------------------------------------
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
])

# ----------------------------------------------------------------------
# 7. MODEL TRAINING & COMPARISON (5-fold CV on training set)
# ----------------------------------------------------------------------
print("=" * 70, "\n6. MODEL COMPARISON (5-fold cross-validation)\n" + "=" * 70)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                   max_depth=3, random_state=RANDOM_STATE),
}
def logy(est):
    """Train on log(1+price) and convert predictions back - handles the skewed price target."""
    return TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, mdl in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", logy(mdl))])
    r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    rmse = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    cv_rows.append({"Model": name, "CV R2 (mean)": r2.mean(), "CV R2 (std)": r2.std(),
                    "CV RMSE (mean)": rmse.mean()})
cv_df = pd.DataFrame(cv_rows).sort_values("CV R2 (mean)", ascending=False).reset_index(drop=True)
print(cv_df.round(4).to_string(index=False), "\n")

# ----------------------------------------------------------------------
# 8. HYPERPARAMETER TUNING (top tree-ensemble candidates)
# ----------------------------------------------------------------------
print("=" * 70, "\n7. HYPERPARAMETER TUNING\n" + "=" * 70)
searches = {
    "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                      {"model__regressor__n_estimators": [200, 400],
                       "model__regressor__max_depth": [None, 8, 12],
                       "model__regressor__min_samples_leaf": [1, 2, 4]}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=RANDOM_STATE),
                          {"model__regressor__n_estimators": [200, 400],
                           "model__regressor__learning_rate": [0.03, 0.05, 0.1],
                           "model__regressor__max_depth": [2, 3, 4]}),
}
tuned = {}
for name, (est, grid) in searches.items():
    gs = GridSearchCV(Pipeline([("prep", preprocess), ("model", logy(est))]), grid,
                      cv=kf, scoring="r2", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned[name] = gs.best_estimator_
    print(f"{name}: best CV R2 = {gs.best_score_:.4f} | params = {gs.best_params_}")
print()

# ----------------------------------------------------------------------
# 9. FINAL EVALUATION ON HELD-OUT TEST SET
# ----------------------------------------------------------------------
print("=" * 70, "\n8. FINAL TEST-SET EVALUATION\n" + "=" * 70)
final_candidates = {
    "Linear Regression": Pipeline([("prep", preprocess), ("model", logy(LinearRegression()))]),
    **{f"{k} (tuned)": v for k, v in tuned.items()},
}
rows, preds = [], {}
for name, pipe in final_candidates.items():
    if "tuned" not in name:
        pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    preds[name] = p
    rows.append({"Model": name,
                 "MAE": mean_absolute_error(y_test, p),
                 "RMSE": np.sqrt(mean_squared_error(y_test, p)),
                 "R2": r2_score(y_test, p)})
res_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
print(res_df.round(4).to_string(index=False), "\n")

best_name = res_df.loc[0, "Model"]
best_model = final_candidates[best_name]
best_pred = preds[best_name]
print(f"BEST MODEL: {best_name}  (R2 = {res_df.loc[0,'R2']:.4f}, MAE = {res_df.loc[0,'MAE']:.3f} lakhs)\n")

# ----------------------------------------------------------------------
# 10. DIAGNOSTIC PLOTS
# ----------------------------------------------------------------------
# Actual vs predicted
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_test, best_pred, alpha=0.7, c="#2a6fb0", edgecolor="white")
lims = [0, max(y_test.max(), best_pred.max()) * 1.05]
ax[0].plot(lims, lims, "r--", label="Perfect prediction")
ax[0].set(title=f"Actual vs Predicted - {best_name}", xlabel="Actual price (lakhs)",
          ylabel="Predicted price (lakhs)"); ax[0].legend()
resid = y_test - best_pred
ax[1].scatter(best_pred, resid, alpha=0.7, c="#e07b39", edgecolor="white")
ax[1].axhline(0, color="k", ls="--")
ax[1].set(title="Residual Plot", xlabel="Predicted price (lakhs)", ylabel="Residual (actual - predicted)")
plt.tight_layout(); plt.savefig(f"{OUT}/05_actual_vs_predicted.png"); plt.close()

# Model comparison bar chart
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].barh(cv_df["Model"][::-1], cv_df["CV R2 (mean)"][::-1], color="#2a6fb0")
ax[0].set(title="Cross-validated R2 (training set)", xlabel="R2")
ax[1].barh(res_df["Model"][::-1], res_df["R2"][::-1], color="#3a9d5d")
ax[1].set(title="Test-set R2 (final models)", xlabel="R2")
plt.tight_layout(); plt.savefig(f"{OUT}/06_model_comparison.png"); plt.close()

# Feature importance (permutation importance works for any model)
from sklearn.inspection import permutation_importance
pi = permutation_importance(best_model, X_test, y_test, n_repeats=20,
                            random_state=RANDOM_STATE, scoring="r2")
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(imp.index, imp.values, xerr=pi.importances_std[imp.index.map(FEATURES.index)],
        color="#2a6fb0")
ax.set(title=f"Permutation Feature Importance - {best_name}", xlabel="Drop in R2 when shuffled")
plt.tight_layout(); plt.savefig(f"{OUT}/07_feature_importance.png"); plt.close()
print("Feature importance (permutation, test set):")
print(imp.sort_values(ascending=False).round(4), "\n")

# ----------------------------------------------------------------------
# 11. SAVE ARTEFACTS
# ----------------------------------------------------------------------
cv_df.to_csv(f"{OUT}/cv_results.csv", index=False)
res_df.to_csv(f"{OUT}/test_results.csv", index=False)
pd.DataFrame({"Actual": y_test.values, "Predicted": best_pred.round(3)}).to_csv(
    f"{OUT}/test_predictions.csv", index=False)
joblib.dump({"model": best_model, "goodwill_map": goodwill_map, "global_goodwill": global_ret,
             "features": FEATURES, "reference_year": int(CURRENT_YEAR)},
            f"{OUT}/best_model.joblib")

# ----------------------------------------------------------------------
# 12. EXAMPLE PREDICTION
# ----------------------------------------------------------------------
def predict_price(brand, present_price, year, kms, fuel, seller, transmission, owner=0):
    """Predict selling price (lakhs) for a single car."""
    art = joblib.load(f"{OUT}/best_model.joblib")
    row = pd.DataFrame([{
        "Log_Present_Price": np.log1p(present_price),
        "Car_Age": art["reference_year"] - year,
        "Log_Kms": np.log1p(kms),
        "Owner": owner,
        "Brand_Goodwill": art["goodwill_map"].get(brand, art["global_goodwill"]),
        "Fuel_Type": fuel, "Selling_type": seller, "Transmission": transmission,
    }])[art["features"]]
    return float(art["model"].predict(row)[0])

print("=" * 70, "\n9. EXAMPLE PREDICTION\n" + "=" * 70)
ex = predict_price("Toyota", 9.5, 2015, 40000, "Petrol", "Dealer", "Manual")
print(f"Toyota, 2015, 40,000 km, Petrol, Dealer, Manual, showroom price 9.5 lakhs "
      f"-> predicted selling price: {ex:.2f} lakhs")
