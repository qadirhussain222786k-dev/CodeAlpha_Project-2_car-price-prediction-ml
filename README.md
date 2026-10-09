# Task 3: Car Price Prediction with Machine Learning

Predicts the selling price (in lakhs) of used vehicles using regression models built with
**Pandas, Scikit-learn and Matplotlib**.

## Project Structure
```
car_price_prediction/
├── car_price_prediction.py   # full workflow (run this)
├── requirements.txt
├── data/car_data.csv         # dataset (301 rows, 9 columns)
└── outputs/                  # plots, result tables, saved model
```

## How to Run
```bash
pip install -r requirements.txt
python car_price_prediction.py
```

## Workflow
1. **Data loading and inspection** - shape, types, summary statistics.
2. **Cleaning** - no missing values; 2 duplicate rows removed (299 remain).
3. **Feature engineering**
   - `Car_Age` = reference year (2019) - manufacturing year
   - `Brand` extracted from the vehicle name (14 manufacturers)
   - `Brand_Goodwill` = smoothed average price-retention (Selling / Present price) per brand,
     computed on the **training set only** to avoid target leakage
   - `Log_Kms`, `Log_Present_Price` to reduce skew
4. **EDA** - price distribution, correlation matrix, feature-vs-price scatter plots, categorical effects.
5. **Preprocessing** - `ColumnTransformer` with StandardScaler (numeric) and OneHotEncoder (categorical),
   wrapped in a `Pipeline` so no information leaks from the test set.
6. **Models compared** (5-fold CV): Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting.
   Target is modelled as log(1 + price) to handle the skewed price distribution.
7. **Hyperparameter tuning** - GridSearchCV for Random Forest and Gradient Boosting.
8. **Evaluation** - MAE, RMSE and R2 on a held-out 20% test set; residual and feature-importance plots.

## Results
| Model | CV R2 (train, 5-fold) | Test MAE (lakhs) | Test RMSE | Test R2 |
|---|---|---|---|---|
| **Linear Regression (log target)** | **0.955** | **0.95** | 1.70 | **0.888** |
| Gradient Boosting (tuned) | 0.904 | 0.90 | 1.95 | 0.853 |
| Random Forest (tuned) | 0.897 | 1.33 | 3.04 | 0.642 |

**Best model: Linear Regression with log-transformed target and log-transformed present price.**
It is the most stable in cross-validation and extrapolates better than tree models to expensive
vehicles, which are rare in this small dataset.

**Key findings**
- Present (showroom) price is by far the strongest predictor, followed by vehicle age.
- Fuel type has a smaller effect; mileage, owner count and transmission contribute little.
- Brand goodwill adds almost nothing once present price and age are known, because the price
  already encodes brand positioning.

## Limitations
- Small dataset (299 rows); the test set has only 60 rows, so test metrics vary with the split.
- The data mixes cars and motorcycles; one very expensive vehicle (about 33 lakhs) is under-predicted.
- Tree models cannot predict above the prices seen in training, which explains their weaker test scores.

## Real-World Applications of ML in Price Prediction
Used-car marketplaces (instant valuation tools), dealer trade-in pricing, insurance claim and
resale valuation, loan and leasing residual-value estimation, and dynamic pricing in e-commerce,
real estate and airlines.

## Output Files (`outputs/`)
`01_price_distribution.png`, `02_correlation_heatmap.png`, `03_key_relationships.png`,
`04_categorical_effects.png`, `05_actual_vs_predicted.png`, `06_model_comparison.png`,
`07_feature_importance.png`, `cv_results.csv`, `test_results.csv`, `test_predictions.csv`,
`best_model.joblib`
