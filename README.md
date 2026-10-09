# Car Price Prediction with Machine Learning

**Data Science Internship - Machine Learning Project**

**Intern:** Qadir Hussain
**Company:** Code Alpha

---

## Project Overview

The Car Price Prediction project is a Machine Learning application developed in Python using Scikit-learn. It predicts the selling price (in lakhs) of a used vehicle from features such as present (showroom) price, age, kilometres driven, fuel type, seller type, transmission, number of previous owners and brand goodwill.

The project loads and cleans `car_data.csv` using Pandas, engineers new features, compares several regression models using cross-validation, tunes the best candidates, and evaluates the final model on unseen test data using MAE, RMSE and R². All plots, result tables and the trained model are saved automatically.

This project demonstrates data preprocessing, feature engineering, model comparison, hyperparameter tuning, model evaluation and the real-world use of machine learning in price prediction.

## Features

* Load and inspect `car_data.csv` using Pandas
* Remove missing values and duplicate rows (2 duplicates removed)
* Extract the brand (manufacturer) from the vehicle name
* Create `Car_Age` from the manufacturing year
* Create a leakage-safe `Brand_Goodwill` score (average price retention per brand, learned from training data only)
* Log-transform skewed features (`Driven_kms`, `Present_Price`) and the target price
* Exploratory data analysis with 4 saved plots
* Split data into training (80%) and testing (20%) sets
* Preprocess with `ColumnTransformer` (StandardScaler + OneHotEncoder) inside a `Pipeline`
* Compare 5 regression models using 5-fold cross-validation
* Tune Random Forest and Gradient Boosting using `GridSearchCV`
* Evaluate with MAE, RMSE and R²
* Generate actual vs predicted, residual and feature importance plots
* Save the best model (`best_model.joblib`) and a reusable `predict_price()` function

## Bonus Features

* Log-transformed target for better handling of skewed prices
* Permutation feature importance, which works with any model
* Leakage-safe feature engineering (brand goodwill is calculated after the split)
* Cross-validated model comparison table saved as CSV
* Test-set predictions saved as CSV

## Technologies Used

* Python 3.10+
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

## Project Structure

```
car-price-prediction-ml/
│
├── car_price_prediction.py
├── requirements.txt
├── README.md
│
├── data/
│   └── car_data.csv
│
└── outputs/
    ├── 01_price_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_key_relationships.png
    ├── 04_categorical_effects.png
    ├── 05_actual_vs_predicted.png
    ├── 06_model_comparison.png
    ├── 07_feature_importance.png
    ├── cv_results.csv
    ├── test_results.csv
    ├── test_predictions.csv
    └── best_model.joblib
```

## Installation

### 1. Clone Repository

```
git clone https://github.com/qadirhussain222786k-dev/CodeAlpha_Project-2_car-price-prediction-ml.git
```

### 2. Open Project

```
cd CodeAlpha_Project-2_car-price-prediction-ml
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

The dataset (`data/car_data.csv`) is already included.

## Running the Project

```
python car_price_prediction.py
```

or

```
py car_price_prediction.py
```

The script prints each step in the terminal and saves all plots and results in the `outputs/` folder.

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
joblib
```

## Dataset

| Column | Description |
|---|---|
| Car_Name | Name of the vehicle |
| Year | Year of manufacture |
| Selling_Price | Target: price at which it is sold (lakhs) |
| Present_Price | Current showroom price (lakhs) |
| Driven_kms | Kilometres driven |
| Fuel_Type | Petrol / Diesel / CNG |
| Selling_type | Dealer / Individual |
| Transmission | Manual / Automatic |
| Owner | Number of previous owners |

## Results

| Model | CV R² (5-fold) | Test MAE (lakhs) | Test RMSE | Test R² |
|---|---|---|---|---|
| **Linear Regression (log target)** | **0.955** | **0.95** | 1.70 | **0.888** |
| Gradient Boosting (tuned) | 0.904 | 0.90 | 1.95 | 0.853 |
| Random Forest (tuned) | 0.897 | 1.33 | 3.04 | 0.642 |

**Best model:** Linear Regression with a log-transformed target and log-transformed present price. It was the most stable in cross-validation and extrapolates better to expensive vehicles than tree-based models.

**Key findings**

* Present (showroom) price is the strongest predictor, followed by vehicle age.
* Fuel type has a smaller effect; mileage, owner count and transmission contribute little.
* Brand goodwill adds almost nothing once present price and age are known.

## Sample Prediction

```python
predict_price("Toyota", present_price=9.5, year=2015, kms=40000,
              fuel="Petrol", seller="Dealer", transmission="Manual")
```

```
Predicted selling price: 5.74 lakhs
```

## Limitations

* Small dataset (299 rows), so test metrics vary with the train/test split.
* The data contains both cars and motorcycles.
* One very expensive vehicle is under-predicted because such cars are rare in the data.

## Real-World Applications

* Used-car marketplaces and instant valuation tools
* Dealer trade-in pricing
* Insurance and resale valuation
* Leasing and loan residual-value estimation
* Dynamic pricing in e-commerce, real estate and airlines

## Concepts Covered

* Python Fundamentals
* Functions
* File Handling (CSV)
* Data Preprocessing with Pandas
* Feature Engineering
* Train/Test Split
* Feature Scaling and One-Hot Encoding
* Regression (Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting)
* Cross-Validation
* Hyperparameter Tuning (GridSearchCV)
* Model Evaluation (MAE, RMSE, R²)
* Data Visualization with Matplotlib
* Third-Party Packages (Pandas, NumPy, Scikit-learn, Matplotlib)

## Author

**Name:** Qadir Hussain
**University:** Capital University of Science and Technology (CUST)
**Program:** BS Software Engineering
