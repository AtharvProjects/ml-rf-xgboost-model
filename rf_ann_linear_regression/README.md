# 🏡 Property Valuation: Linear Regression vs. Random Forest vs. ANN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning regression benchmark comparing three foundational paradigms:
1. **Parametric Linear Modeling**: Ordinary Least Squares (OLS) Linear Regression
2. **Ensemble Tree Learning**: Random Forest Regressor (Bagging)
3. **Deep Learning / Neural Networks**: Artificial Neural Network (ANN / Multi-Layer Perceptron)

The models are trained and benchmarked on a curated housing valuation dataset containing **250 entries** (>100 records) with realistic multi-variable distributions and non-linear interactions.

---

## 📌 Project Overview

This module provides a production-grade, modular pipeline for continuous valuation forecasting. It features automated synthetic generation, feature standardization, cross-paradigm model training, exhaustive performance evaluation ($R^2$, RMSE, MAE, MAPE), analytical plotting, and interactive CLI inference.

### 🌟 Key Highlights
- **100+ Realistic Data Entries**: 250 synthesized property records with realistic real estate dynamics (depreciation, neighborhood multiplier, square footage economics, luxury amenities synergy).
- **Three Core ML Paradigms**:
  - **Linear Regression**: Direct parametric weights capturing global linear trends.
  - **Random Forest**: 150 bagging estimators capturing non-linear thresholds and complex feature splits without overfitting.
  - **Artificial Neural Network (ANN)**: 3-layer deep MLP `(128 -> 64 -> 32)` with ReLU activations, Adam optimization, adaptive learning rate, and early stopping.
- **Publication-Ready Visual Analytics**:
  - Parity plots (Actual vs. Predicted with ideal 1:1 line)
  - Residual error distribution and normality checks (KDE & histograms)
  - Benchmark metric comparisons across models ($R^2$, RMSE, MAE, MAPE)
  - Tree feature importance (MDI) vs. linear coefficients
  - ANN training loss convergence curve across training epochs
- **Interactive Multi-Model Inference CLI**: Predict property value using all 3 algorithms and an automated consensus ensemble.

---

## 📊 Dataset Schema (250 Records)

| Feature Name | Type | Unit / Range | Description |
|---|---|---|---|
| `square_footage` | Integer | 900 - 4,800 sq ft | Total living interior area |
| `bedrooms` | Integer | 1 - 6 | Total count of bedrooms |
| `bathrooms` | Float | 1.0 - 5.0 | Total count of bathrooms |
| `lot_size_sqft` | Integer | 1,200 - 15,000 sq ft | Property parcel/lot land size |
| `property_age_years` | Integer | 0 - 49 years | Age of the property structure |
| `distance_to_center_km` | Float | 1.5 - 28.0 km | Distance to primary metropolitan city center |
| `neighborhood_quality_score` | Float | 1.0 - 10.0 | Composite index of school ratings, safety, and amenities |
| `has_garage` | Binary | 0 or 1 | Dedicated garage availability |
| `has_pool` | Binary | 0 or 1 | Private swimming pool amenity |
| `annual_tax` | Float | $1,800 - $12,800 | Annual municipal property taxes |
| **`price`** *(Target)* | Float | $180,000 - $1,390,000 | Property market valuation in USD |

---

## 🏗️ Folder Architecture

```
rf_ann_linear_regression/
├── data/
│   ├── generate_data.py               # Generates 250 curated records
│   └── housing_prices_data.csv        # 250 data entries CSV
├── models/
│   ├── linear_regression_model.joblib # Serialized Linear Regression model
│   ├── random_forest_regressor.joblib # Serialized Random Forest model
│   ├── ann_mlp_regressor.joblib       # Serialized Artificial Neural Network model
│   └── scaler.joblib                  # Fitted StandardScaler transformation
├── plots/
│   ├── actual_vs_predicted.png         # Parity scatter comparison with 1:1 line
│   ├── residuals_distribution.png     # Residual error distributions
│   ├── model_metrics_comparison.png   # Bar chart comparing R², RMSE, MAE, MAPE
│   ├── feature_importance_comparison.png # RF importance vs LR coefficients
│   └── ann_loss_curve.png             # ANN epoch loss progression
├── src/
│   ├── __init__.py
│   ├── preprocess.py                  # Train/test splitting (80/20) & scaling
│   ├── train.py                       # Fits & serializes all 3 models
│   └── evaluate.py                    # Evaluates metrics & generates visual plots
├── predict.py                         # Multi-model interactive CLI inference
├── run_pipeline.sh                    # One-click execution shell script
├── metrics.json                       # Serialized benchmark test metrics
└── README.md                          # Comprehensive documentation
```

---

## 🚀 Quickstart Guide

### 1. Run Complete Pipeline
Execute the complete end-to-end workflow (generation, training, evaluation, and test prediction):

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### 2. Run Modular Steps Individually

```bash
# Step 1: Generate 100+ dataset entries
python3 data/generate_data.py

# Step 2: Train Linear Regression, Random Forest, and ANN
python3 src/train.py

# Step 3: Run comprehensive evaluation and generate plots
python3 src/evaluate.py
```

---

## 📈 Model Performance Benchmark (Held-out Test Set)

Evaluated on a 20% holdout split ($N = 50$ test instances):

| Metric | Linear Regression | Random Forest | Artificial Neural Network (ANN) |
|---|:---:|:---:|:---:|
| **$R^2$ Score** | **0.9784** | 0.9147 | **0.9823** 🏆 |
| **RMSE ($ USD)** | $22,906.11 | $45,457.78 | **$20,713.07** 🏆 |
| **MAE ($ USD)** | $18,616.35 | $35,472.50 | **$16,053.29** 🏆 |
| **MAPE (%)** | 2.75% | 5.17% | **2.46%** 🏆 |

### 🔍 Analytical Insights
- **Artificial Neural Network (ANN)** achieved the highest overall accuracy ($R^2 = 0.9823$, MAPE = $2.46\%$), effectively capturing non-linear amenity synergies and distance-decay curvatures.
- **Linear Regression** served as an exceptionally strong baseline ($R^2 = 0.9784$), demonstrating that primary square footage and tax features have strong underlying linear components.
- **Random Forest** achieved strong performance ($R^2 = 0.9147$) without requiring feature normalization, isolating clear hierarchical decision thresholds.

---

## 🔮 Interactive Prediction (CLI)

Run inference for custom property specifications using `predict.py`:

```bash
python3 predict.py \
  --square_footage 2600 \
  --bedrooms 4 \
  --bathrooms 3.5 \
  --lot_size_sqft 7200 \
  --property_age_years 8 \
  --distance_to_center_km 6.5 \
  --neighborhood_quality_score 8.8 \
  --has_garage 1 \
  --has_pool 1 \
  --annual_tax 8900.0
```

**Output:**
```
=================================================================
🏡 PROPERTY PROFILE INPUT SPECIFICATIONS
=================================================================
  square_footage              : 2,600
  bedrooms                    : 4
  bathrooms                   : 3.50
  lot_size_sqft               : 7,200
  property_age_years          : 8
  distance_to_center_km       : 6.50
  neighborhood_quality_score  : 8.80
  has_garage                  : 1
  has_pool                    : 1
  annual_tax                  : 8,900.00

=================================================================
💰 MODEL VALUATION PREDICTIONS
=================================================================
  [1] Linear Regression (OLS)        : $957,897.88
  [2] Random Forest Regressor        : $917,334.55
  [3] Artificial Neural Network (ANN): $970,228.16
-----------------------------------------------------------------
  ★ Consensus Ensemble (Average)     : $948,486.86
=================================================================
```

---

## 📜 License
This module is open-source and released under the [MIT License](LICENSE).
