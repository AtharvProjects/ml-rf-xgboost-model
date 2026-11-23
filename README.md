# 🚀 Customer Churn Prediction: Random Forest vs XGBoost

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning pipeline comparing **Random Forest** and **XGBoost (Extreme Gradient Boosting)** on tabular customer financial & churn data with **100+ data records** (260 curated entries).

---

## 📌 Project Overview

This project demonstrates a production-style machine learning workflow for predicting customer churn risk based on account demographics, activity levels, credit ratings, and product engagement.

### 🌟 Key Features
- **100+ Realistic Data Entries**: 260 customer records with realistic financial distributions, correlated churn risks, and diverse feature dimensions.
- **Dual Model Implementation**:
  - **Random Forest Classifier**: Ensemble of bagging decision trees with class balancing.
  - **XGBoost Classifier**: Gradient boosted tree model with weighted log-loss optimization for class imbalance.
- **End-to-End Modular Pipeline**:
  - Automated synthetic data generation (`data/generate_data.py`)
  - Stratified train-test splitting and feature standardization (`src/preprocess.py`)
  - Model training with tuned hyperparameters (`src/train.py`)
  - Cross-model performance comparison with publication-ready charts (`src/evaluate.py`)
  - CLI and programmatic inference engine (`predict.py`)
- **Visual Analytics**: Confusion matrices, comparative ROC-AUC curves, and side-by-side feature importance rankings.

---

## 📊 Dataset Overview

The dataset contains **260 customer profiles** with the following features:

| Feature Name | Type | Description |
|---|---|---|
| `credit_score` | Integer | Credit rating score (350 - 850) |
| `age` | Integer | Customer age (18 - 70) |
| `tenure` | Integer | Number of years as a customer (0 - 10) |
| `balance` | Float | Account balance in USD |
| `num_products` | Integer | Number of bank/platform products utilized (1 - 4) |
| `has_credit_card` | Binary | Whether the customer holds an active credit card (0/1) |
| `is_active_member` | Binary | Customer engagement activity status (0/1) |
| `estimated_salary` | Float | Estimated annual salary in USD |
| `satisfaction_score` | Integer | Customer satisfaction rating (1 - 5) |
| **`churn`** *(Target)* | Binary | Whether customer churned (1) or remained (0) |

---

## 🏗️ Project Architecture

```
ml-rf-xgboost-model/
├── data/
│   ├── generate_data.py            # Generates reproducible dataset
│   └── customer_churn_data.csv     # 260 customer records (CSV)
├── models/
│   ├── random_forest_model.joblib  # Serialized Random Forest model
│   ├── xgboost_model.joblib        # Serialized XGBoost model
│   └── scaler.joblib               # StandardScaler transformation model
├── plots/
│   ├── confusion_matrices.png      # Side-by-side confusion matrix heatmaps
│   ├── roc_curve_comparison.png    # ROC curves for RF vs XGBoost
│   └── feature_importance_comparison.png # Relative feature importance chart
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # Data loading, train-test splitting, scaling
│   ├── train.py                    # Model training & serialization
│   └── evaluate.py                 # Performance evaluation & plot generation
├── predict.py                      # Interactive CLI inference script
├── run_pipeline.sh                 # One-click execution shell script
├── metrics.json                    # Benchmark metrics output
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/AtharvProjects/ml-rf-xgboost-model.git
cd ml-rf-xgboost-model
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Complete End-to-End Pipeline
You can run the entire workflow (data generation, model training, evaluation, and test prediction) using:
```bash
./run_pipeline.sh
```

Or run individual components:
```bash
# 1. Generate data
python data/generate_data.py

# 2. Train models
python src/train.py

# 3. Evaluate and generate charts
python src/evaluate.py
```

---

## 🔮 Running Predictions (Inference)

Run inference for any custom customer profile with `predict.py`:

```bash
python predict.py \
  --credit_score 620 \
  --age 48 \
  --tenure 2 \
  --balance 92000 \
  --num_products 1 \
  --has_credit_card 1 \
  --is_active_member 0 \
  --estimated_salary 78000 \
  --satisfaction_score 2
```

**Sample Output:**
```
--- Input Customer Profile ---
  credit_score        : 620
  age                 : 48
  tenure              : 2
  balance             : 92000.0
  num_products        : 1
  has_credit_card     : 1
  is_active_member    : 0
  estimated_salary    : 78000.0
  satisfaction_score  : 2

--- Model Predictions ---
[Random Forest]
  Prediction       : Churn (1)
  Churn Probability: 84.6%

[XGBoost]
  Prediction       : Churn (1)
  Churn Probability: 91.2%
```

---

## 📈 Model Performance & Results

Evaluation conducted on a held-out test split (stratified 20%):

| Metric | Random Forest | XGBoost |
|---|---|---|
| **Accuracy** | 86.54% | 82.69% |
| **Precision** | 50.00% | 37.50% |
| **Recall** | 71.43% | 42.86% |
| **F1-Score** | 0.5882 | 0.4000 |
| **ROC-AUC** | 0.7079 | 0.6921 |

Visualizations generated during evaluation:
- `plots/confusion_matrices.png`: Confusion matrices comparing true vs predicted positive/negative churn.
- `plots/roc_curve_comparison.png`: Receiver Operating Characteristic (ROC) curves with AUC metrics.
- `plots/feature_importance_comparison.png`: Relative feature contribution ranking for both algorithms.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
