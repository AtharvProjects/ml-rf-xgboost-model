# 🚀 Machine Learning Suite: Classification & Regression Benchmarks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning repository featuring production-ready implementations of foundational classification and regression paradigms on curated datasets with **100+ data entries**.

---

## 🗂️ Modules in this Repository

| Module | Task Type | Models Implemented | Dataset | Records |
|---|---|---|---|:---:|
| **[Customer Churn Prediction](./)** *(Root)* | **Binary Classification** | Random Forest vs. XGBoost | Customer Financial & Churn Data | 260 Entries |
| **[Property Valuation Benchmark](./rf_ann_linear_regression/)** | **Continuous Regression** | Linear Regression vs. Random Forest vs. ANN | Real Estate & Valuation Data | 250 Entries |
| **[100-Entry Benchmark & Formula Derivation](./benchmark_100_samples/)** | **9-Input Regression** | Linear Reg, Ridge, GBM, RF, ANN (70/30 split) | Power Plant Net Output Optimization | 100 Entries |
| **[NSM FRP Bond Strength & Failure Mode](./nsm_frp_bond_strength/)** | **Regression + Multi-Class + GUI** | XGBoost, RF, GBT, DT, ANN, LR + 5 Design Codes | NSM FRP-to-Concrete Experimental DB | 416 Entries |

---

## 📂 Module 1: Customer Churn Classification (Random Forest vs. XGBoost)

This module demonstrates a production-style machine learning workflow for predicting customer churn risk based on account demographics, activity levels, credit ratings, and product engagement.

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

## 📂 Module 2: Property Valuation Continuous Regression (LR vs. RF vs. ANN)

Located in [`rf_ann_linear_regression/`](./rf_ann_linear_regression/):
- **Models**: Linear Regression (OLS), Random Forest Regressor, and Artificial Neural Network (MLP: 128-64-32).
- **Data**: 250 curated property records (`rf_ann_linear_regression/data/housing_prices_data.csv`).
- **Metrics**: $R^2$ Score (ANN: **0.9823**, LR: **0.9784**, RF: **0.9147**), RMSE, MAE, MAPE.
- **Visuals**: Parity plots, residual distributions, metric comparisons, ANN loss curves, and feature importance.
- See the dedicated [rf_ann_linear_regression README](./rf_ann_linear_regression/README.md) for detailed documentation.

---

## 📂 Module 3: 100-Entry Benchmark, Formula Derivation & Custom Testing

Located in [`benchmark_100_samples/`](./benchmark_100_samples/):

### 📌 Problem Formulation
- **Dataset**: Exactly 100 data entries with **10 parameters** per entry (9 inputs $X_1 \dots X_9$, 1 output $Y$).
- **Partitioning**: Exact **70 entries for training** and **30 entries for validation**.
- **Model Comparison**: 5 diverse algorithms trained under the identical strategy.

### 📊 Benchmark Results (30 Validation Entries)

| Rank | Model Architecture | $R^2$ Score (Closest to 1.0) | RMSE (MW) | MAE (MW) | MSE |
|:---:|---|:---:|:---:|:---:|:---:|
| 🥇 | **Multiple Linear Regression (OLS)** | **0.999695** 🏆 | **0.8041** | **0.6589** | **0.6466** |
| 🥈 | **Ridge Regression ($L_2$)** | **0.998560** | 1.7467 | 1.3719 | 3.0509 |
| 🥉 | **Gradient Boosting Regressor** | **0.828397** | 19.0707 | 16.8452 | 363.6904 |
| 4 | **Random Forest Regressor** | **0.754297** | 22.8196 | 19.2857 | 520.7358 |
| 5 | **Artificial Neural Network (MLP)** | **-5.097646** | 113.6800 | 98.7559 | 12,923.15 |

### 📐 Derived Closed-Form Mathematical Formula (Best Model: OLS)

$$\boxed{\hat{Y} = 419.8034 - 1.8575 X_1 + 42.2294 X_2 - 0.1166 X_3 - 0.3831 X_4 + 4.2214 X_5 + 6.1404 X_6 + 2.7977 X_7 - 1.4406 X_8 + 87.9806 X_9}$$

---

### 🧪 Step-by-Step: How to Test Against Custom Entries

#### Step 1: Prepare Your 9 Inputs
Ensure your custom values correspond to the 9 thermodynamic parameters:
- `X1` (Ambient Temp): $10 - 35$ °C
- `X2` (Ambient Press): $0.98 - 1.04$ bar
- `X3` (Humidity): $30 - 90$ %
- `X4` (Exhaust Vac): $25 - 75$ cm Hg
- `X5` (Turbine Press): $12 - 20$ bar
- `X6` (Fuel Flow): $8 - 25$ kg/s
- `X7` (Steam Flow): $15 - 50$ kg/s
- `X8` (Cooling Temp): $12 - 30$ °C
- `X9` (Gen Efficiency): $0.92 - 0.98$

#### Step 2: Choose Your Testing Method

**1. CLI Single Entry Test:**
```bash
cd benchmark_100_samples
python3 predict_custom.py \
  --x1 25.0 --x2 1.015 --x3 60.0 --x4 45.0 --x5 16.5 \
  --x6 18.0 --x7 32.0 --x8 21.0 --x9 0.955 --actual_y 725.50
```

**2. CSV Batch Testing:**
```bash
cd benchmark_100_samples
python3 predict_custom.py --csv data/sample_custom_inputs.csv --output data/predictions_output.csv
```

**3. Direct Python Code:**
```python
def predict_power(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return (
        419.8034 - 1.8575*x1 + 42.2294*x2 - 0.1166*x3 - 0.3831*x4
        + 4.2214*x5 + 6.1404*x6 + 2.7977*x7 - 1.4406*x8 + 87.9806*x9
    )
print(predict_power(25.0, 1.015, 60.0, 45.0, 16.5, 18.0, 32.0, 21.0, 0.955))
```

**4. Excel / Google Sheets Formula:**
In cell `J2` (where `A2:I2` are inputs $X_1 \dots X_9$):
```excel
= 419.8034 - (1.8575*A2) + (42.2294*B2) - (0.1166*C2) - (0.3831*D2) + (4.2214*E2) + (6.1404*F2) + (2.7977*G2) - (1.4406*H2) + (87.9806*I2)
```

---

## 📂 Module 4: NSM FRP Bond Strength & Failure Mode (Kumar et al., 2024 Replication)

Located in [`nsm_frp_bond_strength/`](./nsm_frp_bond_strength/):
- **Datasheet**: 416 NSM FRP-to-concrete pullout records with **9 input parameters** ($f'_c, L_b, D_f, E_f, f_f, W_g, D_g, FRP_{BT}, FRP_{ST}$) and **2 outputs** (Bond strength $P_u$ in kN & Failure Mode $F_N$ 1–16), split **70% Training (291) / 30% Testing (125)**.
- **Models**: 6 ML models (`XGBoost`, `GBT`, `RF`, `ANN`, `DT`, `LR`) compared against 5 structural design guidelines & analytical equations (`ACI 440.2R-08`, `SA HB 305`, `ACI 440.1R-15`, `Zhang et al.`, `CSA-S806`).
- **Best Performance (XGBoost)**: $R = \mathbf{0.9945}$, $R^2 = \mathbf{0.9891}$, $\text{RMSE} = \mathbf{2.41\text{ kN}}$, $\text{SI} = \mathbf{0.0578}$.
- **Diagrams Generated**: Donut variable distributions (`Fig. 1–2`), 3-panel Scatter / Error Frequency / Performance Ratio diagnostics (`Fig. 4–13`), Radar comparisons (`Fig. 14`), Taylor plots (`Fig. 15`), Failure mode scatter (`Fig. 16`), TreeSHAP explainability (`Fig. 17`), Partial Dependence & ICE curves (`Fig. 18`), and NSM specimen schematic (`Fig. 19`).
- **Interactive Web GUI & CLI**: Launch `streamlit run app.py` or test custom specimens via `python3 predict_custom.py`. See [nsm_frp_bond_strength/README.md](./nsm_frp_bond_strength/README.md) for complete tables and equations.

---

## 🏗️ Complete Repository Tree

```
ml-rf-xgboost-model/
├── README.md                              # Main overview linking all 3 modules
├── requirements.txt                       # Shared dependencies
├── run_pipeline.sh                        # Module 1 root runner
├── predict.py                             # Module 1 churn inference CLI
├── data/                                  # Module 1 Churn data (260 entries)
├── models/                                # Module 1 saved joblib models
├── plots/                                 # Module 1 confusion matrix & ROC charts
│
├── rf_ann_linear_regression/              # Module 2: Property Valuation Regression
│   ├── README.md                          # Dedicated Module 2 docs
│   ├── run_pipeline.sh                    # One-click Module 2 runner
│   ├── predict.py                         # Property valuation inference CLI
│   ├── data/housing_prices_data.csv       # 250 property valuation records
│   ├── models/                            # Trained LR, RF, and ANN models
│   └── plots/                             # Parity, residual, and metric charts
│
└── benchmark_100_samples/                 # Module 3: 100-Entry Benchmark & Formula
    ├── README.md                          # Dedicated Module 3 docs & all steps
    ├── models_benchmark.py                # 70/30 split & 5-model evaluation
    ├── predict_custom.py                  # Custom data testing engine (CLI & batch)
    ├── data/dataset_100_entries.csv       # 100 entries with 10 parameters
    ├── data/sample_custom_inputs.csv      # Batch test custom samples
    └── plots/model_comparison_benchmark.png # R² & parity comparison charts

└── nsm_frp_bond_strength/                 # Module 4: Kumar et al. (2024) NSM FRP Study
    ├── README.md                          # Complete metrics, formulas & diagram guide
    ├── run_pipeline.sh                    # One-click runner for all steps & plots
    ├── app.py                             # Streamlit Web GUI (Fig. 19 replica)
    ├── predict_custom.py                  # Single & CSV batch CLI prediction tool
    ├── data/nsm_frp_bond_dataset.csv      # 416 NSM FRP records (9 inputs, Pu & FN)
    ├── models/                            # 6 ML models + scaler + derived formulas
    └── plots/                             # Figs. 1, 2, 4-19 (SHAP, PDP, Taylor, Radar)
```

---

## 🚀 Environment Setup

```bash
git clone https://github.com/AtharvProjects/ml-rf-xgboost-model.git
cd ml-rf-xgboost-model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📜 License
This repository is open-source and released under the [MIT License](LICENSE).
