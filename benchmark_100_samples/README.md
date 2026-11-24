# ⚡ 100-Entry Regression Benchmark: 9 Inputs, 1 Output

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning benchmark on a **100-entry engineering dataset** with **10 parameters** (9 inputs, 1 output), trained with a **70/30 train-validation split**, evaluated across **5 distinct ML architectures**, with mathematical formula derivation and step-by-step custom testing tools.

---

## 📌 Complete Workflow: All Steps

```mermaid
flowchart LR
    A["Step 1: Dataset (100 Entries, 9 Inputs + 1 Output)"] --> B["Step 2: 70/30 Train-Validation Split"]
    B --> C["Step 3: Train 5 Diverse ML Models"]
    C --> D["Step 4: Evaluate R², RMSE, MAE, MSE"]
    D --> E["Step 5: Select Best Model (R² ≈ 1.0)"]
    E --> F["Step 6: Derive Mathematical Formula"]
    F --> G["Step 7: Test Against Custom Entries"]
```

---

## 📊 Step 1: Dataset Schema (100 Entries, 10 Parameters)

Dataset file: [`data/dataset_100_entries.csv`](./data/dataset_100_entries.csv)

| Parameter | Symbol | Type | Typical Range | Engineering Description |
|---|:---:|:---:|:---:|---|
| Ambient Temperature | $X_1$ | Input | $10.0 - 35.0$ °C | Ambient atmospheric intake air temperature |
| Ambient Pressure | $X_2$ | Input | $0.98 - 1.04$ bar | Atmospheric barometric air pressure |
| Relative Humidity | $X_3$ | Input | $30.0 - 90.0$ % | Ambient relative moisture humidity |
| Exhaust Vacuum | $X_4$ | Input | $25.0 - 75.0$ cm Hg | Condenser exhaust backpressure vacuum |
| Turbine Inlet Pressure | $X_5$ | Input | $12.0 - 20.0$ bar | High-pressure steam turbine inlet pressure |
| Fuel Flow Rate | $X_6$ | Input | $8.0 - 25.0$ kg/s | Primary boiler/combustor fuel mass flow |
| Steam Flow Rate | $X_7$ | Input | $15.0 - 50.0$ kg/s | Generated main steam flow rate |
| Cooling Water Temp | $X_8$ | Input | $12.0 - 30.0$ °C | Heat exchanger cooling water inlet temperature |
| Generator Efficiency | $X_9$ | Input | $0.92 - 0.98$ | Electro-mechanical generation efficiency ratio |
| **Net Power Output** | **$Y$** | **Output** | **$480.0 - 820.0$ MW** | **Target net electrical energy output** |

---

## ✂️ Step 2: 70/30 Train-Validation Splitting

- **Training Records**: 70 entries (70% of dataset, random state `42`)
- **Validation Records**: 30 entries (held-out test split, evaluated strictly after fitting)

---

## 🧠 Step 3 & 4: Model Training and Cross-Evaluation Benchmark

Five diverse machine learning algorithms were trained on the exact same 70 records and benchmarked on the 30 validation records:

| Rank | Model Architecture | $R^2$ Score (Closest to 1.0) | RMSE (MW) | MAE (MW) | MSE | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Multiple Linear Regression (OLS)** | **0.999695** 🏆 | **0.8041** | **0.6589** | **0.6466** | **Selected Model** |
| 🥈 | **Ridge Regression ($L_2$ Regularized)** | **0.998560** | 1.7467 | 1.3719 | 3.0509 | Strong Linear Regularizer |
| 🥉 | **Gradient Boosting Regressor (GBM)** | **0.828397** | 19.0707 | 16.8452 | 363.6904 | Sub-optimal ($N=70$) |
| 4 | **Random Forest Regressor** | **0.754297** | 22.8196 | 19.2857 | 520.7358 | Step-function Quantization |
| 5 | **Artificial Neural Network (MLP)** | **-5.097646** | 113.6800 | 98.7559 | 12,923.15 | Overfitting ($N=70$) |

### Analytical Insights:
- **Multiple Linear Regression (OLS)** captured the true physical continuous dynamics best, delivering an almost perfect correlation ($R^2 = 0.999695$) and sub-megawatt error (MAE = $0.66$ MW).
- Tree ensembles suffered from step-wise discretization errors on continuous variables given only 70 samples.
- The multi-layer perceptron neural network lacked the sample volume needed to tune its internal weight matrices.

---

## 🏆 Step 5 & 6: Selected Best Model & Formula Derivation

Since **Multiple Linear Regression** achieved the $R^2$ closest to 1.0, we derived its exact closed-form mathematical expression:

$$\hat{Y} = \beta_0 + \sum_{i=1}^9 \beta_i X_i$$

### Fitted Parameter Values:
- $\beta_0$ (Intercept) = $+419.803448$
- $\beta_1$ ($X_1$, Ambient Temp) = $-1.857517$
- $\beta_2$ ($X_2$, Ambient Pressure) = $+42.229437$
- $\beta_3$ ($X_3$, Relative Humidity) = $-0.116604$
- $\beta_4$ ($X_4$, Exhaust Vacuum) = $-0.383138$
- $\beta_5$ ($X_5$, Turbine Pressure) = $+4.221380$
- $\beta_6$ ($X_6$, Fuel Flow Rate) = $+6.140368$
- $\beta_7$ ($X_7$, Steam Flow Rate) = $+2.797738$
- $\beta_8$ ($X_8$, Cooling Water Temp) = $-1.440591$
- $\beta_9$ ($X_9$, Generator Efficiency) = $+87.980594$

### 📐 Final Predictive Closed-Form Formula:

$$\boxed{\hat{Y} = 419.8034 - 1.8575 X_1 + 42.2294 X_2 - 0.1166 X_3 - 0.3831 X_4 + 4.2214 X_5 + 6.1404 X_6 + 2.7977 X_7 - 1.4406 X_8 + 87.9806 X_9}$$

---

## 🧪 Step 7: Step-by-Step Guide: How to Test Against Custom Data Entries

Follow these steps to evaluate any custom entry (single sample or batch) against the trained model and derived formula.

### Step 7.1: Identify Your 9 Input Parameters
Gather your custom values within their expected physical ranges:
- $X_1$ (`X1_AmbTemp`): $10.0$ to $35.0$ °C
- $X_2$ (`X2_AmbPress`): $0.98$ to $1.04$ bar
- $X_3$ (`X3_Humidity`): $30.0$ to $90.0$ %
- $X_4$ (`X4_ExhaustVac`): $25.0$ to $75.0$ cm Hg
- $X_5$ (`X5_TurbinePress`): $12.0$ to $20.0$ bar
- $X_6$ (`X6_FuelFlow`): $8.0$ to $25.0$ kg/s
- $X_7$ (`X7_SteamFlow`): $15.0$ to $50.0$ kg/s
- $X_8$ (`X8_CoolingTemp`): $12.0$ to $30.0$ °C
- $X_9$ (`X9_GenEfficiency`): $0.92$ to $0.98$

---

### Step 7.2: Choose Your Testing Method

#### 🔹 Method A: Interactive Single Entry via Terminal (CLI)
Run `predict_custom.py` with custom flags:
```bash
cd benchmark_100_samples

python3 predict_custom.py \
  --x1 25.0 \
  --x2 1.015 \
  --x3 60.0 \
  --x4 45.0 \
  --x5 16.5 \
  --x6 18.0 \
  --x7 32.0 \
  --x8 21.0 \
  --x9 0.955 \
  --actual_y 725.50
```

**Terminal Output:**
```text
======================================================================
🔍 CUSTOM ENTRY INPUT SPECIFICATIONS
======================================================================
  X1_AmbTemp           : 25.0
  X2_AmbPress          : 1.015
  X3_Humidity          : 60.0
  X4_ExhaustVac        : 45.0
  X5_TurbinePress      : 16.5
  X6_FuelFlow          : 18.0
  X7_SteamFlow         : 32.0
  X8_CoolingTemp       : 21.0
  X9_GenEfficiency     : 0.955

======================================================================
⚡ PREDICTION RESULT
======================================================================
  Derived Formula Output (Y) : 715.4670 MW
  Actual Ground Truth (Y)    : 725.5000 MW
  Absolute Residual Error    : 10.0330 MW
  Percentage Error           : 1.38%
======================================================================
```

---

#### 🔹 Method B: Batch Testing with a Custom CSV
1. **Prepare your CSV file** (e.g. `my_custom_entries.csv`) with the 9 parameter column headers:
   ```csv
   X1_AmbTemp,X2_AmbPress,X3_Humidity,X4_ExhaustVac,X5_TurbinePress,X6_FuelFlow,X7_SteamFlow,X8_CoolingTemp,X9_GenEfficiency,actual_y
   22.40,1.0125,58.20,40.10,16.80,18.50,33.00,18.40,0.9550,743.85
   15.20,1.0020,42.00,31.50,18.50,22.00,41.20,15.10,0.9700,795.12
   31.10,1.0250,75.00,60.20,13.20,11.50,20.00,26.50,0.9320,642.50
   ```
   *(Note: Headers can also be simplified to `X1,X2,X3,X4,X5,X6,X7,X8,X9`)*

2. **Execute the batch prediction command**:
   ```bash
   python3 predict_custom.py \
     --csv data/sample_custom_inputs.csv \
     --output data/predictions_output.csv
   ```

3. **Check the Output**:
   The script calculates predictions for every entry and, if `actual_y` is present, computes individual residuals and overall batch Mean Absolute Error (MAE).

---

#### 🔹 Method C: Standalone Python Function (Jupyter / Script)
You can embed this lightweight function into any Python application:

```python
def predict_power(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    """Calculates Net Power Output (MW) using the derived model formula."""
    return (
        419.8034
        - 1.8575 * x1
        + 42.2294 * x2
        - 0.1166 * x3
        - 0.3831 * x4
        + 4.2214 * x5
        + 6.1404 * x6
        + 2.7977 * x7
        - 1.4406 * x8
        + 87.9806 * x9
    )

# Test with your custom values:
predicted_mw = predict_power(
    x1=24.5,
    x2=1.015,
    x3=55.0,
    x4=42.0,
    x5=17.5,
    x6=20.0,
    x7=35.0,
    x8=19.5,
    x9=0.965
)
print(f"Predicted Net Power Output: {predicted_mw:.4f} MW")
```

---

#### 🔹 Method D: In Excel or Google Sheets
If your 9 parameters are in columns **A to I** (row 2):
1. Click on cell **J2**.
2. Paste this exact formula:
   ```excel
   = 419.8034 - (1.8575 * A2) + (42.2294 * B2) - (0.1166 * C2) - (0.3831 * D2) + (4.2214 * E2) + (6.1404 * F2) + (2.7977 * G2) - (1.4406 * H2) + (87.9806 * I2)
   ```
3. Drag the bottom right corner of cell J2 down to apply across all your custom entries.

---

## 🏗️ Folder Structure

```
benchmark_100_samples/
├── data/
│   ├── dataset_100_entries.csv       # Exactly 100 entries with 10 parameters
│   └── sample_custom_inputs.csv      # Sample custom test inputs for batch testing
├── plots/
│   └── model_comparison_benchmark.png # R² comparison bar plot & 1:1 parity scatter
├── models_benchmark.py               # Complete training & 5-model evaluation script
├── predict_custom.py                 # Multi-mode custom entry prediction tool
└── README.md                         # Complete documentation & test steps
```
