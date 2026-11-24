# ⚡ 100-Entry Regression Benchmark: 9 Inputs, 1 Output

This benchmark evaluates 5 distinct machine learning algorithms on a 100-entry engineering dataset (9 input parameters, 1 output parameter) with an exact 70/30 train-validation split.

---

## 📊 Dataset Parameters (100 Entries)

| Parameter | Symbol | Type | Range | Description |
|---|---|---|---|---|
| Ambient Temperature | $X_1$ | Input | 10.0 - 35.0 °C | Ambient atmospheric temperature |
| Ambient Pressure | $X_2$ | Input | 0.98 - 1.04 bar | Ambient atmospheric barometric pressure |
| Relative Humidity | $X_3$ | Input | 30.0 - 90.0 % | Ambient air humidity percentage |
| Exhaust Vacuum | $X_4$ | Input | 25.0 - 75.0 cm Hg | Steam turbine exhaust vacuum |
| Turbine Pressure | $X_5$ | Input | 12.0 - 20.0 bar | High-pressure turbine inlet steam pressure |
| Fuel Flow Rate | $X_6$ | Input | 8.0 - 25.0 kg/s | Primary combustion fuel flow rate |
| Steam Flow Rate | $X_7$ | Input | 15.0 - 50.0 kg/s | Generated steam mass flow rate |
| Cooling Water Temp | $X_8$ | Input | 12.0 - 30.0 °C | Condenser cooling water supply temp |
| Generator Efficiency | $X_9$ | Input | 0.92 - 0.98 | Generator electrical mechanical efficiency |
| **Net Power Output** | **$Y$** | **Output** | **480 - 820 MW** | **Total net electric power output** |

- **Training Records**: 70 entries (indices sampled randomly with fixed seed `42`)
- **Validation Records**: 30 entries (held-out test split)

---

## 📈 Model Performance Benchmark (30 Validation Entries)

| Rank | Model Architecture | $R^2$ Score | RMSE (MW) | MAE (MW) | MSE | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Multiple Linear Regression (OLS)** | **0.999695** | **0.8041** | **0.6589** | **0.6466** | **Selected Best** |
| 🥈 | **Ridge Regression ($L_2$ Regularized)** | **0.998560** | 1.7467 | 1.3719 | 3.0509 | Runner-up |
| 🥉 | **Gradient Boosting Regressor (GBM)** | **0.828397** | 19.0707 | 16.8452 | 363.6904 | Sub-optimal ($N=70$) |
| 4 | **Random Forest Regressor** | **0.754297** | 22.8196 | 19.2857 | 520.7358 | Tree step quantization |
| 5 | **Artificial Neural Network (MLP)** | **-5.097646** | 113.6800 | 98.7559 | 12,923.15 | Overfitting ($N=70$) |

---

## 📐 Derived Mathematical Formula for Selected Model

The selected model is **Multiple Linear Regression (OLS)** with an $R^2$ of **0.999695** (closest to 1.0).

$$\hat{Y} = \beta_0 + \sum_{i=1}^9 \beta_i X_i$$

### Explicit Parameter Coefficients:
- $\beta_0$ (Intercept) = $+419.8034$
- $\beta_1$ ($X_1$, Ambient Temp) = $-1.8575$
- $\beta_2$ ($X_2$, Ambient Pressure) = $+42.2294$
- $\beta_3$ ($X_3$, Relative Humidity) = $-0.1166$
- $\beta_4$ ($X_4$, Exhaust Vacuum) = $-0.3831$
- $\beta_5$ ($X_5$, Turbine Pressure) = $+4.2214$
- $\beta_6$ ($X_6$, Fuel Flow Rate) = $+6.1404$
- $\beta_7$ ($X_7$, Steam Flow Rate) = $+2.7977$
- $\beta_8$ ($X_8$, Cooling Water Temp) = $-1.4406$
- $\beta_9$ ($X_9$, Generator Efficiency) = $+87.9806$

### Complete Prediction Formula:
$$\boxed{\hat{Y} = 419.8034 - 1.8575 X_1 + 42.2294 X_2 - 0.1166 X_3 - 0.3831 X_4 + 4.2214 X_5 + 6.1404 X_6 + 2.7977 X_7 - 1.4406 X_8 + 87.9806 X_9}$$
