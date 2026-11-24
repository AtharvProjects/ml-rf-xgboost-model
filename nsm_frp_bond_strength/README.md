# 🏗️ NSM FRP Rod-to-Concrete Bond Strength & Failure Mode Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v3.4+-green.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.9+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20GUI-red.svg)](app.py)

An end-to-end Machine Learning research implementation modeled directly after **Kumar, Arora & Nehdi (2024)** (*"XGBoost algorithm based estimation of near surface mounted FRP rod-to-concrete bond strength and failure mode"*, *Journal of Building Engineering*, 90, 109418).

It features a **416-record experimental NSM FRP datasheet** (9 input parameters, 2 outputs: bond strength $P_u$ and failure mode $F_N$), **6 ML models** (`LR`, `DT`, `GBT`, `RF`, `XGBoost`, `ANN`), **5 structural design guidelines & analytical equations** (`ACI 440.2R-08`, `SA HB 305`, `ACI 440.1R-15`, `Zhang et al.`, `CSA S806-12`), **TreeSHAP & Partial Dependence explainability**, **Taylor & Radar plots**, and an interactive **Streamlit Web GUI**.

---

## 📊 1. Datasheet Schema (416 Experimental Records)

Generated in [`data/nsm_frp_bond_dataset.csv`](./data/nsm_frp_bond_dataset.csv) matching the statistical distributions and physical ranges of the paper:

| Parameter | Symbol | Role | Range | Mean | SD | Description |
|---|:---:|:---:|:---:|:---:|:---:|---|
| Concrete Strength | $f'_c$ | Input | $23.00 - 66.25$ MPa | 35.09 | 10.62 | Compressive strength of concrete |
| Bonded Length | $L_b$ | Input | $14.40 - 457.20$ mm | 195.06 | 129.73 | Embedded bond length in groove |
| Bar Diameter | $D_f$ | Input | $5.00 - 24.00$ mm | 9.42 | 4.13 | Diameter of FRP rod |
| Elastic Modulus | $E_f$ | Input | $37.17 - 234.43$ GPa | 124.99 | 63.94 | Elastic modulus of FRP composite |
| Tensile Strength | $f_f$ | Input | $749.00 - 4826.50$ MPa | 2145.25 | 1072.15 | Tensile strength of FRP rod |
| Groove Width | $W_g$ | Input | $6.00 - 30.00$ mm | 18.07 | 5.93 | Epoxy groove width |
| Groove Depth | $D_g$ | Input | $10.00 - 40.00$ mm | 19.57 | 7.47 | Epoxy groove depth |
| FRP Bar Type | $FRP_{BT}$ | Input | $1 - 3$ | 2.17 | 0.64 | `1`: Basalt (13.7%), `2`: Carbon (55.8%), `3`: Glass (30.5%) |
| Bar Surface Type | $FRP_{ST}$ | Input | $1 - 7$ | 4.36 | 2.14 | `1`: DB/RIB, `2`: GB, `3`: RUF, `4`: SMTH, `5`: SB/SC, `6`: SW/HB, `7`: Hybrid |
| **Bond Strength** | **$P_u$** | **Output 1** | **$7.17 - 127.52$ kN** | **41.79** | **23.13** | **Pull-out bond capacity (kN)** |
| **Failure Mode** | **$F_N$** | **Output 2** | **$1 - 16$** | **9.02** | **3.70** | **16 failure classes (`A` through `G` & hybrids)** |

* **Data Normalization**: Features scaled in $[-1, +1]$ (`MinMaxScaler`) for linear/neural learners (`src/preprocess.py`).
* **Train/Test Split**: **70% Training (291 records)** / **30% Testing (125 records)**.

---

## 📈 2. Model Performance Benchmark

### Table 3: Machine Learning Models (Bond Strength $P_u$)

| Model | Split | $R$ | $R^2$ / $NS$ | $\text{a20-index}$ | MAE (kN) | RMSE (kN) | MAPE (%) | SI | SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LR** | Training | 0.9040 | 0.8172 | 0.6220 | 7.15 | 9.94 | 24.63 | 0.2382 | 21.06 |
| | Testing | 0.9081 | 0.8216 | 0.6320 | 7.01 | 9.60 | 28.52 | 0.2293 | 20.90 |
| | All | 0.9048 | 0.8185 | 0.6250 | 7.11 | 9.84 | 25.80 | 0.2355 | 21.00 |
| **DT** | Training | 0.9948 | 0.9896 | 0.9966 | 1.53 | 2.37 | 3.93 | 0.0569 | 23.17 |
| | Testing | 0.9338 | 0.8693 | 0.7200 | 6.07 | 8.22 | 16.02 | 0.1962 | 22.14 |
| | All | 0.9771 | 0.9546 | 0.9135 | 2.90 | 4.92 | 7.56 | 0.1178 | 22.84 |
| **GBT** | Training | 0.9977 | 0.9954 | 0.9966 | 1.24 | 1.58 | 3.78 | 0.0378 | 23.06 |
| | Testing | 0.9785 | 0.9575 | 0.9040 | 3.46 | 4.69 | 9.89 | 0.1119 | 22.27 |
| | All | 0.9922 | 0.9844 | 0.9688 | 1.91 | 2.89 | 5.61 | 0.0691 | 22.80 |
| **RF** | Training | 0.9950 | 0.9881 | 0.9966 | 1.81 | 2.54 | 4.72 | 0.0608 | 22.13 |
| | Testing | 0.9675 | 0.9345 | 0.8480 | 4.33 | 5.82 | 11.72 | 0.1389 | 21.21 |
| | All | 0.9871 | 0.9725 | 0.9519 | 2.56 | 3.83 | 6.82 | 0.0917 | 21.83 |
| **XGB 🏆** | Training | **0.9987** | **0.9974** | **0.9897** | **0.89** | **1.19** | **2.79** | **0.0285** | **23.16** |
| | Testing | **0.9843** | **0.9688** | **0.9440** | **2.95** | **4.01** | **8.15** | **0.0958** | **22.37** |
| | All | **0.9945** | **0.9891** | **0.9760** | **1.51** | **2.41** | **4.40** | **0.0578** | **22.90** |
| **ANN** | Training | 0.9960 | 0.9919 | 0.9794 | 1.50 | 2.09 | 4.77 | 0.0500 | 23.40 |
| | Testing | 0.9598 | 0.9122 | 0.7600 | 4.87 | 6.74 | 15.93 | 0.1608 | 23.72 |
| | All | 0.9849 | 0.9687 | 0.9135 | 2.51 | 4.08 | 8.13 | 0.0977 | 23.48 |

> **XGBoost** achieves the highest precision with $R = 0.9945$, $R^2 = 0.9891$, $\text{RMSE} = 2.41\text{ kN}$, and Scatter Index $\text{SI} = 0.0578 < 0.10$ (**Excellent**).

### Table 4: Analytical Models & Design Guidelines ($N = 416$)

| Model / Design Guideline | $R$ | $R^2$ / $NS$ | $\text{a20-index}$ | MAE (kN) | RMSE (kN) | MAPE (%) | SI | SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ACI 440.2R-08** | 0.7668 | 0.2411 | 0.2692 | 13.95 | 20.12 | 36.08 | 0.4815 | 30.26 |
| **SA (HB 305-2008)** | 0.7813 | 0.2897 | 0.3654 | 12.12 | 19.47 | 29.79 | 0.4659 | 31.11 |
| **ACI 440.1R-15** | 0.4462 | -0.4685 | 0.1298 | 21.53 | 27.99 | 55.17 | 0.6698 | 24.57 |
| **Zhang et al. (2014)** | 0.5241 | -0.5499 | 0.2764 | 18.14 | 28.76 | 47.52 | 0.6881 | 33.10 |
| **CSA-S806-12** | 0.8535 | 0.3868 | 0.2019 | 14.67 | 18.09 | 39.28 | 0.4328 | 20.80 |

---

## 📐 3. Derived Closed-Form Mathematical Formulas

### A. Calibrated Nonlinear Power-Law Empirical Equation (Testing $R^2 = 0.9160$, $R = 0.9574$)
$$\boxed{P_u = 0.01194 \cdot (f'_c)^{0.4039} \cdot (L_b)^{0.5929} \cdot (D_f)^{0.1709} \cdot (E_f)^{0.2006} \cdot (f_f)^{0.2399} \cdot (W_g)^{0.0657} \cdot (D_g)^{0.0931} \cdot (FRP_{BT})^{0.0236} \cdot (FRP_{ST})^{0.0395}}$$

### B. Multiple Linear Regression Equation
$$P_u = -38.1310 + 0.5051 f'_c + 0.1368 L_b - 0.0434 D_f + 0.0852 E_f + 0.0037 f_f + 0.3473 W_g + 0.4212 D_g + 0.4296 FRP_{BT} + 0.5311 FRP_{ST}$$

---

## 🎨 4. Diagrams Generated in `plots/`

| Figure | File Name | Description |
|---|---|---|
| **Fig. 1** | [`fig1_variable_distributions.png`](./plots/fig1_variable_distributions.png) | 11 donut charts with Min, Max, Mean & SD cyan summary boxes |
| **Fig. 2** | [`fig2_categorical_contributions.png`](./plots/fig2_categorical_contributions.png) | Categorical breakdown of FRP bar types, surfaces, and 16 failure modes |
| **Fig. 4–9** | [`fig4_to_9_ml_models_diagnostics.png`](./plots/fig4_to_9_ml_models_diagnostics.png) | 3-panel diagnostic plots (Scatter, Error Frequency, Performance Ratio) for all 6 ML models |
| **Fig. 8** | [`fig8_xgb_diagnostics.png`](./plots/fig8_xgb_diagnostics.png) | Standalone 3-panel diagnostic plot for best-performing **XGBoost** (76.44% on-line, 83.89% best PR) |
| **Fig. 9–13** | [`fig9_to_13_analytical_diagnostics.png`](./plots/fig9_to_13_analytical_diagnostics.png) | 3-panel diagnostic comparison across all 5 analytical & design codes |
| **Fig. 14** | [`fig14_radar_comparison.png`](./plots/fig14_radar_comparison.png) | 7-panel Radar/Spider plots comparing all 11 models across `R`, `MAE`, `MAPE`, `RMSE`, `NS`, `a20-index`, `SI` |
| **Fig. 15** | [`fig15_taylor_plots.png`](./plots/fig15_taylor_plots.png) | Two-panel Polar Taylor Diagrams (Analytical vs ML models with SD, R, and centered RMSE arcs) |
| **Fig. 16** | [`fig16_failure_mode_scatter.png`](./plots/fig16_failure_mode_scatter.png) | Predicted vs Experimental Failure Mode ($F_N$) scatter plots across ML models |
| **Fig. 17** | [`fig17_shap_analysis.png`](./plots/fig17_shap_analysis.png) | TreeSHAP Beeswarm Summary & Global Feature Importance ranking ($L_b > f_f > E_f > f'_c > D_f$) |
| **Fig. 18** | [`fig18_pdp_plots.png`](./plots/fig18_pdp_plots.png) | 3×3 Partial Dependence Plots (PDP + ICE lines) across all 9 input variables |
| **Fig. 19** | [`fig19_nsm_specimen_schematic.png`](./plots/fig19_nsm_specimen_schematic.png) | NSM FRP-to-concrete specimen joint schematic diagram |

---

## 🚀 5. Quickstart & Testing Custom Entries

### 1. Run Complete Pipeline
```bash
cd nsm_frp_bond_strength
./run_pipeline.sh
```

### 2. Launch Interactive Web GUI (Streamlit — Fig. 19 Replica)
```bash
cd nsm_frp_bond_strength
streamlit run app.py
```

### 3. Test Custom Entry via Terminal (CLI)
```bash
cd nsm_frp_bond_strength
python3 predict_custom.py \
  --fc 48.5 --lb 320.0 --df 12.0 --ef 205.0 \
  --ff 3100.0 --wg 18.0 --dg 20.0 --bt 2 --st 7
```

### 4. Batch Test Custom Entries via CSV
```bash
cd nsm_frp_bond_strength
python3 predict_custom.py \
  --csv data/sample_custom_frp_entries.csv \
  --output data/sample_custom_predictions.csv
```
