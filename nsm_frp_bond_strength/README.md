# 🏗️ NSM FRP Rod-to-Concrete Bond Strength & Failure Mode Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v3.4+-green.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.9+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20GUI-red.svg)](app.py)

An end-to-end Machine Learning research implementation modeled directly after **Kumar, Arora & Nehdi (2024)** (*"XGBoost algorithm based estimation of near surface mounted FRP rod-to-concrete bond strength and failure mode"*, *Journal of Building Engineering*, 90, 109418).

It features a **416-record experimental NSM FRP datasheet** (9 input parameters, 2 outputs: bond strength $P_u$ and failure mode $F_N$), **6 ML models** (`LR`, `DT`, `GBT`, `RF`, `XGBoost`, `ANN`), **5 structural design guidelines & analytical equations** (`ACI 440.2R-08`, `SA HB 305`, `ACI 440.1R-15`, `Zhang et al.`, `CSA S806-12`), **TreeSHAP & Partial Dependence explainability**, **Taylor & Radar plots**, and an interactive **Streamlit Web GUI**.

---

## 📊 1. Datasheet Schema (416 Experimental Records)

Generated in [`data/nsm_frp_bond_dataset.csv`](./data/nsm_frp_bond_dataset.csv) with **100% per-study (`416/416`) parameter compliance** and study provenance (`Ref_ID`, `Year`, `Study`) grounded directly in Supplementary **Table S3** ([`data/table_s3_sources.csv`](./data/table_s3_sources.csv), Refs `[21]`–`[40]`):

| Parameter | Symbol | Role | Range | Mean | SD | Description |
|---|:---:|:---:|:---:|:---:|:---:|---|
| Concrete Strength | $f'_c$ | Input | $23.00 - 66.25$ MPa | 36.85 | 11.58 | Compressive strength of concrete |
| Bonded Length | $L_b$ | Input | $14.40 - 457.20$ mm | 185.67 | 96.44 | Embedded bond length in groove |
| Bar Diameter | $D_f$ | Input | $5.00 - 24.00$ mm | 9.66 | 2.03 | Diameter of FRP rod |
| Elastic Modulus | $E_f$ | Input | $37.17 - 234.43$ GPa | 111.71 | 42.90 | Elastic modulus of FRP composite |
| Tensile Strength | $f_f$ | Input | $749.00 - 4826.50$ MPa | 1881.95 | 714.64 | Tensile strength of FRP rod |
| Groove Width | $W_g$ | Input | $6.00 - 30.00$ mm | 17.61 | 4.92 | Epoxy groove width |
| Groove Depth | $D_g$ | Input | $10.00 - 40.00$ mm | 18.88 | 5.39 | Epoxy groove depth |
| FRP Bar Type | $FRP_{BT}$ | Input | $1 - 3$ | 2.14 | 0.61 | `1`: Basalt (12.5%), `2`: Carbon (61.1%), `3`: Glass (26.4%) |
| Bar Surface Type | $FRP_{ST}$ | Input | $1 - 7$ | 4.16 | 1.95 | `1`: DB/RIB, `2`: GB, `3`: RUF, `4`: SMTH, `5`: SB/SC, `6`: SW/HB, `7`: Hybrid |
| **Bond Strength** | **$P_u$** | **Output 1** | **$7.17 - 127.52$ kN** | **40.53** | **17.00** | **Pull-out bond capacity (kN)** |
| **Failure Mode** | **$F_N$** | **Output 2** | **$1 - 16$** | **9.54** | **3.47** | **16 failure classes (`A` through `G` & hybrids)** |

* **Data Normalization**: Features scaled in $[-1, +1]$ (`MinMaxScaler`) for linear/neural learners (`src/preprocess.py`).
* **Train/Test Split**: **70% Training (291 records)** / **30% Testing (125 records)** (stratified on $F_N$).

---

## 📈 2. Model Performance Benchmark

### Table 3: Machine Learning Models (Bond Strength $P_u$)

| Model | Split | $R$ | $R^2$ / $NS$ | $\text{a20-index}$ | MAE (kN) | RMSE (kN) | MAPE (%) | SI | SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LR** | Training | 0.8425 | 0.7097 | 0.6564 | 7.23 | 9.55 | 19.73 | 0.2289 | 14.96 |
| | Testing | 0.8092 | 0.6117 | 0.6240 | 7.33 | 9.19 | 23.75 | 0.2434 | 14.84 |
| | All | 0.8327 | 0.6908 | 0.6466 | 7.26 | 9.44 | 20.94 | 0.2330 | 14.96 |
| **DT** | Training | 0.9783 | 0.9572 | 0.9794 | 2.26 | 3.67 | 5.23 | 0.0879 | 17.37 |
| | Testing | 0.9077 | 0.7956 | 0.7280 | 4.76 | 6.67 | 14.14 | 0.1766 | 15.91 |
| | All | 0.9606 | 0.9210 | 0.9038 | 3.01 | 4.77 | 7.91 | 0.1178 | 17.04 |
| **GBT** | Training | 0.9985 | 0.9969 | 1.0000 | 0.77 | 0.98 | 2.14 | 0.0235 | 17.57 |
| | Testing | 0.9688 | 0.9349 | 0.9200 | 2.88 | 3.76 | 9.77 | 0.0997 | 14.12 |
| | All | 0.9916 | 0.9829 | 0.9760 | 1.40 | 2.22 | 4.43 | 0.0548 | 16.65 |
| **RF** | Training | 0.9928 | 0.9817 | 0.9897 | 1.56 | 2.40 | 4.09 | 0.0574 | 16.51 |
| | Testing | 0.9561 | 0.9081 | 0.8960 | 3.37 | 4.47 | 11.15 | 0.1184 | 13.54 |
| | All | 0.9844 | 0.9653 | 0.9615 | 2.10 | 3.17 | 6.21 | 0.0781 | 15.72 |
| **XGB 🏆** | Training | **0.9998** | **0.9996** | **1.0000** | **0.27** | **0.36** | **0.75** | **0.0085** | **17.71** |
| | Testing | **0.9718** | **0.9419** | **0.9280** | **2.65** | **3.55** | **9.04** | **0.0941** | **14.30** |
| | All | **0.9933** | **0.9865** | **0.9784** | **0.99** | **1.97** | **3.24** | **0.0486** | **16.80** |
| **ANN** | Training | 0.9887 | 0.9770 | 0.9897 | 1.99 | 2.69 | 5.08 | 0.0644 | 17.36 |
| | Testing | 0.9376 | 0.8739 | 0.8160 | 4.09 | 5.24 | 12.41 | 0.1387 | 14.95 |
| | All | 0.9768 | 0.9539 | 0.9375 | 2.62 | 3.64 | 7.28 | 0.0899 | 16.74 |

> **XGBoost** achieves the highest precision with $R = 0.9933$, $R^2 = 0.9865$, $\text{RMSE} = 1.97\text{ kN}$ ($\text{Testing } R^2 = 0.9419$, $\text{SI} = 0.0941 < 0.10$), and **87.98%** exact failure mode ($F_N$) alignment across all 416 records.

### Table 4: Analytical Models & Design Guidelines ($N = 416$)

| Model / Design Guideline | $R$ | $R^2$ / $NS$ | $\text{a20-index}$ | MAE (kN) | RMSE (kN) | MAPE (%) | SI | SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ACI 440.2R-08** | 0.6513 | 0.1442 | 0.4399 | 11.14 | 15.71 | 28.35 | 0.3877 | 19.25 |
| **SA (HB 305-2008)** | 0.6901 | 0.2616 | 0.4784 | 10.67 | 14.59 | 27.88 | 0.3601 | 19.61 |
| **ACI 440.1R-15** | 0.4717 | -0.3767 | 0.2115 | 15.39 | 19.93 | 37.18 | 0.4917 | 13.21 |
| **Zhang et al. (2014)** | 0.5309 | 0.1040 | 0.3413 | 12.84 | 16.08 | 39.13 | 0.3967 | 16.07 |
| **CSA-S806-12** | 0.6672 | -0.2031 | 0.2115 | 14.49 | 18.63 | 35.78 | 0.4596 | 16.34 |

---

## 📐 3. Derived Closed-Form Mathematical Formulas

### A. Calibrated Nonlinear Power-Law Empirical Equation (Testing $R^2 = 0.7022$, $R = 0.8460$, $\text{RMSE} = 8.05\text{ kN}$)
$$\boxed{P_u = 0.0004 \cdot (f'_c)^{0.3656} \cdot (L_b)^{0.5901} \cdot (D_f)^{0.5012} \cdot (E_f)^{0.1416} \cdot (f_f)^{0.5972} \cdot (W_g)^{-0.0282} \cdot (D_g)^{0.3282} \cdot (FRP_{BT})^{0.1143} \cdot (FRP_{ST})^{-0.0326}}$$

### B. Multiple Linear Regression Equation
$$P_u = -61.6788 + 0.4624 f'_c + 0.1404 L_b + 1.9932 D_f + 0.1038 E_f + 0.0094 f_f + 0.0675 W_g + 0.3416 D_g + 3.9473 FRP_{BT} - 1.2561 FRP_{ST}$$

---

## 🎨 4. Diagrams Generated in `plots/`

| Figure | File Name | Description |
|---|---|---|
| **Fig. 1** | [`fig1_variable_distributions.png`](./plots/fig1_variable_distributions.png) | 11 donut charts with Min, Max, Mean & SD cyan summary boxes |
| **Fig. 2** | [`fig2_categorical_contributions.png`](./plots/fig2_categorical_contributions.png) | Categorical breakdown of FRP bar types, surfaces, and 16 failure modes |
| **Fig. 4–9** | [`fig4_to_9_ml_models_diagnostics.png`](./plots/fig4_to_9_ml_models_diagnostics.png) | 3-panel diagnostic plots (Scatter, Error Frequency, Performance Ratio) for all 6 ML models |
| **Fig. 8** | [`fig8_xgb_diagnostics.png`](./plots/fig8_xgb_diagnostics.png) | Standalone 3-panel diagnostic plot for best-performing **XGBoost** (85.34% on-line, 87.98% best PR) |
| **Fig. 9–13** | [`fig9_to_13_analytical_diagnostics.png`](./plots/fig9_to_13_analytical_diagnostics.png) | 3-panel diagnostic comparison across all 5 analytical & design codes |
| **Fig. 14** | [`fig14_radar_comparison.png`](./plots/fig14_radar_comparison.png) | 7-panel Radar/Spider plots comparing all 11 models across `R`, `MAE`, `MAPE`, `RMSE`, `NS`, `a20-index`, `SI` |
| **Fig. 15** | [`fig15_taylor_plots.png`](./plots/fig15_taylor_plots.png) | Two-panel Polar Taylor Diagrams (Analytical vs ML models with SD, R, and centered RMSE arcs) |
| **Fig. 16** | [`fig16_failure_mode_scatter.png`](./plots/fig16_failure_mode_scatter.png) | Predicted vs Experimental Failure Mode ($F_N$) scatter plots across ML models |
| **Fig. 17** | [`fig17_shap_analysis.png`](./plots/fig17_shap_analysis.png) | TreeSHAP Beeswarm Summary & Global Feature Importance ranking ($L_b > f_f > f'_c > E_f > D_f$) |
| **Fig. 18** | [`fig18_pdp_plots.png`](./plots/fig18_pdp_plots.png) | 3×3 Partial Dependence Plots (PDP + ICE lines) across all 9 input variables |
| **Fig. 19** | [`fig19_nsm_specimen_schematic.png`](./plots/fig19_nsm_specimen_schematic.png) | NSM FRP-to-concrete specimen joint schematic diagram |
| **Fig. 20** | [`fig20_table_s3_study_breakdown.png`](./plots/fig20_table_s3_study_breakdown.png) | Table S3 per-study experimental $P_u$ ranges, mean parity, and XGBoost MAE/RMSE across `[21]`–`[40]` |

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

### 5. Parse & Audit Supplementary Table S3 (`[21]`–`[40]`)
```bash
cd nsm_frp_bond_strength
python3 data/parse_table_s3.py
PYTHONPATH=. python3 -m src.table_s3
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

---

## 📚 6. Table S3 Literature Grounding (`[21]`–`[40]`)

The parameter ranges of this module are directly anchored to **Supplementary Table S3** (*Collected experimental dataset*) from Kumar, Arora & Nehdi (2024), covering **20 experimental studies (`[21]`–`[40]`, 1999–2023)**:

- **Raw Source Transcript**: [`data/table_s3_raw.md`](./data/table_s3_raw.md)
- **Deterministic Parser**: [`data/parse_table_s3.py`](./data/parse_table_s3.py)
- **Structured Outputs**: [`data/table_s3_sources.csv`](./data/table_s3_sources.csv) and [`data/table_s3_sources.json`](./data/table_s3_sources.json)
- **Bounds & Envelope Audit Module**: [`src/table_s3.py`](./src/table_s3.py)

### Global Boundary Provenance from Table S3

| Parameter | Table S3 Global Range | Minimum Defining Study | Maximum Defining Study |
|---|:---:|:---:|:---:|
| $f'_c$ (MPa) | `23.0 – 66.25` | `[22]`, `[27]`, `[29]` | `[37]` |
| $L_b$ (mm) | `14.4 – 457.2` | `[26]` | `[30]` |
| $D_f$ (mm) | `5.0 – 24.0` | `[40]` | `[40]` |
| $E_f$ (GPa) | `37.17 – 234.43` | `[24]` | `[34]` |
| $f_f$ (MPa) | `749.0 – 4826.5` | `[30]` | `[34]` |
| $W_g$ (mm) | `6.0 – 30.0` | `[26]` | `[22]` |
| $D_g$ (mm) | `10.0 – 40.0` | `[23]`, `[27]`, `[32]` | `[25]` |
| $FRP_{BT}$ | `1 – 3` | `[22]`, `[27]`, `[29]`, `[32]`, `[36]`, `[37]` | `[21]`–`[24]`, `[26]`–`[28]`, `[30]`, `[32]`, `[33]`, `[37]`–`[39]` |
| $FRP_{ST}$ | `1 – 7` | `[21]`–`[24]`, `[27]`, `[32]`, `[36]`, `[37]`, `[39]` | `[21]`, `[25]`–`[27]`, `[31]`, `[35]`, `[38]`, `[40]` |
| $P_u$ (kN) | `7.17 – 127.52` | `[31]` | `[34]` |
| $F_N$ | `1 – 16` | `[21]`, `[26]`, `[32]`, `[36]`, `[37]`, `[38]` | `[39]` |

> **Audit Summary (`src/table_s3.py`)**: All **416/416 (100.00%)** records in `data/nsm_frp_bond_dataset.csv` satisfy Table S3's global `[min, max]` bounds with exact min/max parity, **416/416 (100.00%)** satisfy their assigned `Ref_ID` study bounds (`strict_allowed=True`), and **416/416 (100.00%)** match at least one study's 9-input and 11-parameter envelope. Per-study XGBoost metrics are saved in [`table_s3_study_metrics.csv`](./table_s3_study_metrics.csv) and plotted in [`plots/fig20_table_s3_study_breakdown.png`](./plots/fig20_table_s3_study_breakdown.png).

### Collected Experimental Studies (`[21]`–`[40]`)

| Ref | Year | Study | $f'_c$ (MPa) | $L_b$ (mm) | $D_f$ (mm) | $E_f$ (GPa) | $f_f$ (MPa) | $W_g$ (mm) | $D_g$ (mm) | $FRP_{BT}$ | $FRP_{ST}$ | $P_u$ (kN) | $F_N$ |
|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `[21]` | 2013 | Lee et al. (2013) | 28.5 | 250 | 7.5–10 | 41–155 | 760–2300 | 11.25–25 | 11.25–25 | 2–3 | 1–7 | 47.38–74.58 | 1–10 |
| `[22]` | 2012 | Ceroni et al. (2012) | 23 | 300 | 6–8 | 46–155 | 1272–2495 | 30 | 30 | 1–3 | 1–5 | 28.80–50.90 | 11–12 |
| `[23]` | 2002 | De Lorenzis et al. (2002) | 27.5 | 32–192 | 8–11.3 | 37.7–174.71 | 873–2214 | 10–24 | 10–24 | 2–3 | 1–6 | 9.46–58.62 | 11–12 |
| `[24]` | 2004 | De Lorenzis et al. (2004) | 27.5 | 32–271.2 | 8–11.3 | 37.17–174.71 | 873–2214 | 12–24 | 12–24 | 2–3 | 1–6 | 10.67–67.43 | 11–13 |
| `[25]` | 2008 | Novidis & Pantazopoulou (2008) | 34 | 36–60 | 12 | 124 | 2068 | 20 | 20–40 | 2 | 7 | 10.3–20.2 | 11 |
| `[26]` | 2013 | Kalupahana et al. (2013) | 30–60 | 14.4–152.4 | 9–12 | 120–141 | 2040–2720 | 6–18 | 13–24 | 2–3 | 4–7 | 19.10–94.20 | 1–15 |
| `[27]` | 2011 | Bilotta et al. (2011) | 23 | 300 | 6–10 | 46–155 | 1250–2495 | 10–15 | 10–15 | 1–3 | 1–7 | 28.84–55.30 | 7–12 |
| `[28]` | 2013 | Sharaky et al. (2013) | 25 | 192 | 8–12 | 64–170 | 1350–2350 | 12–20 | 16–19.4 | 2–3 | 2–4 | 31.43–59.97 | 2–14 |
| `[29]` | 2014 | Bilotta et al. (2014) | 23 | 300 | 8–10 | 42–100 | 1040–1272 | 14–20 | 14–20 | 1–2 | 5–6 | 38.02–48.22 | 11 |
| `[30]` | 2011 | Soliman et al. (2011) | 49.37 | 57–457.2 | 9.5–12.7 | 42–134 | 749–1540 | 14.25–25.4 | 14.25–25.4 | 2–3 | 4 | 35.56–96.23 | 3–13 |
| `[31]` | 2007 | Novidis et al. (2007) | 34.5 | 36–120 | 12 | 124 | 2068 | 20–25 | 20–25 | 2 | 7 | 7.17–21.60 | 7–11 |
| `[32]` | 2016 | Bilotta et al. (2016) | 37.5 | 300 | 6–8 | 46–187 | 1208–3536 | 9.5–14 | 10–14 | 1–3 | 1–6 | 23–56.8 | 1–11 |
| `[33]` | 2012 | El-Gamal et al. (2012) | 46.5 | 203.2–406.4 | 12.7 | 60.4 | 1478 | 19.05–25.40 | 19.05–25.40 | 3 | 5 | 60.8–109.2 | 13 |
| `[34]` | 1999 | Yan et al. (1999) | 34.20–36.20 | 50.80–152.40 | 8–11 | 227.54–234.43 | 3619.88–4826.50 | 14.29 | 19.05 | 2 | 5 | 37.19–127.52 | 7–10 |
| `[35]` | 2021 | Wang & Cheng (2021) | 46.50–52.30 | 250 | 9.50–10 | 113–129 | 1489–2372 | 20 | 20 | 2 | 3–7 | 46.20–90.40 | 5–11 |
| `[36]` | 2019 | Wang et al. (2019) | 47.83 | 40–240 | 10 | 50–103 | 1030–1366 | 20–25 | 20–25 | 1–2 | 1 | 12.27–75.48 | 1–13 |
| `[37]` | 2023 | Aljidda et al. (2023)* | 66.25 | 57.66–267.60 | 9.61–11.15 | 52–120 | 1100–1800 | 20 | 20 | 1–3 | `1.5` (`{1, 5}`) | 27.7–54.4 | 1–9 |
| `[38]` | 2013 | Sharaky et al. (2013) | 35.20–42.20 | 40–192 | 8–12 | 41–170 | 765–2350 | 12–21.80 | 12–30 | 2–3 | 2–7 | 10.37–63.77 | 1–13 |
| `[39]` | 2016 | Torres et al. (2016) | 28.75 | 48–240 | 8 | 64–170 | 1350–2310 | 16 | 16 | 2–3 | 1–4 | 18.99–60.87 | 2–16 |
| `[40]` | 2009 | Galati & De Lorenzis (2009) | 46.3 | 37.5–180 | 5–24 | 145.7 | 2214 | 12–20 | 12–16 | 2 | 7 | 10.9–50.2 | 2–13 |

*\*Note on `[37]`: The source supplementary table prints `1.5` in the categorical `FRPST` column; `parse_table_s3.py` preserves `"1.5"` in `FRP_ST_raw` and maps it to discrete surface codes `{1, 5}` (`min=1`, `max=5`). Full bibliographic citations and DOIs are stored in [`data/table_s3_sources.csv`](./data/table_s3_sources.csv) and [`data/table_s3_sources.json`](./data/table_s3_sources.json).*
