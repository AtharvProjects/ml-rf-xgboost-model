"""
Interactive Streamlit Web GUI for NSM FRP Rod-to-Concrete Bond Strength & Failure Mode
Mirrors Fig. 19 of Kumar et al. (Journal of Building Engineering 90, 2024, 109418).
Run with: streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st

from src.preprocess import FEATURE_COLUMNS
from src.analytical_models import ANALYTICAL_MODELS
from src.train import power_law_func
from data.generate_data import FAILURE_DESCRIPTIONS, FRP_BAR_TYPES, FRP_SURFACE_TYPES

st.set_page_config(
    page_title="NSM FRP-to-Concrete Bond Strength Predictor",
    page_icon="🏗️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "minmax_scaler.joblib"))
    fn_encoder = joblib.load(os.path.join(MODELS_DIR, "fn_label_encoder.joblib"))
    pu_models = {name: joblib.load(os.path.join(MODELS_DIR, f"pu_{name}.joblib")) for name in ["LR", "DT", "GBT", "RF", "XGB", "ANN"]}
    fn_models = {name: joblib.load(os.path.join(MODELS_DIR, f"fn_{name}.joblib")) for name in ["LR", "DT", "GBT", "RF", "XGB", "ANN"]}
    with open(os.path.join(MODELS_DIR, "derived_formulas.json"), "r") as f:
        formulas = json.load(f)
    return scaler, fn_encoder, pu_models, fn_models, formulas


def main():
    scaler, fn_encoder, pu_models, fn_models, formulas = load_artifacts()

    # Sidebar: Dependent Parameters (Sliders matching Fig. 19)
    st.sidebar.header("Dependent Parameters")
    fc = st.sidebar.slider("Compressive strength of concrete (MPa)", 23.00, 66.25, 37.90, 0.25)
    lb = st.sidebar.slider("Bonded length (mm)", 14.40, 457.20, 179.90, 1.0)
    df_bar = st.sidebar.slider("Diameter of FRP bar (mm)", 5.00, 24.00, 9.52, 0.1)
    ef = st.sidebar.slider("Elastic modulus of FRP composite (GPa)", 37.17, 234.43, 112.41, 0.5)
    ff = st.sidebar.slider("Tensile strength of FRP Composite (MPa)", 749.00, 4826.50, 1926.36, 10.0)
    wg = st.sidebar.slider("Width of groove (mm)", 6.00, 30.00, 16.93, 0.1)
    dg = st.sidebar.slider("Depth of groove (mm)", 10.00, 40.00, 17.48, 0.1)

    frp_bt = st.sidebar.selectbox(
        "FRP bar type (1: Basalt, 2: Carbon, 3: Glass)",
        options=[1, 2, 3],
        index=1,
        format_func=lambda x: f"{x} - {FRP_BAR_TYPES[x]}",
    )
    frp_st = st.sidebar.selectbox(
        "FRP bar surface type (1 to 7)",
        options=[1, 2, 3, 4, 5, 6, 7],
        index=3,
        format_func=lambda x: f"{x} - {FRP_SURFACE_TYPES[x]}",
    )

    # Header matching Fig. 19
    st.markdown(
        "<h2 style='text-align: center; color: #1a237e;'>Prediction of FRP-to-Concrete Bond Strength of NSM Bars Adhered with Epoxy</h2>",
        unsafe_allow_html=True,
    )

    schematic_path = os.path.join(PLOTS_DIR, "fig19_nsm_specimen_schematic.png")
    if os.path.exists(schematic_path):
        st.image(schematic_path, use_container_width=True)

    # Input DataFrame
    input_df = pd.DataFrame([{
        "fc_MPa": fc,
        "Lb_mm": lb,
        "Df_mm": df_bar,
        "Ef_GPa": ef,
        "ff_MPa": ff,
        "Wg_mm": wg,
        "Dg_mm": dg,
        "FRP_BT": frp_bt,
        "FRP_ST": frp_st,
    }])[FEATURE_COLUMNS]
    input_sc = pd.DataFrame(scaler.transform(input_df), columns=FEATURE_COLUMNS)

    st.subheader("Specified Input Parameters")
    display_df = input_df.copy()
    display_df.columns = ["fc", "Lb", "Df", "Ef", "ff", "Wg", "Dg", "FRPBT", "FRPST"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Primary XGBoost Prediction
    xgb_pu = float(pu_models["XGB"].predict(input_df)[0])
    xgb_fn_code = int(fn_encoder.inverse_transform(fn_models["XGB"].predict(input_df))[0])
    fail_id, fail_desc = FAILURE_DESCRIPTIONS[xgb_fn_code]

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"### Predicted Bond Strength ($P_u$): **{xgb_pu:.2f} kN**")
        st.caption("Model: Extreme Gradient Boosting (XGBoost)")
    with col2:
        st.info(f"### Predicted Failure Mode: **FN {xgb_fn_code} ({fail_id})**")
        st.caption(f"Mechanism: {fail_desc}")

    # Cross-model comparison + Analytical Codes + Empirical Formula
    st.subheader("Comparison Across ML Models, Empirical Formula & Design Guidelines")
    comp_rows = []
    for m_name, mdl in pu_models.items():
        p_val = float(mdl.predict(input_sc if m_name == "ANN" else input_df)[0])
        comp_rows.append({"Category": "ML Model", "Model / Guideline": m_name, "Predicted Pu (kN)": round(p_val, 2)})

    pw = formulas["nonlinear_power_law"]
    exps = [pw["exponents"][c] for c in FEATURE_COLUMNS]
    p_pw = float(power_law_func(input_df.values, pw["alpha"], *exps)[0])
    comp_rows.append({"Category": "Derived Formula", "Model / Guideline": "Empirical Nonlinear Power-Law", "Predicted Pu (kN)": round(p_pw, 2)})

    for g_name, g_func in ANALYTICAL_MODELS.items():
        p_g = float(g_func(input_df)[0])
        comp_rows.append({"Category": "Design Code / Analytical", "Model / Guideline": g_name, "Predicted Pu (kN)": round(p_g, 2)})

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Live TreeSHAP Local Explanation
    st.subheader("Local SHAP Explanation for Current Specimen (XGBoost)")
    dmat = xgb.DMatrix(input_df)
    contribs = pu_models["XGB"].get_booster().predict(dmat, pred_contribs=True)[0]
    shap_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Input Value": input_df.iloc[0].values,
        "SHAP Contribution (kN)": np.round(contribs[:-1], 3),
    }).sort_values(by="SHAP Contribution (kN)", key=abs, ascending=False)
    st.bar_chart(shap_df.set_index("Feature")["SHAP Contribution (kN)"])


if __name__ == "__main__":
    main()
