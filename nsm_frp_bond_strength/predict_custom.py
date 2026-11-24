"""
CLI & Batch CSV Prediction Tool for Custom NSM FRP Entries
Evaluates custom specimens using:
  1. Best-performing XGBoost model (Pu & Failure Mode FN)
  2. All other ML models (LR, DT, GBT, RF, ANN)
  3. Derived Nonlinear Power-Law & Multiple Linear Regression Formulas
  4. Design Guidelines (ACI 440.2R-08, SA, ACI 440.1R-15, Zhang et al., CSA-S806)
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from src.preprocess import FEATURE_COLUMNS
from src.analytical_models import ANALYTICAL_MODELS
from src.train import power_law_func
from data.generate_data import FAILURE_DESCRIPTIONS, FRP_BAR_TYPES, FRP_SURFACE_TYPES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_models():
    scaler = joblib.load(os.path.join(MODELS_DIR, "minmax_scaler.joblib"))
    fn_encoder = joblib.load(os.path.join(MODELS_DIR, "fn_label_encoder.joblib"))
    pu_models = {name: joblib.load(os.path.join(MODELS_DIR, f"pu_{name}.joblib")) for name in ["LR", "DT", "GBT", "RF", "XGB", "ANN"]}
    fn_models = {name: joblib.load(os.path.join(MODELS_DIR, f"fn_{name}.joblib")) for name in ["LR", "DT", "GBT", "RF", "XGB", "ANN"]}
    with open(os.path.join(MODELS_DIR, "derived_formulas.json"), "r") as f:
        formulas = json.load(f)
    return scaler, fn_encoder, pu_models, fn_models, formulas


def predict_single(features: dict, actual_pu: float = None):
    scaler, fn_encoder, pu_models, fn_models, formulas = load_models()
    df_in = pd.DataFrame([features])[FEATURE_COLUMNS]
    df_sc = pd.DataFrame(scaler.transform(df_in), columns=FEATURE_COLUMNS)

    print("\n" + "=" * 72)
    print("🏗️  CUSTOM NSM FRP SPECIMEN PARAMETERS")
    print("=" * 72)
    for k, v in features.items():
        extra = ""
        if k == "FRP_BT":
            extra = f" ({FRP_BAR_TYPES.get(int(v), '')})"
        elif k == "FRP_ST":
            extra = f" ({FRP_SURFACE_TYPES.get(int(v), '')})"
        print(f"  {k:<12}: {v}{extra}")

    print("\n" + "=" * 72)
    print("📊 BOND STRENGTH (Pu) & FAILURE MODE (FN) PREDICTIONS")
    print("=" * 72)
    for m_name, mdl in pu_models.items():
        p_val = float(mdl.predict(df_sc if m_name == "ANN" else df_in)[0])
        fn_code = int(fn_encoder.inverse_transform(fn_models[m_name].predict(df_sc if m_name in ["LR", "ANN"] else df_in))[0])
        fid, _ = FAILURE_DESCRIPTIONS[fn_code]
        marker = " 🏆" if m_name == "XGB" else ""
        print(f"  {m_name:<5} -> Pu = {p_val:>7.2f} kN | Failure Mode = FN {fn_code:>2} ({fid}){marker}")

    # Empirical nonlinear power-law formula
    pw = formulas["nonlinear_power_law"]
    exps = [pw["exponents"][c] for c in FEATURE_COLUMNS]
    p_pw = float(power_law_func(df_in.values, pw["alpha"], *exps)[0])
    print("-" * 72)
    print(f"  Empirical Power-Law Formula -> Pu = {p_pw:>7.2f} kN")

    # Analytical Design Codes
    print("\n📐 Design Guidelines & Analytical Comparisons:")
    for g_name, g_func in ANALYTICAL_MODELS.items():
        print(f"  {g_name:<15}: {float(g_func(df_in)[0]):>7.2f} kN")

    if actual_pu is not None:
        xgb_pu = float(pu_models["XGB"].predict(df_in)[0])
        err = abs(actual_pu - xgb_pu)
        print("-" * 72)
        print(f"  Actual Experimental Pu: {actual_pu:.2f} kN | XGB Error: {err:.2f} kN ({err/actual_pu*100:.2f}%)")
    print("=" * 72 + "\n")


def predict_batch(csv_path: str, output_path: str = None):
    scaler, fn_encoder, pu_models, fn_models, formulas = load_models()
    df = pd.read_csv(csv_path)
    X_in = df[FEATURE_COLUMNS]
    X_sc = pd.DataFrame(scaler.transform(X_in), columns=FEATURE_COLUMNS)

    df["Pred_Pu_XGB_kN"] = np.round(pu_models["XGB"].predict(X_in), 2)
    df["Pred_Pu_RF_kN"] = np.round(pu_models["RF"].predict(X_in), 2)
    df["Pred_Pu_ANN_kN"] = np.round(pu_models["ANN"].predict(X_sc), 2)
    df["Pred_Pu_LR_kN"] = np.round(pu_models["LR"].predict(X_in), 2)

    fn_preds = fn_encoder.inverse_transform(fn_models["XGB"].predict(X_in))
    df["Pred_FN_XGB"] = fn_preds
    df["Pred_Failure_ID"] = [FAILURE_DESCRIPTIONS[int(f)][0] for f in fn_preds]

    if "Pu_kN" in df.columns:
        df["XGB_Abs_Error_kN"] = np.round(np.abs(df["Pu_kN"] - df["Pred_Pu_XGB_kN"]), 2)
        print(f"Batch XGBoost MAE: {df['XGB_Abs_Error_kN'].mean():.2f} kN")

    print("\nBatch Prediction Preview:")
    cols = ["fc_MPa", "Lb_mm", "Df_mm", "Pred_Pu_XGB_kN", "Pred_FN_XGB", "Pred_Failure_ID"]
    print(df[cols].to_string(index=False))

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved batch predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict NSM FRP Bond Strength & Failure Mode.")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV with custom samples")
    parser.add_argument("--output", type=str, default=None, help="Path to save batch CSV predictions")
    parser.add_argument("--fc", type=float, default=37.90, help="Concrete compressive strength f'c (MPa)")
    parser.add_argument("--lb", type=float, default=180.0, help="Bonded length Lb (mm)")
    parser.add_argument("--df", type=float, default=10.0, help="FRP rod diameter Df (mm)")
    parser.add_argument("--ef", type=float, default=140.0, help="FRP elastic modulus Ef (GPa)")
    parser.add_argument("--ff", type=float, default=2100.0, help="FRP tensile strength ff (MPa)")
    parser.add_argument("--wg", type=float, default=16.0, help="Groove width Wg (mm)")
    parser.add_argument("--dg", type=float, default=18.0, help="Groove depth Dg (mm)")
    parser.add_argument("--bt", type=int, default=2, choices=[1, 2, 3], help="FRP bar type (1:B, 2:C, 3:G)")
    parser.add_argument("--st", type=int, default=5, choices=[1, 2, 3, 4, 5, 6, 7], help="FRP surface type (1..7)")
    parser.add_argument("--actual_pu", type=float, default=None, help="Optional experimental Pu (kN)")

    args = parser.parse_args()
    if args.csv:
        predict_batch(args.csv, args.output)
    else:
        feats = {
            "fc_MPa": args.fc,
            "Lb_mm": args.lb,
            "Df_mm": args.df,
            "Ef_GPa": args.ef,
            "ff_MPa": args.ff,
            "Wg_mm": args.wg,
            "Dg_mm": args.dg,
            "FRP_BT": args.bt,
            "FRP_ST": args.st,
        }
        predict_single(feats, args.actual_pu)


if __name__ == "__main__":
    main()
