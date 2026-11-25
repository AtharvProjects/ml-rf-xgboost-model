"""
Comprehensive Evaluation Script for NSM FRP Bond Strength & Failure Mode
Computes:
  - Table 3: ML Model Metrics (Training 70%, Testing 30%, All 100%)
  - Table 4: Analytical & Design Guidelines Metrics
  - Table 5: Graphical Diagnostic Breakdown (SP, FEP, PRP)
  - Failure Mode (FN) exact & tolerance accuracy
  - Closed-form Empirical Nonlinear Power-Law & Linear Regression Formula evaluation
Saves CSV tables and metrics_summary.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.preprocess import prepare_splits, FEATURE_COLUMNS
from src.analytical_models import (
    ANALYTICAL_MODELS,
    compute_paper_metrics,
    compute_table5_breakdown,
)
from src.train import power_law_func


def evaluate_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)

    splits = prepare_splits(test_size=0.30, random_state=42, save_scaler=False)
    df = splits["df"]
    X_tr, X_te = splits["X_train"], splits["X_test"]
    X_tr_sc, X_te_sc = splits["X_train_scaled"], splits["X_test_scaled"]
    y_pu_tr, y_pu_te = splits["y_pu_train"].values, splits["y_pu_test"].values
    y_fn_tr, y_fn_te = splits["y_fn_train"].values, splits["y_fn_test"].values
    y_pu_all = df["Pu_kN"].values
    y_fn_all = df["FN"].values

    X_all = df[FEATURE_COLUMNS]
    X_all_sc = pd.DataFrame(
        splits["scaler"].transform(X_all), columns=FEATURE_COLUMNS, index=X_all.index
    )

    ml_names = ["LR", "DT", "GBT", "RF", "XGB", "ANN"]
    fn_encoder = joblib.load(os.path.join(models_dir, "fn_label_encoder.joblib"))

    # -------------------------------------------------------------------------
    # 1. Table 3: ML Model Metrics (Training, Testing, All)
    # -------------------------------------------------------------------------
    table3_rows = []
    ml_all_preds = {}
    ml_test_preds = {}
    ml_fn_all_preds = {}

    for name in ml_names:
        pu_model = joblib.load(os.path.join(models_dir, f"pu_{name}.joblib"))
        fn_model = joblib.load(os.path.join(models_dir, f"fn_{name}.joblib"))

        if name == "ANN":
            p_tr = pu_model.predict(X_tr_sc)
            p_te = pu_model.predict(X_te_sc)
            p_all = pu_model.predict(X_all_sc)
        else:
            p_tr = pu_model.predict(X_tr)
            p_te = pu_model.predict(X_te)
            p_all = pu_model.predict(X_all)

        if name in ["LR", "ANN"]:
            fn_p_all = fn_encoder.inverse_transform(fn_model.predict(X_all_sc))
        else:
            fn_p_all = fn_encoder.inverse_transform(fn_model.predict(X_all))

        ml_all_preds[name] = p_all
        ml_test_preds[name] = p_te
        ml_fn_all_preds[name] = fn_p_all

        for split_label, yt, yp in [("Training", y_pu_tr, p_tr), ("Testing", y_pu_te, p_te), ("All", y_pu_all, p_all)]:
            m = compute_paper_metrics(yt, yp)
            table3_rows.append({"Model": name, "Split": split_label, **m})

    table3_df = pd.DataFrame(table3_rows)
    table3_df.to_csv(os.path.join(base_dir, "table3_ml_metrics.csv"), index=False)

    print("=" * 95)
    print("📊 TABLE 3: ML MODELS STATISTICAL PERFORMANCE (Bond Strength Pu in kN)")
    print("=" * 95)
    print(table3_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. Table 4: Design Guidelines & Analytical Model Metrics (All Dataset)
    # -------------------------------------------------------------------------
    table4_rows = []
    analytical_all_preds = {}
    for name, func in ANALYTICAL_MODELS.items():
        p_all = func(df)
        analytical_all_preds[name] = p_all
        m = compute_paper_metrics(y_pu_all, p_all)
        table4_rows.append({"Model": name, **m})

    table4_df = pd.DataFrame(table4_rows)
    table4_df.to_csv(os.path.join(base_dir, "table4_analytical_metrics.csv"), index=False)

    print("\n" + "=" * 95)
    print("📐 TABLE 4: DESIGN GUIDELINES & ANALYTICAL MODELS PERFORMANCE (All N=416)")
    print("=" * 95)
    print(table4_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 3. Table 5: Scatter Plot (SP), Frequency Error (FEP), Performance Ratio (PRP)
    # -------------------------------------------------------------------------
    table5_rows = []
    for name, p_all in {**analytical_all_preds, **ml_all_preds}.items():
        row = compute_table5_breakdown(y_pu_all, p_all, tol_kn=2.0, ratio_tol=0.08)
        table5_rows.append({"Model": name, **row})

    table5_df = pd.DataFrame(table5_rows)
    table5_df.to_csv(os.path.join(base_dir, "table5_plot_analysis.csv"), index=False)

    print("\n" + "=" * 95)
    print("📈 TABLE 5: GRAPHICAL DIAGNOSTIC BREAKDOWN (SP, FEP, PRP)")
    print("=" * 95)
    print(table5_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. Evaluate Derived Closed-Form Mathematical Formulas on Testing Set
    # -------------------------------------------------------------------------
    with open(os.path.join(models_dir, "derived_formulas.json"), "r") as f:
        formulas = json.load(f)

    # Power-law formula prediction on test set
    pw = formulas["nonlinear_power_law"]
    exps = [pw["exponents"][c] for c in FEATURE_COLUMNS]
    p_power_te = power_law_func(X_te.values, pw["alpha"], *exps)
    m_power_te = compute_paper_metrics(y_pu_te, p_power_te)

    # Failure mode accuracy
    fn_accuracies = {}
    for name, fn_p in ml_fn_all_preds.items():
        exact_acc = float(np.mean(fn_p == y_fn_all) * 100.0)
        fn_accuracies[name] = round(exact_acc, 2)

    # -------------------------------------------------------------------------
    # 5. Table S3 Per-Study Breakdown ([21]-[40])
    # -------------------------------------------------------------------------
    study_rows = []
    if "Ref_ID" in df.columns:
        xgb_all = ml_all_preds["XGB"]
        xgb_fn = ml_fn_all_preds["XGB"]
        test_idx_set = set(splits["X_test"].index.tolist())
        df_eval = df[["Ref_ID", "Year", "Study", "Pu_kN", "FN"]].copy()
        df_eval["XGB_Pu"] = xgb_all
        df_eval["XGB_FN"] = xgb_fn
        df_eval["is_test"] = df_eval.index.isin(test_idx_set)
        df_eval["ref_num"] = df_eval["Ref_ID"].str.strip("[]").astype(int)

        for _, grp in df_eval.sort_values("ref_num").groupby("ref_num", sort=True):
            yt_s = grp["Pu_kN"].values
            yp_s = grp["XGB_Pu"].values
            m_s = compute_paper_metrics(yt_s, yp_s)
            fn_acc_s = float(np.mean(grp["XGB_FN"].values == grp["FN"].values) * 100.0)
            study_rows.append({
                "Ref_ID": grp["Ref_ID"].iloc[0],
                "Year": int(grp["Year"].iloc[0]),
                "Study": grp["Study"].iloc[0],
                "N_total": int(len(grp)),
                "N_test": int(grp["is_test"].sum()),
                "Exp_Pu_Min": round(float(yt_s.min()), 2),
                "Exp_Pu_Max": round(float(yt_s.max()), 2),
                "Exp_Pu_Mean": round(float(yt_s.mean()), 2),
                "XGB_Pred_Mean": round(float(yp_s.mean()), 2),
                "XGB_R": m_s["R"],
                "XGB_R2": m_s["R2"],
                "XGB_MAE": m_s["MAE"],
                "XGB_RMSE": m_s["RMSE"],
                "FN_Match_Pct": round(fn_acc_s, 2),
            })
        pd.DataFrame(study_rows).to_csv(os.path.join(base_dir, "table_s3_study_metrics.csv"), index=False)

    print("\n" + "=" * 95)
    print("🔍 FAILURE MODE (FN 1..16) EXACT CLASSIFICATION ALIGNMENT (All N=416)")
    for name, acc in fn_accuracies.items():
        print(f"  {name:<5}: {acc:.2f}% exact match")
    print(f"\n📐 Empirical Nonlinear Power-Law Formula (Testing Split R² = {m_power_te['R2']}, R = {m_power_te['R']}, RMSE = {m_power_te['RMSE']} kN)")
    print("=" * 95)

    # Save full summary JSON
    summary = {
        "dataset_size": int(len(df)),
        "train_size": int(len(X_tr)),
        "test_size": int(len(X_te)),
        "experimental_pu_mean": round(float(np.mean(y_pu_all)), 2),
        "experimental_pu_sd": round(float(np.std(y_pu_all, ddof=1)), 2),
        "table3_ml": table3_rows,
        "table4_analytical": table4_rows,
        "table5_diagnostics": table5_rows,
        "table_s3_study_metrics": study_rows,
        "failure_mode_accuracy_pct": fn_accuracies,
        "empirical_power_law_test_metrics": m_power_te,
        "derived_formulas": formulas,
    }
    with open(os.path.join(base_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return summary, splits, ml_all_preds, analytical_all_preds, ml_fn_all_preds


if __name__ == "__main__":
    summary, splits, ml_all_preds, analytical_all_preds, ml_fn_all_preds = evaluate_all()
    from src.visualizations import generate_all_diagrams
    generate_all_diagrams(summary, splits, ml_all_preds, analytical_all_preds, ml_fn_all_preds)
