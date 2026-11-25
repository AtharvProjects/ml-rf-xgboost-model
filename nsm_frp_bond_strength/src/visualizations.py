"""
Publication-Grade Visualization Suite Replicating All Key Diagrams from Kumar et al. (2024):
  - Fig 1 : Variable Distributions (11 Donut Charts + Cyan Min/Max/Mean/SD Summary Boxes)
  - Fig 2 : Categorical Contribution Donut Charts (FRP Bar Type, Surface Type, Failure Mode)
  - Fig 4-9 : 3-Panel Diagnostic Plots per ML Model (Scatter, Error Frequency, Performance Ratio)
  - Fig 9-13: 3-Panel Diagnostic Grid for Analytical/Design Guidelines
  - Fig 14: 7-Panel Radar/Spider Performance Comparison Charts
  - Fig 15: Two-Panel Polar Taylor Diagrams (Analytical vs ML Models)
  - Fig 16: Failure Mode Scatter Plots (FN_exp vs FN_pred)
  - Fig 17: SHAP Explainability Plots (Beeswarm Summary + Mean |SHAP| Feature Importance)
  - Fig 18: Partial Dependence Plots (3x3 PDP + ICE Grid for all 9 Inputs)
  - Fig 19: NSM FRP-to-Concrete Pullout Specimen Engineering Schematic Diagram
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.projections import PolarAxes
from sklearn.inspection import partial_dependence

from src.preprocess import FEATURE_COLUMNS
from src.analytical_models import compute_table5_breakdown


def plot_fig1_variable_distributions(df: pd.DataFrame, plots_dir: str):
    """Generates Fig. 1: 11 Donut Charts with Cyan Statistical Summary Boxes."""
    specs = [
        ("fc_MPa", r"(a) $f'_c$ (MPa)", [(23, 28.75), (28.75, 34), (34, 49.37), (49.37, 67)], ["23-28.75", "28.75-34", "34-49.37", "49.37-66.25"]),
        ("Lb_mm", r"(b) $L_b$ (mm)", [(14, 96), (96, 195), (195, 290), (290, 460)], ["14.4-96", "101.6-192", "200-271.2", "300-457.2"]),
        ("Df_mm", r"(c) $D_f$ (mm)", [(4.5, 8.5), (8.5, 10.5), (10.5, 12.5), (12.5, 25)], ["5-8", "9-10.28", "11-12", "12.7-24"]),
        ("Ef_GPa", r"(d) $E_f$ (GPa)", [(37, 65), (65, 125), (125, 175), (175, 240)], ["37.17-65", "65-125", "125-175", "175-234.43"]),
        ("ff_MPa", r"(e) $f_f$ (MPa)", [(740, 1400), (1400, 2100), (2100, 2800), (2800, 4900)], ["749-1400", "1400-2100", "2100-2800", "2800-4826.5"]),
        ("Wg_mm", r"(f) $W_g$ (mm)", [(5, 13.5), (13.5, 17), (17, 22), (22, 31)], ["6-13.5", "14-17", "17-22", "22-30"]),
        ("Dg_mm", r"(g) $D_g$ (mm)", [(9, 14.5), (14.5, 18), (18, 24), (24, 41)], ["10-14.5", "15-18", "18-24", "24-40"]),
        ("FRP_BT", r"(h) $FRP_{BT}$", [1, 2, 3], ["1 (Basalt)", "2 (Carbon)", "3 (Glass)"]),
        ("FRP_ST", r"(i) $FRP_{ST}$", [(0.5, 2.5), (2.5, 4.5), (4.5, 6.5), (6.5, 7.5)], ["1-2", "3-4", "5-6", "7"]),
        ("Pu_kN", r"(j) $P_u$ (kN)", [(7, 26), (26, 42), (42, 70), (70, 130)], ["7.17-26", "26-42", "42-70", "70-127.52"]),
        ("FN", r"(k) $F_N$", [(0.5, 5.5), (5.5, 9.5), (9.5, 12.5), (12.5, 16.5)], ["1-5", "6-9", "10-12", "13-16"]),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15, 17))
    axes_flat = axes.flatten()
    palette = ["#26b800", "#ff9900", "#ffea00", "#cc0000"]

    for i, (col, title, bins, labels) in enumerate(specs):
        ax = axes_flat[i]
        vals = df[col].values
        if isinstance(bins[0], int):
            counts = [np.sum(vals == b) for b in bins]
        else:
            counts = [np.sum((vals >= lo) & (vals < hi)) for lo, hi in bins]

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=palette[: len(counts)],
            wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.2),
            textprops=dict(fontsize=8.5),
            pctdistance=0.73,
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")

        stat_text = f"Min. = {vals.min():.2f}; Max. = {vals.max():.2f}\nMean = {vals.mean():.2f}; SD = {vals.std(ddof=1):.2f}"
        ax.text(
            0.5,
            -0.08,
            stat_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#00bcd4", edgecolor="#006064", alpha=0.9),
        )
        ax.set_title(f"{title}", fontsize=11, fontweight="bold", y=-0.28)

    axes_flat[-1].axis("off")
    plt.suptitle("Fig. 1: Distribution of Input and Output Variables with Statistical Parameters (N = 416)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(plots_dir, "fig1_variable_distributions.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out)}")


def plot_fig2_categorical_contributions(df: pd.DataFrame, plots_dir: str):
    """Generates Fig. 2: Categorical Contribution Donut Charts."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # (a) FRP Bar Types
    bt_counts = df["FRP_BT"].value_counts().sort_index()
    bt_labels = [f"{k}\n({bt_counts.get(i, 0)/len(df)*100:.2f}%)" for i, k in [(1, "Basalt (B)"), (2, "Carbon (C)"), (3, "Glass (G)")]]
    axes[0].pie([bt_counts.get(1, 0), bt_counts.get(2, 0), bt_counts.get(3, 0)], labels=bt_labels, colors=["#2e7d32", "#1565c0", "#ba68c8"], wedgeprops=dict(width=0.45, edgecolor="w"))
    axes[0].set_title("(a) Contribution of FRP Bar Types", fontweight="bold")

    # (b) Surface Types
    st_names = {1: "DB/RIB", 2: "GB", 3: "RUF", 4: "SMTH", 5: "SB/SC", 6: "SW/HB", 7: "Hybrid"}
    st_counts = df["FRP_ST"].value_counts().sort_index()
    st_lbls = [f"{st_names[k]} ({v/len(df)*100:.1f}%)" for k, v in st_counts.items()]
    axes[1].pie(st_counts.values, labels=st_lbls, colors=plt.cm.Set2.colors[:len(st_counts)], wedgeprops=dict(width=0.45, edgecolor="w"), textprops={"fontsize": 8.5})
    axes[1].set_title("(b) Contribution of Bar Surface Types", fontweight="bold")

    # (c) Failure Modes (1..16)
    fn_counts = df["FN"].value_counts().sort_index()
    fn_lbls = [f"F{k} ({v/len(df)*100:.1f}%)" for k, v in fn_counts.items()]
    axes[2].pie(fn_counts.values, labels=fn_lbls, colors=plt.cm.tab20.colors[:len(fn_counts)], wedgeprops=dict(width=0.45, edgecolor="w"), textprops={"fontsize": 8})
    axes[2].set_title("(c) Failure Mode Pattern Distribution", fontweight="bold")

    plt.suptitle("Fig. 2: Categorical Contributions of FRP Bars, Surfaces, and Failure Modes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(plots_dir, "fig2_categorical_contributions.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out)}")


def _draw_3panel_diagnostic(ax_row, y_true: np.ndarray, y_pred: np.ndarray, title_prefix: str, tol_kn: float = 2.0, ratio_tol: float = 0.08):
    """Draws exact 3-panel layout matching Figs. 4-13: (a) Scatter, (b) Error Freq, (c) Perf Ratio."""
    stats = compute_table5_breakdown(y_true, y_pred, tol_kn=tol_kn, ratio_tol=ratio_tol)
    ax1, ax2, ax3 = ax_row

    # Panel (a): Scatter Plot (Neon Pink above, Slate Purple on line, Dark Mint Green below)
    diff = y_pred - y_true
    on_mask = np.abs(diff) <= tol_kn
    above_mask = diff > tol_kn
    below_mask = diff < -tol_kn

    ax1.scatter(y_true[above_mask], y_pred[above_mask], color="#ff1493", s=24, alpha=0.8, label=f"{stats['SP_above_pct']:.2f}% Above")
    ax1.scatter(y_true[on_mask], y_pred[on_mask], color="#6a5acd", s=26, alpha=0.85, label=f"{stats['SP_on_pct']:.2f}% On line")
    ax1.scatter(y_true[below_mask], y_pred[below_mask], color="#00a86b", s=24, alpha=0.8, label=f"{stats['SP_below_pct']:.2f}% Below")
    lim_max = max(135, y_true.max() * 1.05, y_pred.max() * 1.05)
    ax1.plot([0, lim_max], [0, lim_max], "k--", linewidth=1.2)
    ax1.set_xlim(0, lim_max)
    ax1.set_ylim(0, lim_max)
    ax1.set_xlabel(r"$P_{u,exp}$ (kN)", fontsize=9.5)
    ax1.set_ylabel(r"$P_{u,pred}$ (kN)", fontsize=9.5)
    ax1.set_title(f"{title_prefix} (a) Scatter Plot", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel (b): Frequency Error Plot (Exp - Pred)
    err = y_true - y_pred
    ax2.hist(err, bins=12, color="#00c853", edgecolor="black", linewidth=0.7)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Error (kN)", fontsize=9.5)
    ax2.set_ylabel("Frequency", fontsize=9.5)
    ax2.set_title(f"{title_prefix} (b) Error Frequency", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel (c): Performance Ratio Plot (Pred / Exp vs Sample Index)
    idx = np.arange(1, len(y_true) + 1)
    pr = y_pred / np.maximum(y_true, 1e-6)
    pr_best = np.abs(pr - 1.0) <= ratio_tol
    pr_over = pr > (1.0 + ratio_tol)
    pr_under = pr < (1.0 - ratio_tol)

    ax3.scatter(idx[pr_over], pr[pr_over], color="#d50000", s=16, alpha=0.75, label=f"Over ({stats['PRP_over_pct']:.1f}%)")
    ax3.scatter(idx[pr_best], pr[pr_best], color="#007e33", s=18, alpha=0.85, label=f"Best ({stats['PRP_best_pct']:.1f}%)")
    ax3.scatter(idx[pr_under], pr[pr_under], color="#0000cd", s=16, alpha=0.75, label=f"Under ({stats['PRP_under_pct']:.1f}%)")
    ax3.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    ax3.set_xlabel("Number of dataset", fontsize=9.5)
    ax3.set_ylabel("Pred./Exp.", fontsize=9.5)
    ax3.set_title(f"{title_prefix} (c) Performance Ratio", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=7, ncol=1, frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.4)


def plot_diagnostic_grids(y_true: np.ndarray, ml_preds: dict, analytical_preds: dict, plots_dir: str):
    """Generates Figs. 4-9 (ML 3-panel plots) and Figs. 9-13 (Analytical 3-panel grid)."""
    # Individual ML 3-panel plots + combined ML grid
    fig_ml, axes_ml = plt.subplots(len(ml_preds), 3, figsize=(15, 3.6 * len(ml_preds)))
    for i, (name, yp) in enumerate(ml_preds.items()):
        _draw_3panel_diagnostic(axes_ml[i], y_true, yp, f"{name}:")
    plt.suptitle("ML Models Diagnostic Comparison: Scatter, Error Frequency, and Performance Ratio", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out_ml = os.path.join(plots_dir, "fig4_to_9_ml_models_diagnostics.png")
    plt.savefig(out_ml, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out_ml)}")

    # Highlight XGBoost standalone 3-panel plot (Fig. 8 replica)
    fig_xgb, axes_xgb = plt.subplots(1, 3, figsize=(15, 4.3))
    _draw_3panel_diagnostic(axes_xgb, y_true, ml_preds["XGB"], "XGBoost:")
    plt.suptitle("Fig. 8: Results of Best-Performing XGBoost Model (Scatter, Error Frequency, Performance Ratio)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_xgb = os.path.join(plots_dir, "fig8_xgb_diagnostics.png")
    plt.savefig(out_xgb, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out_xgb)}")

    # Analytical models 5-row grid (Figs. 9-13 replica)
    fig_an, axes_an = plt.subplots(len(analytical_preds), 3, figsize=(15, 3.6 * len(analytical_preds)))
    for i, (name, yp) in enumerate(analytical_preds.items()):
        _draw_3panel_diagnostic(axes_an[i], y_true, yp, f"{name}:")
    plt.suptitle("Design Guidelines & Analytical Models Diagnostic Comparison (Figs. 9-13)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out_an = os.path.join(plots_dir, "fig9_to_13_analytical_diagnostics.png")
    plt.savefig(out_an, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out_an)}")


def plot_fig14_radar_comparison(table3_df: pd.DataFrame, table4_df: pd.DataFrame, plots_dir: str):
    """Generates Fig. 14: 7-Panel Radar/Spider Performance Comparison Charts."""
    ml_all = table3_df[table3_df["Split"] == "All"][["Model", "R", "MAE", "MAPE", "RMSE", "NS", "a20_index", "SI"]]
    combined = pd.concat([table4_df[["Model", "R", "MAE", "MAPE", "RMSE", "NS", "a20_index", "SI"]], ml_all], ignore_index=True)

    models = combined["Model"].tolist()
    n_models = len(models)
    angles = np.linspace(0, 2 * np.pi, n_models, endpoint=False).tolist()
    angles += angles[:1]

    metrics_cfg = [
        ("R", "(a) R-value", "#00e5ff"),
        ("MAE", "(b) MAE (kN)", "#00c853"),
        ("MAPE", "(c) MAPE (%)", "#536dfe"),
        ("RMSE", "(d) RMSE (kN)", "#1de9b6"),
        ("NS", "(e) NS", "#aa00ff"),
        ("a20_index", "(f) a20-index", "#ffd600"),
        ("SI", "(g) SI", "#ff5252"),
    ]

    fig = plt.figure(figsize=(16, 15))
    for i, (col, title, color) in enumerate(metrics_cfg, 1):
        ax = fig.add_subplot(3, 3, i, polar=True)
        vals = combined[col].tolist()
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, color=color, linewidth=2)
        ax.fill(angles, vals_closed, color=color, alpha=0.45)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(models, fontsize=7.5)
        ax.tick_params(axis="x", pad=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=18)

    plt.suptitle("Fig. 14: Radar Comparison of ML Models, Design Guidelines, and Analytical Model", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(plots_dir, "fig14_radar_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out)}")


def _draw_taylor_panel(ax, exp_sd: float, models_stats: list, title: str):
    """Draws a quadrant polar Taylor Diagram (SD, R, and centered RMSE arcs)."""
    max_sd = max(exp_sd * 1.45, max(s["SD"] for s in models_stats) * 1.2)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim(0, max_sd)

    # Correlation angle ticks
    r_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
    ax.set_xticks(np.arccos(r_ticks))
    ax.set_xticklabels([str(r) for r in r_ticks], fontsize=8)
    ax.set_xlabel("SD (kN)", fontsize=9.5, labelpad=18)
    ax.set_ylabel("SD (kN)", fontsize=9.5, labelpad=22)

    # Reference experimental arc & star
    theta_arc = np.linspace(0, np.pi / 2, 200)
    ax.plot(theta_arc, np.full_like(theta_arc, exp_sd), "k:", linewidth=1.2, label="Ref SD")
    ax.plot(0.0, exp_sd, marker="*", markersize=14, color="black", linestyle="None", label="Reference (Exp)")

    # Centered RMSE contour circles around (exp_sd, 0)
    rs, ts = np.meshgrid(np.linspace(0, max_sd, 150), np.linspace(0, np.pi / 2, 150))
    rms_grid = np.sqrt(exp_sd ** 2 + rs ** 2 - 2 * exp_sd * rs * np.cos(ts))
    levels = np.linspace(5, max_sd * 0.9, 4).round(0)
    cs = ax.contour(ts, rs, rms_grid, levels=levels, colors="#1e88e5", linestyles="--", linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.0f kN")

    colors = plt.cm.tab10.colors
    for idx, st in enumerate(models_stats):
        r_val = np.clip(st["R"], -1.0, 1.0)
        theta = np.arccos(max(0.0, r_val))
        ax.plot(theta, st["SD"], "o", markersize=8, color=colors[idx % len(colors)], markeredgecolor="k", label=st["Model"])

    ax.set_title(title, fontsize=11, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=7.5)


def plot_fig15_taylor_diagrams(exp_sd: float, table3_df: pd.DataFrame, table4_df: pd.DataFrame, plots_dir: str):
    """Generates Fig. 15: Two-Panel Taylor Plots (a) Analytical vs (b) ML Models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={"projection": "polar"})

    an_stats = table4_df[["Model", "R", "SD"]].to_dict("records")
    ml_stats = table3_df[table3_df["Split"] == "All"][["Model", "R", "SD"]].to_dict("records")

    _draw_taylor_panel(axes[0], exp_sd, an_stats, "(a) Design Guidelines & Analytical Model")
    _draw_taylor_panel(axes[1], exp_sd, ml_stats, "(b) Developed Machine Learning Models")

    plt.suptitle("Fig. 15: Taylor Diagrams Comparing Standard Deviation, Correlation (R), and Centered RMSE", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(plots_dir, "fig15_taylor_plots.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out)}")


def plot_fig16_failure_mode_scatter(y_fn_true: np.ndarray, fn_preds: dict, plots_dir: str):
    """Generates Fig. 16: Scatter Plots of Predicted vs Experimental Failure Mode FN."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for ax, (name, yp) in zip(axes_flat, fn_preds.items()):
        above = np.mean(yp > y_fn_true) * 100.0
        on_line = np.mean(yp == y_fn_true) * 100.0
        below = np.mean(yp < y_fn_true) * 100.0

        ax.scatter(y_fn_true[yp > y_fn_true], yp[yp > y_fn_true], color="#ff1493", s=26, alpha=0.8, label=f"{above:.2f}% Above")
        ax.scatter(y_fn_true[yp == y_fn_true], yp[yp == y_fn_true], color="#00a86b", s=30, alpha=0.85, label=f"{on_line:.2f}% Exact Match")
        ax.scatter(y_fn_true[yp < y_fn_true], yp[yp < y_fn_true], color="#6a5acd", s=26, alpha=0.8, label=f"{below:.2f}% Below")
        ax.plot([0, 17], [0, 17], "k--", linewidth=1.1)
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 17)
        ax.set_xticks(range(0, 19, 3))
        ax.set_yticks(range(0, 19, 3))
        ax.set_xlabel(r"$F_{N,exp}$", fontsize=10)
        ax.set_ylabel(r"$F_{N,pred}$", fontsize=10)
        ax.set_title(f"{name} Failure Mode", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Fig. 16: Scatter Plot of Bond Failure Modes (FN_exp vs FN_pred)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(plots_dir, "fig16_failure_mode_scatter.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out)}")


def plot_fig17_shap_analysis(xgb_model, X_data: pd.DataFrame, plots_dir: str):
    """Generates Fig. 17: Exact TreeSHAP Beeswarm Summary & Mean |SHAP| Importance Bar Plot."""
    dmat = xgb.DMatrix(X_data)
    contribs = xgb_model.get_booster().predict(dmat, pred_contribs=True)
    shap_vals = contribs[:, :-1]  # Exclude bias column

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs_shap)  # Ascending so top feature is at top of horizontal plot
    ordered_cols = [FEATURE_COLUMNS[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1.0]})

    # (a) Beeswarm-style SHAP Summary Scatter
    cmap = plt.cm.coolwarm
    for pos, feat_idx in enumerate(order):
        sv = shap_vals[:, feat_idx]
        fv = X_data.iloc[:, feat_idx].values
        fv_norm = (fv - fv.min()) / max(fv.max() - fv.min(), 1e-9)
        jitter = np.random.default_rng(42).normal(0, 0.12, size=len(sv))
        sc = ax1.scatter(sv, np.full_like(sv, pos) + jitter, c=fv_norm, cmap=cmap, s=16, alpha=0.8)

    ax1.axvline(0, color="gray", linestyle="-", linewidth=0.9)
    ax1.set_yticks(range(len(ordered_cols)))
    ax1.set_yticklabels(ordered_cols, fontsize=10, fontweight="bold")
    ax1.set_xlabel("SHAP value (impact on model output Pu in kN)", fontsize=10)
    ax1.set_title("(a) SHAP Summary Plot", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.4)
    cbar = plt.colorbar(sc, ax=ax1, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=9.5)

    # (b) Mean |SHAP value| Horizontal Bar Chart
    ax2.barh(range(len(ordered_cols)), mean_abs_shap[order], color="#1e88e5", edgecolor="black", height=0.65)
    ax2.set_yticks(range(len(ordered_cols)))
    ax2.set_yticklabels(ordered_cols, fontsize=10, fontweight="bold")
    ax2.set_xlabel("mean(|SHAP value|) (average impact magnitude in kN)", fontsize=10)
    ax2.set_title("(b) Global Feature Importance", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.suptitle("Fig. 17: Feature Importance & Explainability via TreeSHAP (XGBoost)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(plots_dir, "fig17_shap_analysis.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out)}")


def plot_fig18_pdp_plots(xgb_model, X_data: pd.DataFrame, plots_dir: str):
    """Generates Fig. 18: 3x3 Partial Dependence Plots (PDP + ICE curves) for all 9 inputs."""
    labels_map = {
        "fc_MPa": r"(a) $f'_c$ (MPa)",
        "Lb_mm": r"(b) $L_b$ (mm)",
        "Df_mm": r"(c) $D_f$ (mm)",
        "Ef_GPa": r"(d) $E_f$ (GPa)",
        "ff_MPa": r"(e) $f_f$ (MPa)",
        "Wg_mm": r"(f) $W_g$ (mm)",
        "Dg_mm": r"(g) $D_g$ (mm)",
        "FRP_BT": r"(h) $FRP_{BT}$",
        "FRP_ST": r"(i) $FRP_{ST}$",
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes_flat = axes.flatten()
    X_float = X_data.astype(float)

    for ax, col in zip(axes_flat, FEATURE_COLUMNS):
        res = partial_dependence(xgb_model, X_float, features=[col], kind="both", grid_resolution=35)
        grid_vals = res["grid_values"][0]
        ice_lines = res["individual"][0]
        pdp_avg = res["average"][0]

        # Plot subset of ICE curves in light blue (matching Fig. 18)
        for row_idx in range(0, len(ice_lines), 4):
            ax.plot(grid_vals, ice_lines[row_idx], color="#90caf9", alpha=0.35, linewidth=0.7)
        ax.plot(grid_vals, pdp_avg, color="#0d47a1", linewidth=2.4, label="Average PDP")
        ax.set_xlabel(labels_map[col], fontsize=10.5, fontweight="bold")
        ax.set_ylabel(r"$P_u$ (kN)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Fig. 18: Partial Dependence Plots (PDP & ICE) of Bond Strength Pu Across All 9 Inputs", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(plots_dir, "fig18_pdp_plots.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out)}")


def generate_specimen_schematic(plots_dir: str):
    """Generates clean engineering schematic of the NSM FRP-to-concrete joint (Fig. 19 visual)."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Concrete block
    concrete = patches.Rectangle((1.2, 0.6), 4.0, 2.8, linewidth=1.8, edgecolor="#424242", facecolor="#e0e0e0", label="Concrete block")
    ax.add_patch(concrete)
    ax.text(3.2, 0.85, r"Concrete Block ($f'_c$)", ha="center", fontsize=10, fontweight="bold", color="#212121")

    # Epoxy groove
    epoxy = patches.Rectangle((1.2, 2.3), 2.8, 0.8, linewidth=1.2, edgecolor="#880e4f", facecolor="#f8bbd0", alpha=0.85)
    ax.add_patch(epoxy)
    ax.text(2.6, 2.05, r"Epoxy Groove ($W_g \times D_g$)", ha="center", fontsize=9, fontweight="bold", color="#880e4f")

    # FRP Rod embedded in epoxy + sticking out for pullout
    frp_rod = patches.Rectangle((1.2, 2.58), 5.6, 0.24, linewidth=1.5, edgecolor="#1b5e20", facecolor="#4caf50")
    ax.add_patch(frp_rod)
    ax.text(5.32, 3.05, r"FRP Bar ($D_f, E_f, f_f$)", ha="left", fontsize=9.5, fontweight="bold", color="#1b5e20")

    # Bonded length dimension arrow
    ax.annotate("", xy=(1.2, 3.65), xytext=(4.0, 3.65), arrowprops=dict(arrowstyle="<->", color="blue", lw=1.6))
    ax.text(2.6, 3.8, r"Bonded Length ($L_b$)", ha="center", fontsize=10, fontweight="bold", color="blue")

    # Pull-out Load Pu arrow
    ax.annotate("", xy=(9.0, 2.7), xytext=(6.8, 2.7), arrowprops=dict(arrowstyle="->", color="#d50000", lw=3.0))
    ax.text(7.1, 3.05, r"Pull-out Load $P_u$ (kN)", ha="left", fontsize=10.5, fontweight="bold", color="#d50000")

    # Summary box on right
    info_text = (
        "9 Dependent Input Parameters:\n"
        "1. Concrete Strength (f'c, MPa)\n"
        "2. Bonded Length (Lb, mm)\n"
        "3. FRP Bar Diameter (Df, mm)\n"
        "4. Elastic Modulus (Ef, GPa)\n"
        "5. Tensile Strength (ff, MPa)\n"
        "6. Groove Width (Wg, mm)\n"
        "7. Groove Depth (Dg, mm)\n"
        "8. FRP Bar Type (Basalt/Carbon/Glass)\n"
        "9. Bar Surface Type (1 to 7)"
    )
    ax.text(6.6, 0.55, info_text, fontsize=8.5, va="bottom", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e3f2fd", edgecolor="#1565c0"))
    ax.set_title("Near-Surface Mounted (NSM) FRP Rod-to-Concrete Bond Specimen Schematic", fontsize=12, fontweight="bold")

    out = os.path.join(plots_dir, "fig19_nsm_specimen_schematic.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out)}")


def plot_fig20_table_s3_study_breakdown(df: pd.DataFrame, study_metrics: list, plots_dir: str):
    """Generates Fig. 20: Table S3 Per-Study Specimen Range & XGBoost Error Breakdown ([21]-[40])."""
    if not study_metrics:
        return
    sm_df = pd.DataFrame(study_metrics)
    labels = [f"{r} {s.split('(')[0].strip()}" for r, s in zip(sm_df["Ref_ID"], sm_df["Study"])]
    y_pos = np.arange(len(sm_df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)

    # Panel (a): Experimental Pu [min, max] range + Exp Mean vs XGB Pred Mean
    for i, row in sm_df.iterrows():
        ax1.plot([row["Exp_Pu_Min"], row["Exp_Pu_Max"]], [i, i], color="#90caf9", linewidth=6, alpha=0.8, solid_capstyle="round")
    ax1.scatter(sm_df["Exp_Pu_Mean"], y_pos, color="#0d47a1", s=45, zorder=3, label="Experimental Mean Pu")
    ax1.scatter(sm_df["XGB_Pred_Mean"], y_pos, color="#ff6d00", marker="x", s=55, linewidths=2, zorder=4, label="XGBoost Predicted Mean Pu")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{lbl} (n={n})" for lbl, n in zip(labels, sm_df["N_total"])], fontsize=8.5, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel(r"Pull-out Bond Strength $P_u$ (kN)", fontsize=10.5)
    ax1.set_title("(a) Table S3 Experimental Pu Ranges & Mean Parity", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Panel (b): Per-Study XGBoost MAE (kN) & RMSE (kN)
    bar_h = 0.36
    ax2.barh(y_pos - bar_h / 2, sm_df["XGB_MAE"], height=bar_h, color="#00a86b", edgecolor="black", label="XGBoost MAE (kN)")
    ax2.barh(y_pos + bar_h / 2, sm_df["XGB_RMSE"], height=bar_h, color="#5c6bc0", edgecolor="black", label="XGBoost RMSE (kN)")
    ax2.set_xlabel("Error (kN)", fontsize=10.5)
    ax2.set_title("(b) Per-Study XGBoost Prediction Error ([21]–[40])", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.suptitle("Fig. 20: Table S3 Source Study Coverage & Per-Study XGBoost Accuracy ([21]–[40], N = 416)", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(plots_dir, "fig20_table_s3_study_breakdown.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {os.path.basename(out)}")


def generate_all_diagrams(summary: dict, splits: dict, ml_all_preds: dict, analytical_all_preds: dict, ml_fn_all_preds: dict):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = splits["df"]
    y_pu_all = df["Pu_kN"].values
    y_fn_all = df["FN"].values

    table3_df = pd.DataFrame(summary["table3_ml"])
    table4_df = pd.DataFrame(summary["table4_analytical"])
    xgb_model = joblib.load(os.path.join(models_dir, "pu_XGB.joblib"))

    print("\n🎨 Generating All Paper-Style Diagrams...")
    plot_fig1_variable_distributions(df, plots_dir)
    plot_fig2_categorical_contributions(df, plots_dir)
    plot_diagnostic_grids(y_pu_all, ml_all_preds, analytical_all_preds, plots_dir)
    plot_fig14_radar_comparison(table3_df, table4_df, plots_dir)
    plot_fig15_taylor_diagrams(summary["experimental_pu_sd"], table3_df, table4_df, plots_dir)
    plot_fig16_failure_mode_scatter(y_fn_all, ml_fn_all_preds, plots_dir)
    plot_fig17_shap_analysis(xgb_model, df[FEATURE_COLUMNS], plots_dir)
    plot_fig18_pdp_plots(xgb_model, df[FEATURE_COLUMNS], plots_dir)
    generate_specimen_schematic(plots_dir)
    plot_fig20_table_s3_study_breakdown(df, summary.get("table_s3_study_metrics", []), plots_dir)
    print("🎨 All diagrams successfully created!\n")
