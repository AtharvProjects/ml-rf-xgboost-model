"""
Analytical Models, Design Guidelines (Table 1 of Kumar et al., 2024),
and Statistical Evaluation Criteria (Section 3.2, Table 3 & Table 5).
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def predict_aci_440_2r_08(df: pd.DataFrame) -> np.ndarray:
    """ACI 440.2R-08 NSM FRP bond strength equation (Table 1)."""
    db = df["Df_mm"].values
    af = np.pi * (db ** 2) / 4.0
    ffd = 0.70 * df["ff_MPa"].values  # Environmental/design reduction factor
    rho_f = np.pi * db
    tau_avg = 6.90  # MPa
    ld = (af * ffd) / (rho_f * tau_avg)
    lb = df["Lb_mm"].values
    pu_n = np.where(lb >= ld, af * ffd, af * ffd * (lb / np.maximum(ld, 1e-6)))
    return pu_n / 1000.0  # kN


def predict_sa_hb305(df: pd.DataFrame) -> np.ndarray:
    """Standards Australia (SA HB 305-2008) design guideline (Table 1)."""
    fc = df["fc_MPa"].values
    db = df["Df_mm"].values
    af = np.pi * (db ** 2) / 4.0
    ef = df["Ef_GPa"].values * 1000.0  # MPa
    ffd = 0.75 * df["ff_MPa"].values
    dg = df["Dg_mm"].values
    bg = df["Wg_mm"].values
    lb = df["Lb_mm"].values

    phi_per = (dg + 1.0) / (bg + 2.0)
    tau_max = (0.8 + 0.078 * phi_per) * (fc ** 0.6)
    delta_max = (0.73 * (phi_per ** 0.5) * (fc ** 0.67)) / np.maximum(tau_max, 1e-6)
    l_per = 2.0 * (dg + 1.0) + (bg + 2.0)
    # Dimensionally consistent development length Ld in mm: sqrt((delta_max * Ef * Af) / (tau_max * L_per))
    ld = (np.pi / 2.0) * np.sqrt((delta_max * ef * af) / np.maximum(tau_max * l_per, 1e-6))

    pu_base = np.sqrt(tau_max * delta_max) * np.sqrt(l_per * ef * af)
    pu_n = np.where(lb >= ld, pu_base, pu_base * (lb / np.maximum(ld, 1e-6)))
    pu_n = np.minimum(pu_n, af * ffd)
    return pu_n / 1000.0  # kN


def predict_aci_440_1r_15(df: pd.DataFrame) -> np.ndarray:
    """ACI 440.1R-15 bond stress equation converted to pullout force (Table 1)."""
    fc = df["fc_MPa"].values
    db = df["Df_mm"].values
    lb = df["Lb_mm"].values
    # Cover/spacing C approximated as 2.5 * db
    c_over_db = 2.5
    tau_b = (4.0 + 0.3 * c_over_db + 100.0 * (db / np.maximum(lb, 1e-6))) * 0.083 * np.sqrt(fc)
    pu_n = tau_b * np.pi * db * lb
    # Cap at FRP tensile capacity
    af = np.pi * (db ** 2) / 4.0
    pt_n = af * df["ff_MPa"].values
    return np.minimum(pu_n, pt_n) / 1000.0  # kN


def predict_csa_s806_12(df: pd.DataFrame) -> np.ndarray:
    """CSA S806-12 bond strength equation (Table 1 with K1..K5 factors)."""
    fc = df["fc_MPa"].values
    db = df["Df_mm"].values
    lb = df["Lb_mm"].values
    af = np.pi * (db ** 2) / 4.0
    frp_bt = df["FRP_BT"].values
    frp_st = df["FRP_ST"].values

    k1 = 1.0
    k2 = 1.0
    k3 = np.where(af <= 300.0, 0.8, 1.0)
    k4 = np.where(frp_bt == 1, 1.15, 1.0)  # Basalt/Aramid slightly higher factor vs C/G
    k5_map = {1: 1.05, 2: 1.08, 3: 1.00, 4: 1.00, 5: 1.00, 6: 1.05, 7: 1.05}
    k5 = np.array([k5_map[int(s)] for s in frp_st])

    dsc = np.maximum(2.5 * db, 20.0)
    denom = 1.15 * (k1 * k2 * k3 * k4 * k5) * np.pi * db
    tau_b = (dsc * np.sqrt(fc)) / np.maximum(denom, 1e-6)
    pu_n = tau_b * np.pi * db * lb
    pt_n = af * df["ff_MPa"].values
    return np.minimum(pu_n, pt_n) / 1000.0  # kN


def predict_zhang_et_al(df: pd.DataFrame) -> np.ndarray:
    """Zhang et al. (2014) analytical bond strength model (Table 1)."""
    fc = df["fc_MPa"].values
    db = df["Df_mm"].values
    af = np.pi * (db ** 2) / 4.0
    ef = df["Ef_GPa"].values * 1000.0  # MPa
    pt_n = af * df["ff_MPa"].values
    dg = df["Dg_mm"].values
    wg = df["Wg_mm"].values
    lb = df["Lb_mm"].values

    ratio = np.maximum(dg / np.maximum(wg, 1e-6), 0.1)
    tau_max = 1.15 * (fc ** 0.613) * (ratio ** 0.138)
    l_per = 2.0 * dg + wg
    gf = 0.245 * (fc ** 0.632) * (ratio ** 0.785)

    eta = np.sqrt(((tau_max ** 2) * l_per) / np.maximum(2.0 * gf * af * ef, 1e-6))
    le = 1.66 / np.maximum(eta, 1e-6)
    beta_l = (lb / np.maximum(le, 1e-6)) * (2.08 - 1.08 * (lb / np.maximum(le, 1e-6)))
    beta_l = np.clip(beta_l, 0.05, 1.05)

    pu_max = np.sqrt(2.0 * gf * af * ef * l_per)
    pu_n = np.where(lb >= le, pu_max, beta_l * pu_max)
    pu_n = np.minimum(pu_n, pt_n)
    return pu_n / 1000.0  # kN


ANALYTICAL_MODELS = {
    "ACI 440.2R-08": predict_aci_440_2r_08,
    "SA": predict_sa_hb305,
    "ACI 440.1R-15": predict_aci_440_1r_15,
    "Zhang et al.": predict_zhang_et_al,
    "CSA-S806": predict_csa_s806_12,
}


def compute_paper_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes exact evaluation indices used in Table 3 & Table 4 of Kumar et al. (2024):
    R, R2, NS (Nash-Sutcliffe), a20-index, MAE, RMSE, MAPE, SI (Scatter Index), SD.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r_val, _ = pearsonr(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)

    # Nash-Sutcliffe efficiency (NS)
    denom_ns = np.sum((y_true - np.mean(y_true)) ** 2)
    ns_val = 1.0 - np.sum((y_true - y_pred) ** 2) / max(denom_ns, 1e-12)

    # a20-index: fraction of samples where 0.80 <= E/P <= 1.20
    ratio = y_true / np.where(np.abs(y_pred) < 1e-6, 1e-6, y_pred)
    a20_val = float(np.mean((ratio >= 0.80) & (ratio <= 1.20)))

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)
    si = rmse / max(float(np.mean(y_true)), 1e-6)
    sd = float(np.std(y_pred, ddof=1))

    return {
        "R": round(float(r_val), 4),
        "R2": round(float(r2_val), 4),
        "NS": round(float(ns_val), 4),
        "a20_index": round(a20_val, 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "SI": round(float(si), 4),
        "SD": round(sd, 2),
    }


def compute_table5_breakdown(y_true: np.ndarray, y_pred: np.ndarray, tol_kn: float = 1.5, ratio_tol: float = 0.08) -> dict:
    """
    Computes Table 5 graphical breakdown percentages:
    - SP (Scatter Plot): % above, on (within +-tol_kn), and below best-fitted line
    - FEP (Frequency Error Plot): min, max, mean error (Exp - Pred)
    - PRP (Performance Ratio Plot): % over-estimated, best-fitted, under-estimated (Pred/Exp)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Scatter plot tolerance (within +-tol_kn of y_pred == y_true)
    diff_pred_exp = y_pred - y_true
    on_line = np.abs(diff_pred_exp) <= tol_kn
    above_line = diff_pred_exp > tol_kn
    below_line = diff_pred_exp < -tol_kn

    # Frequency Error Plot: Error = Exp - Pred (matching Fig. 4b-8b)
    err = y_true - y_pred

    # Performance Ratio Plot: Pred / Exp
    pr = y_pred / np.maximum(y_true, 1e-6)
    pr_best = np.abs(pr - 1.0) <= ratio_tol
    pr_over = pr > (1.0 + ratio_tol)
    pr_under = pr < (1.0 - ratio_tol)

    return {
        "SP_above_pct": round(float(np.mean(above_line) * 100.0), 2),
        "SP_on_pct": round(float(np.mean(on_line) * 100.0), 2),
        "SP_below_pct": round(float(np.mean(below_line) * 100.0), 2),
        "FEP_min": round(float(np.min(err)), 2),
        "FEP_max": round(float(np.max(err)), 2),
        "FEP_mean": round(float(np.mean(err)), 2),
        "PRP_over_pct": round(float(np.mean(pr_over) * 100.0), 2),
        "PRP_best_pct": round(float(np.mean(pr_best) * 100.0), 2),
        "PRP_under_pct": round(float(np.mean(pr_under) * 100.0), 2),
    }
