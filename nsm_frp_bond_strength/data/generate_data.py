"""
Synthetic Datasheet Generator Matching NSM FRP Rod-to-Concrete Experimental Database
Follows the 9 input parameters, 2 output parameters (Pu in kN, FN failure mode 1-16),
sample count (N=416), statistical distributions (Min, Max, Mean, SD), and physical
mechanics reported in Kumar et al. (Journal of Building Engineering 90, 2024, 109418).
"""

import os
import numpy as np
import pandas as pd

FAILURE_DESCRIPTIONS = {
    1: ("A", "Epoxy adhesive splitting / cracking / longitudinal cracking of epoxy"),
    2: ("A+B", "Epoxy splitting + Debonding at the bar-epoxy interface"),
    3: ("A+B+E", "Epoxy splitting + Bar-epoxy debonding + Concrete cracking"),
    4: ("A+C+E", "Epoxy splitting + Tensile rupture of FRP + Concrete cracking"),
    5: ("A+D", "Epoxy splitting + Debonding at epoxy-concrete interface"),
    6: ("A+E", "Epoxy splitting + Concrete cracking / cover separation"),
    7: ("B", "Debonding at bar-epoxy interface / Pull-out / Slip of bar"),
    8: ("B+D", "Bar-epoxy debonding + Epoxy-concrete interface failure"),
    9: ("B+E", "Bar-epoxy debonding + Concrete cracking"),
    10: ("C", "Tensile failure of FRP reinforcement / Bar rupture / Fiber splitting"),
    11: ("D", "Debonding at epoxy-concrete interface / Concrete cover breakage"),
    12: ("D+E", "Epoxy-concrete debonding + Concrete cracking / splitting"),
    13: ("E", "Concrete cracking / fracture along inclined planes / cover separation"),
    14: ("E+G", "Concrete cracking + Bar surface damage"),
    15: ("F", "Shearing off the outer layer of the FRP bar"),
    16: ("G", "Bar surface damage"),
}

FRP_BAR_TYPES = {1: "Basalt (B)", 2: "Carbon (C)", 3: "Glass (G)"}

FRP_SURFACE_TYPES = {
    1: "DB/RIB (Deformed/Ribbed bar)",
    2: "GB (Grooved bar)",
    3: "RUF (Rough bar)",
    4: "SMTH (Smooth bar)",
    5: "SB/SC (Sand blasting/Sand coating)",
    6: "SW/HB (Spirally wound/Helical bar)",
    7: "SB+H/SC+H/SW+SC (Hybrid sand-coated & helical)",
}


def generate_nsm_frp_dataset(n_samples: int = 416, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 1. FRP_BT (1=Basalt 13.0%, 2=Carbon 56.5%, 3=Glass 30.5%)
    frp_bt = rng.choice([1, 2, 3], size=n_samples, p=[0.130, 0.565, 0.305])

    # 2. FRP_ST (1..7 proportions matching Fig. 2b)
    st_probs = np.array([0.173, 0.067, 0.031, 0.180, 0.231, 0.077, 0.241])
    st_probs = st_probs / st_probs.sum()
    frp_st = rng.choice([1, 2, 3, 4, 5, 6, 7], size=n_samples, p=st_probs)

    # 3. fc (MPa): Range [23.0, 66.25], Mean ~37.90, SD ~11.62
    # Stratified sampling matching Fig. 1a bins
    fc_bins = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.358, 0.291, 0.243, 0.108])
    fc = np.empty(n_samples)
    for idx, b in enumerate(fc_bins):
        if b == 0:
            fc[idx] = rng.uniform(23.00, 28.75)
        elif b == 1:
            fc[idx] = rng.uniform(28.75, 34.00)
        elif b == 2:
            fc[idx] = rng.uniform(34.00, 49.37)
        else:
            fc[idx] = rng.uniform(52.30, 66.25)
    fc[0] = 23.00
    fc[-1] = 66.25
    fc = np.round(fc, 2)

    # 4. Lb (mm): Range [14.4, 457.2], Mean ~179.90, SD ~103.50
    lb_bins = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.317, 0.260, 0.163, 0.260])
    lb = np.empty(n_samples)
    for idx, b in enumerate(lb_bins):
        if b == 0:
            lb[idx] = rng.uniform(14.40, 96.00)
        elif b == 1:
            lb[idx] = rng.uniform(101.60, 192.00)
        elif b == 2:
            lb[idx] = rng.uniform(200.00, 271.20)
        else:
            lb[idx] = rng.uniform(300.00, 457.20)
    lb[1] = 14.40
    lb[-2] = 457.20
    lb = np.round(lb, 2)

    # 5. Df (mm): Range [5.0, 24.0], Mean ~9.52, SD ~3.21
    df_bins = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.488, 0.243, 0.188, 0.081])
    df_bar = np.empty(n_samples)
    for idx, b in enumerate(df_bins):
        if b == 0:
            df_bar[idx] = rng.choice([5.0, 6.0, 6.35, 7.5, 8.0])
        elif b == 1:
            df_bar[idx] = rng.choice([9.0, 9.5, 10.0, 10.28])
        elif b == 2:
            df_bar[idx] = rng.choice([11.0, 12.0])
        else:
            df_bar[idx] = rng.choice([12.7, 14.0, 16.0, 19.0, 22.0, 24.0])
    df_bar[2] = 5.0
    df_bar[-3] = 24.0
    df_bar = np.round(df_bar, 2)

    # 6. Ef (GPa) & ff (MPa): Correlated with FRP_BT (Carbon highest Ef & ff, Glass/Basalt lower)
    ef = np.empty(n_samples)
    ff = np.empty(n_samples)
    for i in range(n_samples):
        if frp_bt[i] == 2:  # Carbon
            ef[i] = rng.uniform(113.0, 234.43)
            ff[i] = rng.uniform(1650.0, 4826.50) if ef[i] > 155 else rng.uniform(1350.0, 2800.0)
        elif frp_bt[i] == 1:  # Basalt
            ef[i] = rng.uniform(45.0, 95.0)
            ff[i] = rng.uniform(1000.0, 1950.0)
        else:  # Glass
            ef[i] = rng.uniform(37.17, 75.0)
            ff[i] = rng.uniform(749.0, 1650.0)
    ef[3] = 37.17
    ef[-4] = 234.43
    ff[3] = 749.00
    ff[-4] = 4826.50
    ef = np.round(ef, 2)
    ff = np.round(ff, 2)

    # 7. Wg (mm) & Dg (mm): Groove width [6, 30] and depth [10, 40], physically >= Df
    wg = np.clip(df_bar * rng.uniform(1.3, 2.2, size=n_samples) + rng.normal(2.5, 2.0, size=n_samples), 6.0, 30.0)
    dg = np.clip(df_bar * rng.uniform(1.4, 2.4, size=n_samples) + rng.normal(2.0, 2.2, size=n_samples), 10.0, 40.0)
    wg[4], wg[-5] = 6.0, 30.0
    dg[4], dg[-5] = 10.0, 40.0
    wg = np.round(wg, 2)
    dg = np.round(dg, 2)

    # 8. Target Pu (kN): Nonlinear mechanics calibrated to Section 4.2 SHAP & PDP trends:
    # - Bonded length Lb dominates (with stepwise increments >50mm and >300mm, plateau 200-300mm)
    # - Tensile strength ff, diameter Df (rise 7-10mm, shear lag damping >12mm), concrete fc (>45MPa boost), Ef (>200GPa boost)
    # - Groove dimensions Wg, Dg, FRP_BT (Carbon>Basalt/Glass), FRP_ST (ribbed/sand-coated/hybrid slightly stronger)
    af = np.pi * (df_bar ** 2) / 4.0  # mm^2
    pt_kn = af * ff / 1000.0  # FRP tensile rupture capacity (kN)

    # Effective bonded length & nonlinear step response matching Fig. 18b
    lb_effect = (
        18.0 * np.power(lb / 100.0, 0.68)
        + 6.5 * (lb > 50.0).astype(float)
        + 6.0 * (lb > 300.0).astype(float)
        - 3.5 * np.maximum(0.0, np.minimum(lb, 300.0) - 200.0) / 100.0
    )

    # Diameter effect matching Fig. 18c (sharp rise 7-10mm, shear-lag moderation >11mm)
    df_effect = 1.0 + 0.35 * (1.0 / (1.0 + np.exp(-(df_bar - 8.5) / 1.2))) - 0.015 * np.maximum(0.0, df_bar - 11.0)

    # Concrete strength effect matching Fig. 18a (boost beyond 45 MPa)
    fc_effect = 0.72 + 0.008 * fc + 0.16 * (1.0 / (1.0 + np.exp(-(fc - 45.0) / 2.5)))

    # FRP tensile strength & modulus effects matching Fig. 18d, e
    ff_effect = 0.75 + 0.25 * (ff / 2000.0) ** 0.55
    ef_effect = 0.92 + 0.08 * (ef / 120.0) + 0.14 * (ef > 200.0).astype(float)

    # Bar type and surface adjustments matching Section 4.2.2.6
    bt_factor = np.where(frp_bt == 2, 1.08, np.where(frp_bt == 1, 0.96, 0.94))
    st_map = {1: 1.04, 2: 1.00, 3: 0.97, 4: 0.91, 5: 1.05, 6: 1.01, 7: 1.06}
    st_factor = np.array([st_map[s] for s in frp_st])

    # Groove geometry ratio factor
    groove_factor = 0.92 + 0.06 * np.clip(dg / np.maximum(wg, 1.0), 0.7, 1.6) + 0.002 * (dg - 15.0)

    raw_pu = lb_effect * df_effect * fc_effect * ff_effect * ef_effect * bt_factor * st_factor * groove_factor
    # Cap softly by FRP bar rupture capacity & add experimental testing noise
    pu = np.minimum(raw_pu, 0.92 * pt_kn + 5.0) + rng.normal(0.0, 1.35, size=n_samples)
    pu = np.clip(np.round(pu, 2), 7.17, 127.52)

    # 9. Target FN (Failure Mode 1..16 matching Fig. 2c and Fig. 3 physical mechanisms)
    # Physical heuristics:
    # - Low Lb & smooth bar (ST=4) -> 7 (B: bar-epoxy pull-out/slip)
    # - Very high Lb and/or low tensile capacity (Pu close to Pt) -> 10 (C: FRP rupture)
    # - High Pu & large groove/bar -> 11 (D: epoxy-concrete debonding), 12 (D+E), 13 (E: concrete splitting)
    # - Thin groove / high shear stress -> 1 (A: epoxy splitting), 2 (A+B), 6 (A+E)
    fn = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        u = rng.random()
        ratio_rupture = pu[i] / max(pt_kn[i], 1.0)
        if ratio_rupture > 0.78 and lb[i] > 240:
            fn[i] = 10 if u < 0.72 else 4  # C or A+C+E
        elif frp_st[i] == 4 and lb[i] < 160:
            fn[i] = 7 if u < 0.68 else (8 if u < 0.85 else 9)  # B, B+D, B+E
        elif wg[i] - df_bar[i] < 4.5 and pu[i] > 35:
            fn[i] = 1 if u < 0.55 else (6 if u < 0.82 else 2)  # A, A+E, A+B
        elif fc[i] < 32 and pu[i] > 42:
            fn[i] = 13 if u < 0.52 else (12 if u < 0.75 else 6)  # E, D+E, A+E
        elif pu[i] > 45:
            fn[i] = 11 if u < 0.58 else (13 if u < 0.80 else (14 if u < 0.90 else 15))  # D, E, E+G, F
        else:
            # Realistic mix matching remaining minor failure patterns
            fn[i] = rng.choice([1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 16], p=[0.12, 0.05, 0.02, 0.03, 0.10, 0.16, 0.06, 0.06, 0.24, 0.04, 0.11, 0.01])

    df = pd.DataFrame({
        "fc_MPa": fc,
        "Lb_mm": lb,
        "Df_mm": df_bar,
        "Ef_GPa": ef,
        "ff_MPa": ff,
        "Wg_mm": wg,
        "Dg_mm": dg,
        "FRP_BT": frp_bt,
        "FRP_ST": frp_st,
        "Pu_kN": pu,
        "FN": fn,
    })
    return df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "nsm_frp_bond_dataset.csv")
    df = generate_nsm_frp_dataset(n_samples=416, seed=42)
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} NSM FRP-to-concrete pullout records -> {out_path}")
    stats = df.describe().T[["count", "min", "max", "mean", "std"]]
    print("\nDataset Statistical Summary:")
    print(stats.round(2).to_string())


if __name__ == "__main__":
    main()
