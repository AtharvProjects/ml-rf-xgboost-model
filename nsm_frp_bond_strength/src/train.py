"""
Model Training Pipeline for NSM FRP Bond Strength (Pu) & Failure Mode (FN)
Trains 6 ML architectures with hyperparameter grid search (Section 3.3):
  1. LR  (Multiple Linear Regression)
  2. DT  (Decision Tree)
  3. GBT (Gradient Boosting Tree)
  4. RF  (Random Forest)
  5. XGB (Extreme Gradient Boosting - XGBoost)
  6. ANN (Artificial Neural Network / MLP)
Also derives explicit closed-form mathematical equations (Multiple Linear Regression
and calibrated Nonlinear Power-Law Empirical Equation) for bond strength Pu.
"""

import os
import json
import time
import joblib
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier

from src.preprocess import prepare_splits, FEATURE_COLUMNS


def power_law_func(X, a, b1, b2, b3, b4, b5, b6, b7, b8, b9):
    """Empirical nonlinear power-law formula over the 9 inputs."""
    fc, lb, df_bar, ef, ff, wg, dg, bt, st = X.T
    return (
        a
        * np.power(np.maximum(fc, 1e-3), b1)
        * np.power(np.maximum(lb, 1e-3), b2)
        * np.power(np.maximum(df_bar, 1e-3), b3)
        * np.power(np.maximum(ef, 1e-3), b4)
        * np.power(np.maximum(ff, 1e-3), b5)
        * np.power(np.maximum(wg, 1e-3), b6)
        * np.power(np.maximum(dg, 1e-3), b7)
        * np.power(np.maximum(bt, 1e-3), b8)
        * np.power(np.maximum(st, 1e-3), b9)
    )


def train_all_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    splits = prepare_splits(test_size=0.30, random_state=42, save_scaler=True)
    X_tr = splits["X_train"]
    X_tr_sc = splits["X_train_scaled"]
    y_pu_tr = splits["y_pu_train"]
    y_fn_tr = splits["y_fn_train"]

    print("=" * 75)
    print("🚀 Training 6 ML Models for Bond Strength (Pu) & Failure Mode (FN)")
    print(f"   Training split: {len(X_tr)} (70%) | Testing split: {len(splits['X_test'])} (30%)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # A. REGRESSION MODELS FOR BOND STRENGTH Pu (kN)
    # -------------------------------------------------------------------------
    pu_models = {}

    # 1. LR (Unscaled for direct algebraic formula + scaled check)
    t0 = time.time()
    lr = LinearRegression()
    lr.fit(X_tr, y_pu_tr)
    pu_models["LR"] = lr
    print(f"  [1/6] LR  (Linear Regression) trained in {time.time() - t0:.2f}s | Train R² = {lr.score(X_tr, y_pu_tr):.4f}")

    # 2. DT (GridSearchCV)
    t0 = time.time()
    dt_gs = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid={"max_depth": [5, 7, 9, 12], "min_samples_split": [2, 4, 6], "min_samples_leaf": [1, 2, 3]},
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    dt_gs.fit(X_tr, y_pu_tr)
    pu_models["DT"] = dt_gs.best_estimator_
    print(f"  [2/6] DT  (Decision Tree)     trained in {time.time() - t0:.2f}s | Best params: {dt_gs.best_params_} | Train R² = {pu_models['DT'].score(X_tr, y_pu_tr):.4f}")

    # 3. GBT (GridSearchCV)
    t0 = time.time()
    gbt_gs = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid={"n_estimators": [100, 150, 200], "learning_rate": [0.05, 0.08], "max_depth": [3, 4], "subsample": [0.85]},
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    gbt_gs.fit(X_tr, y_pu_tr)
    pu_models["GBT"] = gbt_gs.best_estimator_
    print(f"  [3/6] GBT (Gradient Boosting) trained in {time.time() - t0:.2f}s | Best params: {gbt_gs.best_params_} | Train R² = {pu_models['GBT'].score(X_tr, y_pu_tr):.4f}")

    # 4. RF (GridSearchCV)
    t0 = time.time()
    rf_gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid={"n_estimators": [150, 250], "max_depth": [10, 14, None], "min_samples_leaf": [1, 2]},
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    rf_gs.fit(X_tr, y_pu_tr)
    pu_models["RF"] = rf_gs.best_estimator_
    print(f"  [4/6] RF  (Random Forest)     trained in {time.time() - t0:.2f}s | Best params: {rf_gs.best_params_} | Train R² = {pu_models['RF'].score(X_tr, y_pu_tr):.4f}")

    # 5. XGBoost (GridSearchCV)
    t0 = time.time()
    xgb_gs = GridSearchCV(
        XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=-1),
        param_grid={
            "n_estimators": [120, 200],
            "learning_rate": [0.05, 0.08],
            "max_depth": [3, 4, 5],
            "subsample": [0.85],
            "colsample_bytree": [0.85, 1.0],
        },
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    xgb_gs.fit(X_tr, y_pu_tr)
    pu_models["XGB"] = xgb_gs.best_estimator_
    print(f"  [5/6] XGB (XGBoost)           trained in {time.time() - t0:.2f}s | Best params: {xgb_gs.best_params_} | Train R² = {pu_models['XGB'].score(X_tr, y_pu_tr):.4f}")

    # 6. ANN (MLPRegressor on [-1, 1] scaled features)
    t0 = time.time()
    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.05,
        learning_rate_init=0.01,
        max_iter=2500,
        early_stopping=False,
        random_state=42,
    )
    ann.fit(X_tr_sc, y_pu_tr)
    pu_models["ANN"] = ann
    print(f"  [6/6] ANN (Neural Network)    trained in {time.time() - t0:.2f}s | Epochs: {ann.n_iter_} | Train R² = {ann.score(X_tr_sc, y_pu_tr):.4f}")

    for name, mdl in pu_models.items():
        joblib.dump(mdl, os.path.join(models_dir, f"pu_{name}.joblib"))

    # -------------------------------------------------------------------------
    # B. FAILURE MODE (FN) PREDICTION MODELS (Fig. 16 & GUI)
    # -------------------------------------------------------------------------
    print("\nTraining Failure Mode (FN 1..16) Classifiers...")
    fn_encoder = LabelEncoder()
    y_fn_enc = fn_encoder.fit_transform(y_fn_tr)
    joblib.dump(fn_encoder, os.path.join(models_dir, "fn_label_encoder.joblib"))

    fn_models = {
        "LR": LogisticRegression(max_iter=1500, random_state=42),
        "DT": DecisionTreeClassifier(max_depth=8, min_samples_leaf=2, random_state=42),
        "GBT": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        "RF": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=250, learning_rate=0.07, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.1, max_iter=2000, early_stopping=False, random_state=42),
    }

    for name, clf in fn_models.items():
        if name in ["LR", "ANN"]:
            clf.fit(X_tr_sc, y_fn_enc)
        else:
            clf.fit(X_tr, y_fn_enc)
        joblib.dump(clf, os.path.join(models_dir, f"fn_{name}.joblib"))
    print("  ✓ All 6 Failure Mode classifiers serialized.")

    # -------------------------------------------------------------------------
    # C. DERIVED CLOSED-FORM MATHEMATICAL FORMULAS FOR Pu
    # -------------------------------------------------------------------------
    # 1. Multiple Linear Regression coefficients
    lr_formula = {
        "intercept": round(float(lr.intercept_), 4),
        "coefficients": {col: round(float(c), 4) for col, c in zip(FEATURE_COLUMNS, lr.coef_)},
    }

    # 2. Calibrated Nonlinear Power-Law Empirical Equation via curve_fit
    p0 = [0.2, 0.3, 0.5, 0.3, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05]
    popt, _ = curve_fit(power_law_func, X_tr.values, y_pu_tr.values, p0=p0, maxfev=20000)
    power_formula = {
        "alpha": round(float(popt[0]), 5),
        "exponents": {col: round(float(exp), 4) for col, exp in zip(FEATURE_COLUMNS, popt[1:])},
    }

    formulas = {"linear_regression": lr_formula, "nonlinear_power_law": power_formula}
    with open(os.path.join(models_dir, "derived_formulas.json"), "w") as f:
        json.dump(formulas, f, indent=4)

    print("\nDerived Nonlinear Power-Law Empirical Formula:")
    exps = power_formula["exponents"]
    print(
        f"  Pu = {power_formula['alpha']} * "
        f"(fc)^{exps['fc_MPa']} * (Lb)^{exps['Lb_mm']} * (Df)^{exps['Df_mm']} * "
        f"(Ef)^{exps['Ef_GPa']} * (ff)^{exps['ff_MPa']} * (Wg)^{exps['Wg_mm']} * "
        f"(Dg)^{exps['Dg_mm']} * (FRP_BT)^{exps['FRP_BT']} * (FRP_ST)^{exps['FRP_ST']}"
    )
    print("=" * 75)


if __name__ == "__main__":
    train_all_models()
