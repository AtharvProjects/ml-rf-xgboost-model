"""
Evaluation and Visualization Script for Regression Models
Compares Linear Regression, Random Forest, and ANN across multiple metrics:
R², RMSE, MAE, and MAPE.
Generates publication-quality comparison charts and outputs metrics.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from preprocess import preprocess_data, FEATURE_COLUMNS

def evaluate_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "plots")
    metrics_path = os.path.join(base_dir, "metrics.json")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load dataset test split
    X_train, X_test, y_train, y_test, _ = preprocess_data(test_size=0.2, random_state=42)
    
    # Load trained models
    lr_model = joblib.load(os.path.join(models_dir, "linear_regression_model.joblib"))
    rf_model = joblib.load(os.path.join(models_dir, "random_forest_regressor.joblib"))
    ann_model = joblib.load(os.path.join(models_dir, "ann_mlp_regressor.joblib"))
    
    models = {
        "Linear Regression": lr_model,
        "Random Forest": rf_model,
        "Artificial Neural Network (ANN)": ann_model
    }
    
    predictions = {}
    metrics = {}
    
    print("=" * 70)
    print("📊 Evaluating Models on Held-out Test Set (N = %d)" % len(y_test))
    print("=" * 70)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100.0)
        
        metrics[name] = {
            "r2_score": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape_percent": round(mape, 2)
        }
        
        print(f"[{name}]")
        print(f"  R² Score : {r2:.4f}")
        print(f"  RMSE     : ${rmse:,.2f}")
        print(f"  MAE      : ${mae:,.2f}")
        print(f"  MAPE     : {mape:.2f}%")
        print("-" * 50)
        
    # Save metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Configure plotting style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # -------------------------------------------------------------------------
    # Plot 1: Actual vs. Predicted Parity Plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ["#2b5c8f", "#2e7d32", "#c2185b"]
    
    min_val = min(y_test.min(), min(p.min() for p in predictions.values())) * 0.95
    max_val = max(y_test.max(), max(p.max() for p in predictions.values())) * 1.05
    
    for ax, (name, y_pred), color in zip(axes, predictions.items(), colors):
        ax.scatter(y_test, y_pred, color=color, alpha=0.75, edgecolors="k", s=55, label="Test Records")
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal 1:1 Fit")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f"{name}\n$R^2 = {metrics[name]['r2_score']:.4f}$ | MAE = ${metrics[name]['mae']:,.0f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual Price ($ USD)", fontsize=11)
        ax.set_ylabel("Predicted Price ($ USD)", fontsize=11)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)
        
    plt.suptitle("Actual vs. Predicted Property Valuation Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "actual_vs_predicted.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # -------------------------------------------------------------------------
    # Plot 2: Residual Error Distribution Plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, y_pred), color in zip(axes, predictions.items(), colors):
        residuals = y_test - y_pred
        sns.histplot(residuals, kde=True, ax=ax, color=color, bins=15, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2)
        ax.set_title(f"{name} Residuals\nMean: ${residuals.mean():,.0f} | Std: ${residuals.std():,.0f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Residual (Actual - Predicted) [$ USD]", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        
    plt.suptitle("Residual Error Distribution (Check for Bias & Normality)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "residuals_distribution.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot2_path}")
    
    # -------------------------------------------------------------------------
    # Plot 3: Metrics Benchmark Bar Chart
    # -------------------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    model_labels = list(metrics.keys())
    short_labels = ["Linear Reg", "Random Forest", "ANN"]
    bar_colors = ["#3498db", "#2ecc71", "#e74c3c"]
    
    # R2
    r2_vals = [metrics[m]["r2_score"] for m in model_labels]
    bars1 = ax1.bar(short_labels, r2_vals, color=bar_colors, edgecolor="black", width=0.5)
    ax1.set_title("R² Score (Higher is Better)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
        
    # RMSE
    rmse_vals = [metrics[m]["rmse"] for m in model_labels]
    bars2 = ax2.bar(short_labels, rmse_vals, color=bar_colors, edgecolor="black", width=0.5)
    ax2.set_title("RMSE in USD (Lower is Better)", fontsize=12, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval * 1.01, f"${yval:,.0f}", ha="center", va="bottom", fontweight="bold")
        
    # MAE
    mae_vals = [metrics[m]["mae"] for m in model_labels]
    bars3 = ax3.bar(short_labels, mae_vals, color=bar_colors, edgecolor="black", width=0.5)
    ax3.set_title("MAE in USD (Lower is Better)", fontsize=12, fontweight="bold")
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval * 1.01, f"${yval:,.0f}", ha="center", va="bottom", fontweight="bold")
        
    # MAPE
    mape_vals = [metrics[m]["mape_percent"] for m in model_labels]
    bars4 = ax4.bar(short_labels, mape_vals, color=bar_colors, edgecolor="black", width=0.5)
    ax4.set_title("MAPE % (Lower is Better)", fontsize=12, fontweight="bold")
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")
        
    plt.suptitle("Cross-Model Regression Performance Benchmark", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plot3_path = os.path.join(plots_dir, "model_metrics_comparison.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot3_path}")
    
    # -------------------------------------------------------------------------
    # Plot 4: Random Forest Feature Importance & Linear Regression Coefficients
    # -------------------------------------------------------------------------
    fig, (ax_rf, ax_lr) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest Importance
    rf_importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    rf_importances.plot(kind="barh", ax=ax_rf, color="#27ae60", edgecolor="black")
    ax_rf.set_title("Random Forest Feature Importance (MDI)", fontsize=12, fontweight="bold")
    ax_rf.set_xlabel("Relative Importance", fontsize=11)
    
    # Linear Regression Coefficients
    lr_coefs = pd.Series(lr_model.coef_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    lr_colors = ["#e74c3c" if c < 0 else "#2980b9" for c in lr_coefs]
    lr_coefs.plot(kind="barh", ax=ax_lr, color=lr_colors, edgecolor="black")
    ax_lr.set_title("Linear Regression Standardized Coefficients", fontsize=12, fontweight="bold")
    ax_lr.set_xlabel("Coefficient Impact on Price ($ USD)", fontsize=11)
    
    plt.suptitle("Feature Impact Comparison: Non-Linear Trees vs. Linear Weights", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot4_path = os.path.join(plots_dir, "feature_importance_comparison.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot4_path}")
    
    # -------------------------------------------------------------------------
    # Plot 5: ANN Training Loss Curve
    # -------------------------------------------------------------------------
    if hasattr(ann_model, "loss_curve_"):
        plt.figure(figsize=(8, 5))
        plt.plot(ann_model.loss_curve_, color="#c2185b", linewidth=2.5, label="Adam Loss")
        plt.title(f"Artificial Neural Network (ANN) Training Loss Curve ({ann_model.n_iter_} Epochs)", fontsize=12, fontweight="bold")
        plt.xlabel("Iteration / Epoch", fontsize=11)
        plt.ylabel("Squared Loss (MSE)", fontsize=11)
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plot5_path = os.path.join(plots_dir, "ann_loss_curve.png")
        plt.savefig(plot5_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot5_path}")
        
    print("=" * 70)
    print("✅ All evaluation metrics and visual artifacts generated successfully!")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_models()
