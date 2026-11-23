"""
Evaluation module comparing Random Forest and XGBoost classifiers.
Generates metrics and visualization plots.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from src.preprocess import load_and_preprocess_data

def evaluate_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    rf_path = os.path.join(models_dir, "random_forest_model.joblib")
    xgb_path = os.path.join(models_dir, "xgboost_model.joblib")

    if not os.path.exists(rf_path) or not os.path.exists(xgb_path):
        raise FileNotFoundError("Models not found. Please run src/train.py first.")

    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    X_train_scaled, X_test_scaled, y_train, y_test, _, _, _, feature_cols = load_and_preprocess_data()

    models = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }

    metrics_results = {}
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        metrics_results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(auc), 4)
        }
        predictions[name] = y_pred
        probabilities[name] = y_prob

    # Save metrics JSON
    metrics_file = os.path.join(base_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_results, f, indent=4)

    # Print summary table
    df_metrics = pd.DataFrame(metrics_results).T
    print("\n" + "=" * 65)
    print("                MODEL EVALUATION COMPARISON")
    print("=" * 65)
    print(df_metrics.to_string())
    print("=" * 65)

    # Set visualization theme
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Plot Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if i == 0 else "Greens",
                    ax=axes[i], cbar=False,
                    xticklabels=["Retained (0)", "Churned (1)"],
                    yticklabels=["Retained (0)", "Churned (1)"])
        axes[i].set_title(f"{name} Confusion Matrix", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrices plot saved to: {cm_path}")

    # 2. Plot ROC Curves Comparison
    plt.figure(figsize=(8, 6))
    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = metrics_results[name]["ROC_AUC"]
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve Comparison: Random Forest vs XGBoost", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    roc_path = os.path.join(plots_dir, "roc_curve_comparison.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"ROC curve plot saved to: {roc_path}")

    # 3. Feature Importance Comparison
    rf_feat_imp = rf_model.feature_importances_
    xgb_feat_imp = xgb_model.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Random Forest": rf_feat_imp,
        "XGBoost": xgb_feat_imp
    }).sort_values(by="Random Forest", ascending=False)

    df_imp_melted = df_imp.melt(id_vars="Feature", var_name="Model", value_name="Importance")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp_melted, x="Importance", y="Feature", hue="Model", palette=["#2b5c8f", "#2ca02c"])
    plt.title("Feature Importance: Random Forest vs XGBoost", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.legend(title="Algorithm", fontsize=10)
    plt.tight_layout()

    feat_path = os.path.join(plots_dir, "feature_importance_comparison.png")
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f"Feature importance comparison plot saved to: {feat_path}")

    return metrics_results

if __name__ == "__main__":
    evaluate_models()
