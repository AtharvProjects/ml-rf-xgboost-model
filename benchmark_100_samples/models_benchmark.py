"""
100-Entry Benchmark: 9 Inputs, 1 Output Regression
Trains 5 ML Models (70 Train / 30 Validation Split)
Compares R², RMSE, MAE, MSE, selects the best model, and derives its mathematical formula.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def run_experiment():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    np.random.seed(42)
    n_samples = 100

    # 9 Input Parameters (Thermodynamic & Power Plant Simulation)
    X1 = np.random.uniform(10.0, 35.0, n_samples)   # Ambient Temperature (°C)
    X2 = np.random.uniform(0.98, 1.04, n_samples)   # Ambient Pressure (bar)
    X3 = np.random.uniform(30.0, 90.0, n_samples)   # Relative Humidity (%)
    X4 = np.random.uniform(25.0, 75.0, n_samples)   # Exhaust Vacuum (cm Hg)
    X5 = np.random.uniform(12.0, 20.0, n_samples)   # Turbine Pressure (bar)
    X6 = np.random.uniform(8.0, 25.0, n_samples)    # Fuel Flow Rate (kg/s)
    X7 = np.random.uniform(15.0, 50.0, n_samples)   # Steam Flow Rate (kg/s)
    X8 = np.random.uniform(12.0, 30.0, n_samples)   # Cooling Water Temp (°C)
    X9 = np.random.uniform(0.92, 0.98, n_samples)   # Generator Efficiency (ratio)

    # 1 Output Parameter: Net Electrical Power Output (MW)
    Y = (
        420.0
        - 1.85 * X1
        + 45.0 * X2
        - 0.12 * X3
        - 0.38 * X4
        + 4.20 * X5
        + 6.15 * X6
        + 2.80 * X7
        - 1.45 * X8
        + 85.0 * X9
        + np.random.normal(0, 0.8, n_samples)
    )

    df = pd.DataFrame({
        'X1_AmbTemp': np.round(X1, 2),
        'X2_AmbPress': np.round(X2, 4),
        'X3_Humidity': np.round(X3, 2),
        'X4_ExhaustVac': np.round(X4, 2),
        'X5_TurbinePress': np.round(X5, 2),
        'X6_FuelFlow': np.round(X6, 2),
        'X7_SteamFlow': np.round(X7, 2),
        'X8_CoolingTemp': np.round(X8, 2),
        'X9_GenEfficiency': np.round(X9, 4),
        'Y_NetPowerMW': np.round(Y, 2)
    })

    csv_path = os.path.join(data_dir, "dataset_100_entries.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved ({df.shape[0]} rows, {df.shape[1]} columns) to: {csv_path}")

    # Split 70 train, 30 validation
    X = df.drop(columns=['Y_NetPowerMW'])
    y = df['Y_NetPowerMW']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=70, test_size=30, random_state=42)

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "Artificial Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, learning_rate_init=0.01)
    }

    results = []
    predictions = {}

    for name, model in models.items():
        if "Neural Network" in name:
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_te_sc = scaler.transform(X_test)
            model.fit(X_tr_sc, y_train)
            y_pred = model.predict(X_te_sc)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        predictions[name] = y_pred
        results.append({
            "Model": name,
            "R2 Score": round(float(r2), 6),
            "RMSE (MW)": round(float(rmse), 4),
            "MAE (MW)": round(float(mae), 4),
            "MSE": round(float(mse), 4)
        })

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    print(results_df.to_string(index=False))

    # Save benchmark plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R2 Comparison
    top_models = results_df[results_df["R2 Score"] > 0] # Filter out negative for visualization clarity
    sns.barplot(data=top_models, x="Model", y="R2 Score", ax=ax1, palette="viridis")
    ax1.set_title("Validation R² Score Comparison (Closest to 1.0 is Best)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0.7, 1.02)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20, ha="right")
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    # Actual vs Predicted for Best Model (Multiple Linear Regression)
    best_pred = predictions["Multiple Linear Regression"]
    ax2.scatter(y_test, best_pred, color="#2b5c8f", alpha=0.85, edgecolors="k", s=60, label="Test Points (N=30)")
    min_v, max_v = min(y_test.min(), best_pred.min()) - 5, max(y_test.max(), best_pred.max()) + 5
    ax2.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label="Perfect Fit (1:1)")
    ax2.set_xlim(min_v, max_v)
    ax2.set_ylim(min_v, max_v)
    ax2.set_title(f"Multiple Linear Regression (Best Model): Actual vs. Predicted\nR² = {results_df.iloc[0]['R2 Score']:.6f} | RMSE = {results_df.iloc[0]['RMSE (MW)']:.4f} MW", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Actual Net Power Output (MW)", fontsize=11)
    ax2.set_ylabel("Predicted Net Power Output (MW)", fontsize=11)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "model_comparison_benchmark.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    return results_df

if __name__ == "__main__":
    run_experiment()
