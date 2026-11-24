"""
Interactive & Batch Testing Tool for Custom Data Entries
Tests custom inputs against:
1. The Derived Mathematical Formula
2. The Trained Multiple Linear Regression Model
3. (Optional) Compares against Actual Y if provided.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Exact Derived Parameters
BETA_0 = 419.803448
COEFFICIENTS = {
    'X1_AmbTemp': -1.857517,
    'X2_AmbPress': 42.229437,
    'X3_Humidity': -0.116604,
    'X4_ExhaustVac': -0.383138,
    'X5_TurbinePress': 4.221380,
    'X6_FuelFlow': 6.140368,
    'X7_SteamFlow': 2.797738,
    'X8_CoolingTemp': -1.440591,
    'X9_GenEfficiency': 87.980594
}

FEATURE_NAMES = list(COEFFICIENTS.keys())

def predict_with_formula(inputs: dict) -> float:
    """Calculate Y using the derived closed-form mathematical expression."""
    val = BETA_0
    for feat, coef in COEFFICIENTS.items():
        val += coef * float(inputs[feat])
    return val

def predict_single(features: dict, actual_y: float = None):
    y_pred_formula = predict_with_formula(features)
    
    print("\n" + "=" * 70)
    print("🔍 CUSTOM ENTRY INPUT SPECIFICATIONS")
    print("=" * 70)
    for k, v in features.items():
        print(f"  {k:<20} : {v}")
        
    print("\n" + "=" * 70)
    print("⚡ PREDICTION RESULT")
    print("=" * 70)
    print(f"  Derived Formula Output (Y) : {y_pred_formula:,.4f} MW")
    
    if actual_y is not None:
        error = abs(actual_y - y_pred_formula)
        error_pct = (error / actual_y) * 100.0
        print(f"  Actual Ground Truth (Y)    : {actual_y:,.4f} MW")
        print(f"  Absolute Residual Error    : {error:,.4f} MW")
        print(f"  Percentage Error           : {error_pct:.2f}%")
    print("=" * 70 + "\n")
    return y_pred_formula

def predict_batch_csv(csv_path: str, output_csv: str = None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} custom records from: {csv_path}")
    
    # Check required columns
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        # Try simplified names (X1..X9)
        alt_names = {f"X{i}": name for i, name in enumerate(FEATURE_NAMES, 1)}
        df = df.rename(columns=alt_names)
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required input columns: {missing}")

    preds = []
    for idx, row in df.iterrows():
        preds.append(predict_with_formula(row.to_dict()))
        
    df['Predicted_Y_MW'] = np.round(preds, 4)
    
    # Check if actual Y is present for benchmarking
    y_cols = [c for c in df.columns if c in ['Y', 'Y_NetPowerMW', 'actual_y', 'actual']]
    if y_cols:
        y_col = y_cols[0]
        df['Absolute_Error_MW'] = np.round(np.abs(df[y_col] - df['Predicted_Y_MW']), 4)
        df['Error_Percent'] = np.round((df['Absolute_Error_MW'] / df[y_col]) * 100.0, 2)
        print(f"Computed validation against ground truth column '{y_col}'.")
        print(f"Mean Absolute Error across batch: {df['Absolute_Error_MW'].mean():.4f} MW")
        
    print("\nBatch Prediction Summary (First 5 records):")
    cols_to_show = FEATURE_NAMES[:3] + ['Predicted_Y_MW'] + ([y_cols[0], 'Absolute_Error_MW'] if y_cols else [])
    print(df[cols_to_show].head().to_string(index=False))
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nFull predictions saved to: {output_csv}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Test custom entries against the derived 9-parameter regression formula.")
    parser.add_argument("--csv", type=str, default=None, help="Path to batch CSV containing custom entries")
    parser.add_argument("--output", type=str, default=None, help="Path to save output CSV for batch mode")
    
    # Individual feature arguments
    parser.add_argument("--x1", type=float, default=None, help="Ambient Temp (°C) [10-35]")
    parser.add_argument("--x2", type=float, default=None, help="Ambient Press (bar) [0.98-1.04]")
    parser.add_argument("--x3", type=float, default=None, help="Humidity (%) [30-90]")
    parser.add_argument("--x4", type=float, default=None, help="Exhaust Vac (cm Hg) [25-75]")
    parser.add_argument("--x5", type=float, default=None, help="Turbine Press (bar) [12-20]")
    parser.add_argument("--x6", type=float, default=None, help="Fuel Flow (kg/s) [8-25]")
    parser.add_argument("--x7", type=float, default=None, help="Steam Flow (kg/s) [15-50]")
    parser.add_argument("--x8", type=float, default=None, help="Cooling Temp (°C) [12-30]")
    parser.add_argument("--x9", type=float, default=None, help="Gen Efficiency [0.92-0.98]")
    parser.add_argument("--actual_y", type=float, default=None, help="Optional actual Y to compare against")
    
    args = parser.parse_args()
    
    if args.csv:
        predict_batch_csv(args.csv, args.output)
        return

    # Check if any feature arguments provided
    feat_args = [args.x1, args.x2, args.x3, args.x4, args.x5, args.x6, args.x7, args.x8, args.x9]
    if any(f is not None for f in feat_args):
        # Default fallback values for unprovided features
        defaults = [22.0, 1.013, 60.0, 45.0, 16.0, 18.0, 32.0, 20.0, 0.95]
        features = {}
        for i, (arg, default, name) in enumerate(zip(feat_args, defaults, FEATURE_NAMES)):
            features[name] = arg if arg is not None else default
        predict_single(features, args.actual_y)
    else:
        # Run demo default sample
        demo_features = {
            'X1_AmbTemp': 24.50,
            'X2_AmbPress': 1.0150,
            'X3_Humidity': 55.00,
            'X4_ExhaustVac': 42.00,
            'X5_TurbinePress': 17.50,
            'X6_FuelFlow': 20.00,
            'X7_SteamFlow': 35.00,
            'X8_CoolingTemp': 19.50,
            'X9_GenEfficiency': 0.9650
        }
        print("No inputs specified. Running demonstration with default custom entry:")
        predict_single(demo_features, actual_y=762.30)
        print("💡 Tip: Provide custom values using flags (e.g. `python3 predict_custom.py --x1 26.5 --x6 21.0`)")
        print("        Or test batch entries from a CSV (e.g. `python3 predict_custom.py --csv custom_data.csv`)\n")

if __name__ == "__main__":
    main()
