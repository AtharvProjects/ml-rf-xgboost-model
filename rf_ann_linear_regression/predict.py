"""
Interactive CLI Prediction Engine for Housing Price Valuation
Accepts feature parameters and computes valuation estimates from all three models:
Linear Regression, Random Forest, and Artificial Neural Network (ANN).
"""

import os
import argparse
import joblib
import pandas as pd
from src.preprocess import FEATURE_COLUMNS

def load_models_and_scaler():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    lr_path = os.path.join(models_dir, "linear_regression_model.joblib")
    rf_path = os.path.join(models_dir, "random_forest_regressor.joblib")
    ann_path = os.path.join(models_dir, "ann_mlp_regressor.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    
    for path in [lr_path, rf_path, ann_path, scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model or scaler artifact missing: {path}. Run train.py first.")
            
    lr_model = joblib.load(lr_path)
    rf_model = joblib.load(rf_path)
    ann_model = joblib.load(ann_path)
    scaler = joblib.load(scaler_path)
    
    return lr_model, rf_model, ann_model, scaler

def predict_property_value(features: dict):
    lr_model, rf_model, ann_model, scaler = load_models_and_scaler()
    
    # Format input into single-row DataFrame
    input_df = pd.DataFrame([features])[FEATURE_COLUMNS]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURE_COLUMNS)
    
    # Predictions
    pred_lr = float(lr_model.predict(input_scaled_df)[0])
    pred_rf = float(rf_model.predict(input_scaled_df)[0])
    pred_ann = float(ann_model.predict(input_scaled_df)[0])
    pred_ensemble = (pred_lr + pred_rf + pred_ann) / 3.0
    
    print("\n" + "=" * 65)
    print("🏡 PROPERTY PROFILE INPUT SPECIFICATIONS")
    print("=" * 65)
    for col, val in features.items():
        if isinstance(val, float):
            print(f"  {col:<28}: {val:,.2f}")
        else:
            print(f"  {col:<28}: {val}")
            
    print("\n" + "=" * 65)
    print("💰 MODEL VALUATION PREDICTIONS")
    print("=" * 65)
    print(f"  [1] Linear Regression (OLS)        : ${pred_lr:,.2f}")
    print(f"  [2] Random Forest Regressor        : ${pred_rf:,.2f}")
    print(f"  [3] Artificial Neural Network (ANN): ${pred_ann:,.2f}")
    print("-" * 65)
    print(f"  ★ Consensus Ensemble (Average)     : ${pred_ensemble:,.2f}")
    print("=" * 65 + "\n")
    
    return {
        "linear_regression": pred_lr,
        "random_forest": pred_rf,
        "ann": pred_ann,
        "ensemble_average": pred_ensemble
    }

def main():
    parser = argparse.ArgumentParser(description="Estimate Property Valuation using LR, RF, and ANN models.")
    parser.add_argument("--square_footage", type=int, default=2450, help="Living area in sq ft (e.g. 2450)")
    parser.add_argument("--bedrooms", type=int, default=4, help="Number of bedrooms (e.g. 4)")
    parser.add_argument("--bathrooms", type=float, default=3.0, help="Number of bathrooms (e.g. 3.0)")
    parser.add_argument("--lot_size_sqft", type=int, default=6500, help="Lot size in sq ft (e.g. 6500)")
    parser.add_argument("--property_age_years", type=int, default=12, help="Age of property in years (e.g. 12)")
    parser.add_argument("--distance_to_center_km", type=float, default=8.5, help="Distance to city center in km (e.g. 8.5)")
    parser.add_argument("--neighborhood_quality_score", type=float, default=8.2, help="Neighborhood score 1-10 (e.g. 8.2)")
    parser.add_argument("--has_garage", type=int, default=1, choices=[0, 1], help="Garage present: 1 for yes, 0 for no")
    parser.add_argument("--has_pool", type=int, default=1, choices=[0, 1], help="Pool present: 1 for yes, 0 for no")
    parser.add_argument("--annual_tax", type=float, default=8500.0, help="Annual property tax in USD (e.g. 8500.0)")
    
    args = parser.parse_args()
    features = vars(args)
    predict_property_value(features)

if __name__ == "__main__":
    main()
