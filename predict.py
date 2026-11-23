"""
Inference script for predicting customer churn using trained Random Forest & XGBoost models.
"""

import os
import argparse
import joblib
import pandas as pd
from src.preprocess import FEATURE_COLUMNS

def predict_single(data_dict: dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    rf_path = os.path.join(models_dir, "random_forest_model.joblib")
    xgb_path = os.path.join(models_dir, "xgboost_model.joblib")
    
    scaler = joblib.load(scaler_path)
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)
    
    df_sample = pd.DataFrame([data_dict])[FEATURE_COLUMNS]
    scaled_features = scaler.transform(df_sample)
    
    rf_pred = int(rf_model.predict(scaled_features)[0])
    rf_prob = float(rf_model.predict_proba(scaled_features)[0][1])
    
    xgb_pred = int(xgb_model.predict(scaled_features)[0])
    xgb_prob = float(xgb_model.predict_proba(scaled_features)[0][1])
    
    return {
        "Random Forest": {
            "prediction": "Churn (1)" if rf_pred == 1 else "Retained (0)",
            "churn_probability": round(rf_prob, 4)
        },
        "XGBoost": {
            "prediction": "Churn (1)" if xgb_pred == 1 else "Retained (0)",
            "churn_probability": round(xgb_prob, 4)
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict customer churn using Random Forest & XGBoost")
    parser.add_argument("--credit_score", type=int, default=620)
    parser.add_argument("--age", type=int, default=45)
    parser.add_argument("--tenure", type=int, default=2)
    parser.add_argument("--balance", type=float, default=85000.0)
    parser.add_argument("--num_products", type=int, default=1)
    parser.add_argument("--has_credit_card", type=int, default=1)
    parser.add_argument("--is_active_member", type=int, default=0)
    parser.add_argument("--estimated_salary", type=float, default=95000.0)
    parser.add_argument("--satisfaction_score", type=int, default=2)
    
    args = parser.parse_args()
    
    input_data = {
        "credit_score": args.credit_score,
        "age": args.age,
        "tenure": args.tenure,
        "balance": args.balance,
        "num_products": args.num_products,
        "has_credit_card": args.has_credit_card,
        "is_active_member": args.is_active_member,
        "estimated_salary": args.estimated_salary,
        "satisfaction_score": args.satisfaction_score
    }
    
    print("\n--- Input Customer Profile ---")
    for k, v in input_data.items():
        print(f"  {k:20s}: {v}")
    
    results = predict_single(input_data)
    print("\n--- Model Predictions ---")
    for model_name, res in results.items():
        print(f"[{model_name}]")
        print(f"  Prediction       : {res['prediction']}")
        print(f"  Churn Probability: {res['churn_probability'] * 100:.1f}%\n")
