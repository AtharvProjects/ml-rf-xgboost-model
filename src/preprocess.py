"""
Data preprocessing, feature engineering, and train-test splitting module.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "num_products",
    "has_credit_card",
    "is_active_member",
    "estimated_salary",
    "satisfaction_score"
]
TARGET_COLUMN = "churn"

def load_and_preprocess_data(data_path: str = None):
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "customer_churn_data.csv")
        
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # Train / Test split (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler, FEATURE_COLUMNS
