"""
Data Preprocessing Module for Housing Price Valuation
Handles dataset loading, feature scaling, train-test splitting, and serialization of the StandardScaler.
"""

import os
import joblib
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "lot_size_sqft",
    "property_age_years",
    "distance_to_center_km",
    "neighborhood_quality_score",
    "has_garage",
    "has_pool",
    "annual_tax"
]
TARGET_COLUMN = "price"

def get_data_paths() -> Tuple[str, str]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "housing_prices_data.csv")
    models_dir = os.path.join(base_dir, "models")
    return data_path, models_dir

def load_data(file_path: str = None) -> pd.DataFrame:
    if file_path is None:
        file_path, _ = get_data_paths()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Run generate_data.py first.")
    return pd.read_csv(file_path)

def preprocess_data(
    test_size: float = 0.2, 
    random_state: int = 42,
    save_scaler_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    df = load_data()
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    X_train = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS, index=X_test_raw.index)
    
    if save_scaler_path is not None:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)
        print(f"StandardScaler saved to: {save_scaler_path}")
        
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    _, models_dir = get_data_paths()
    scaler_out = os.path.join(models_dir, "scaler.joblib")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(save_scaler_path=scaler_out)
    print(f"Preprocessing complete.")
    print(f"Train samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[0]}, Features: {X_test.shape[1]}")
