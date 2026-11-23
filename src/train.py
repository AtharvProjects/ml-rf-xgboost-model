"""
Training module for Random Forest and XGBoost classifiers.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.preprocess import load_and_preprocess_data

def train_models():
    print("Loading and preprocessing dataset...")
    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler, feature_cols = load_and_preprocess_data()
    print(f"Dataset split: Train shape = {X_train.shape}, Test shape = {X_test.shape}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Random Forest Classifier
    print("\n[1/2] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_path = os.path.join(models_dir, "random_forest_model.joblib")
    joblib.dump(rf_model, rf_path)
    print(f"Random Forest model saved to: {rf_path}")

    # 2. XGBoost Classifier
    print("\n[2/2] Training XGBoost Classifier...")
    # Calculate scale_pos_weight for handling potential class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)

    xgb_model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_path = os.path.join(models_dir, "xgboost_model.joblib")
    joblib.dump(xgb_model, xgb_path)
    print(f"XGBoost model saved to: {xgb_path}")

    return rf_model, xgb_model, (X_test_scaled, y_test, feature_cols)

if __name__ == "__main__":
    train_models()
