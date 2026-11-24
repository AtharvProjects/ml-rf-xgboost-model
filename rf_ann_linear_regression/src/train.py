"""
Model Training Script for Housing Price Valuation
Trains three diverse regression algorithms:
1. Linear Regression (baseline parametric)
2. Random Forest Regressor (ensemble decision trees)
3. Artificial Neural Network (Multi-Layer Perceptron / MLPRegressor)
"""

import os
import time
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from preprocess import preprocess_data, get_data_paths, FEATURE_COLUMNS

def train_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    
    print("=" * 60)
    print("🚀 Starting Training: Linear Regression, Random Forest & ANN")
    print("=" * 60)
    
    # Preprocess and split data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        test_size=0.2, 
        random_state=42, 
        save_scaler_path=scaler_path
    )
    print(f"Dataset split: {len(X_train)} training records, {len(X_test)} testing records.\n")
    
    # -------------------------------------------------------------------------
    # 1. Linear Regression
    # -------------------------------------------------------------------------
    print("1. Training Linear Regression (OLS)...")
    lr_start = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_time = time.time() - lr_start
    lr_train_r2 = lr_model.score(X_train, y_train)
    print(f"   ✓ Linear Regression completed in {lr_time:.4f}s | Train R²: {lr_train_r2:.4f}")
    
    # -------------------------------------------------------------------------
    # 2. Random Forest Regressor
    # -------------------------------------------------------------------------
    print("2. Training Random Forest Regressor...")
    rf_start = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - rf_start
    rf_train_r2 = rf_model.score(X_train, y_train)
    print(f"   ✓ Random Forest completed in {rf_time:.4f}s | Train R²: {rf_train_r2:.4f}")
    
    # -------------------------------------------------------------------------
    # 3. Artificial Neural Network (ANN - Multi-Layer Perceptron)
    # -------------------------------------------------------------------------
    print("3. Training Artificial Neural Network (ANN / MLPRegressor)...")
    ann_start = time.time()
    ann_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=16,
        learning_rate="adaptive",
        learning_rate_init=0.005,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30
    )
    ann_model.fit(X_train, y_train)
    ann_time = time.time() - ann_start
    ann_train_r2 = ann_model.score(X_train, y_train)
    print(f"   ✓ ANN completed in {ann_time:.4f}s ({ann_model.n_iter_} iterations) | Train R²: {ann_train_r2:.4f}")
    
    # -------------------------------------------------------------------------
    # Save Model Artifacts
    # -------------------------------------------------------------------------
    lr_path = os.path.join(models_dir, "linear_regression_model.joblib")
    rf_path = os.path.join(models_dir, "random_forest_regressor.joblib")
    ann_path = os.path.join(models_dir, "ann_mlp_regressor.joblib")
    
    joblib.dump(lr_model, lr_path)
    joblib.dump(rf_model, rf_path)
    joblib.dump(ann_model, ann_path)
    
    print("\n" + "=" * 60)
    print(f"💾 All models and scaler successfully serialized to: {models_dir}")
    print(f"  - Linear Regression : {os.path.basename(lr_path)}")
    print(f"  - Random Forest     : {os.path.basename(rf_path)}")
    print(f"  - ANN (MLP)         : {os.path.basename(ann_path)}")
    print(f"  - Feature Scaler    : {os.path.basename(scaler_path)}")
    print("=" * 60)

if __name__ == "__main__":
    train_models()
