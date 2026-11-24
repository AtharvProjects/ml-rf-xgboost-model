"""
Data Loading, Normalization (-1 to +1 per Section 3.1 of Kumar et al., 2024),
and 70% Training / 30% Testing Split.
"""

import os
import joblib
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    "fc_MPa",
    "Lb_mm",
    "Df_mm",
    "Ef_GPa",
    "ff_MPa",
    "Wg_mm",
    "Dg_mm",
    "FRP_BT",
    "FRP_ST",
]

TARGET_PU = "Pu_kN"
TARGET_FN = "FN"


def get_paths() -> Tuple[str, str]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "nsm_frp_bond_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    return data_path, models_dir


def load_dataset(file_path: str = None) -> pd.DataFrame:
    if file_path is None:
        file_path, _ = get_paths()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Run data/generate_data.py first.")
    return pd.read_csv(file_path)


def prepare_splits(test_size: float = 0.30, random_state: int = 42, save_scaler: bool = True):
    """
    Splits 416 records into 70% training (291 records) and 30% testing (125 records).
    Fits MinMaxScaler in range [-1, +1] per Section 3.1 of Kumar et al. (2024).
    """
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y_pu = df[TARGET_PU]
    y_fn = df[TARGET_FN]

    (
        X_train,
        X_test,
        y_pu_train,
        y_pu_test,
        y_fn_train,
        y_fn_test,
    ) = train_test_split(X, y_pu, y_fn, test_size=test_size, random_state=random_state)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index
    )

    if save_scaler:
        _, models_dir = get_paths()
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(models_dir, "minmax_scaler.joblib"))

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_pu_train": y_pu_train,
        "y_pu_test": y_pu_test,
        "y_fn_train": y_fn_train,
        "y_fn_test": y_fn_test,
        "scaler": scaler,
    }


if __name__ == "__main__":
    splits = prepare_splits()
    print(f"Total entries : {len(splits['df'])}")
    print(f"Training (70%): {len(splits['X_train'])}")
    print(f"Testing  (30%): {len(splits['X_test'])}")
