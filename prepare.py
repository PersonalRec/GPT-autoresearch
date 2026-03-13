"""
Fixed data loading and evaluation for car price prediction autoresearch.

DO NOT MODIFY — contains the fixed train/val split and ground-truth metric.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATA_PATH = "car_sales_data.csv"
TARGET = "Price"
VAL_SIZE = 0.2
RANDOM_STATE = 42

# Raw column names as they appear in the CSV
CAT_COLS = ["Manufacturer", "Model", "Fuel type"]
NUM_COLS = ["Engine size", "Year of manufacture", "Mileage"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """Load the raw car sales CSV. Returns the full DataFrame."""
    df = pd.read_csv(DATA_PATH)
    assert TARGET in df.columns, f"Target column '{TARGET}' not found in {DATA_PATH}"
    return df


def get_train_val_split(df: pd.DataFrame):
    """
    Fixed 80/20 split by row index with a pinned random seed.
    Always returns the same train/val partition regardless of call order.
    Returns: train_df, val_df (both reset-indexed).
    """
    train_df, val_df = train_test_split(df, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred) -> float:
    """
    Returns val_rmse: root mean squared error on raw price (£). Lower is better.

    Both y_true and y_pred must be in the original price scale (not log-transformed).
    If train.py trains on log(price), it must exponentiate predictions before calling this.
    """
    return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))
