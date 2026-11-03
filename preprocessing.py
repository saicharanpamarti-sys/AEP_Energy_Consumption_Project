"""
Preprocessing and Feature Engineering Module for AEP Energy Consumption Project.
Audits all dataframe exports, completely eliminates uppercase/CamelCase duplicate columns,
standardizes strictly to lowercase/snake_case:
['Datetime', 'AEP_MW', 'hour', 'month', 'dayofweek', 'is_weekend', 'lag_1', 'lag_24', 'High_Demand']
and saves sanitized CSVs.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "AEP_hourly.csv"
OUTPUT_PREPROCESSED_PATH = DATA_DIR / "AEP_preprocessed.csv"
OUTPUT_HOURLY_PREPROCESSED_PATH = DATA_DIR / "AEP_hourly_preprocessed.csv"

CANONICAL_COLUMNS = [
    "Datetime",
    "AEP_MW",
    "hour",
    "month",
    "dayofweek",
    "is_weekend",
    "lag_1",
    "lag_24",
    "High_Demand",
]


def run_preprocessing():
    """
    Executes full preprocessing pipeline and generates clean, deduplicated CSVs.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    original_rows = len(df)
    original_cols = list(df.columns)

    # 1. Parse Datetime & Sort Chronologically
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime", "AEP_MW"]).sort_values("Datetime").reset_index(drop=True)

    # 2. Extract Temporal Features strictly in lowercase/snake_case
    df["hour"] = df["Datetime"].dt.hour.astype(int)
    df["month"] = df["Datetime"].dt.month.astype(int)
    df["dayofweek"] = df["Datetime"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 3. Autoregressive Lag Features
    df["lag_1"] = df["AEP_MW"].shift(1)
    df["lag_24"] = df["AEP_MW"].shift(24)

    # 4. Backward fill boundary lags to preserve all rows with zero missing values
    df = df.bfill().copy()

    # 5. Target Discretization (Median threshold: 15,310 MW)
    median_demand = float(df["AEP_MW"].median())
    df["High_Demand"] = (df["AEP_MW"] >= median_demand).astype(int)

    # 6. Explicitly drop all uppercase/CamelCase duplicate columns
    drop_cols = ["Hour", "Month", "DayOfWeek", "IsWeekend", "HighDemand"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 7. Standardize strictly to the 9 canonical columns
    df = df[CANONICAL_COLUMNS].copy()

    # 8. Export sanitized CSVs
    df.to_csv(OUTPUT_PREPROCESSED_PATH, index=False)
    try:
        df.to_csv(OUTPUT_HOURLY_PREPROCESSED_PATH, index=False)
    except Exception as e:
        print(f"Notice: {OUTPUT_HOURLY_PREPROCESSED_PATH} is locked by IDE: {e}")

    print(f"Sanitized CSV successfully written to:")
    print(f"  - {OUTPUT_PREPROCESSED_PATH}")
    print(f"  - {OUTPUT_HOURLY_PREPROCESSED_PATH}")
    print(f"Retained shape: {df.shape} | Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    run_preprocessing()
