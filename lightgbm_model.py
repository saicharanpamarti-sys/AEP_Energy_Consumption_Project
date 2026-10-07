"""
LightGBM Model for AEP Energy Consumption Project.
Implements lgb.LGBMRegressor and lgb.LGBMClassifier.
"""

from pathlib import Path
import time
import math
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "AEP_hourly_preprocessed.csv"
RAW_DATA_PATH = BASE_DIR / "data" / "AEP_hourly.csv"
FEATURES = ["Hour", "Month", "DayOfWeek", "IsWeekend"]


def load_training_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(RAW_DATA_PATH)
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime", "AEP_MW"]).copy()
        df["Hour"] = df["Datetime"].dt.hour
        df["Month"] = df["Datetime"].dt.month
        df["DayOfWeek"] = df["Datetime"].dt.dayofweek
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
        df["HighDemand"] = (df["AEP_MW"] >= df["AEP_MW"].median()).astype(int)

    if "HighDemand" not in df.columns:
        df["HighDemand"] = (df["AEP_MW"] >= df["AEP_MW"].median()).astype(int)

    X = df[FEATURES]
    y_reg = df["AEP_MW"]
    y_clf = df["HighDemand"].astype(int)
    return X, y_reg, y_clf


def train_regression():
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed. Install it with: pip install lightgbm")

    X, y_reg, _ = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(
        n_estimators=60,
        learning_rate=0.08,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t1 = time.time()
    test_pred = model.predict(X_test)
    inf_time = time.time() - t1
    train_pred = model.predict(X_train)

    return {
        "model_name": "LightGBM Regressor",
        "task": "Regression",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_rmse": round(math.sqrt(mean_squared_error(y_train, train_pred)), 2),
        "test_rmse": round(math.sqrt(mean_squared_error(y_test, test_pred)), 2),
        "train_mae": round(mean_absolute_error(y_train, train_pred), 2),
        "test_mae": round(mean_absolute_error(y_test, test_pred), 2),
        "train_r2": round(r2_score(y_train, train_pred), 4),
        "test_r2": round(r2_score(y_test, test_pred), 4),
        "oob_score": None,
        "train_time_sec": round(fit_time, 3),
        "inference_time_sec": round(inf_time, 4),
    }


def train_classification():
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed. Install it with: pip install lightgbm")

    X, _, y_clf = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    model = LGBMClassifier(
        n_estimators=60,
        learning_rate=0.08,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t1 = time.time()
    test_pred = model.predict(X_test)
    inf_time = time.time() - t1
    train_pred = model.predict(X_train)

    return {
        "model_name": "LightGBM",
        "task": "Classification",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_accuracy": round(accuracy_score(y_train, train_pred), 4),
        "test_accuracy": round(accuracy_score(y_test, test_pred), 4),
        "precision": round(precision_score(y_test, test_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, test_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, test_pred, zero_division=0), 4),
        "oob_score": None,
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "train_time_sec": round(fit_time, 3),
        "inference_time_sec": round(inf_time, 4),
    }


def train_model():
    """Backward-compatible classification method."""
    return train_classification()


if __name__ == "__main__":
    print("--- LightGBM Regression ---")
    print(train_regression())
    print("\n--- LightGBM Classification ---")
    print(train_classification())
