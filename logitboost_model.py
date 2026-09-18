from pathlib import Path

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "AEP_hourly.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "AEP_hourly_preprocessed.csv"


def load_training_data():
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
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

    features = ["Hour", "Month", "DayOfWeek", "IsWeekend"]
    X = df[features]
    y = df["HighDemand"].astype(int)
    return X, y


def train_model():
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "model_name": "LogitBoost",
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }


if __name__ == "__main__":
    result = train_model()
    print(result)
