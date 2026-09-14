from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "AEP_hourly.csv"


def load_data():
    data = pd.read_csv(DATA_PATH)
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data = data.dropna(subset=["Datetime", "AEP_MW"]).copy()

    data["Hour"] = data["Datetime"].dt.hour
    data["Month"] = data["Datetime"].dt.month
    data["DayOfWeek"] = data["Datetime"].dt.dayofweek
    data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)

    median_demand = data["AEP_MW"].median()
    data["HighDemand"] = (data["AEP_MW"] >= median_demand).astype(int)

    features = ["Hour", "Month", "DayOfWeek", "IsWeekend"]
    target = "HighDemand"

    X = data[features]
    y = data[target]
    return X, y


def train_decision_tree_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "model_name": "Decision Tree",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }


if __name__ == "__main__":
    summary = train_decision_tree_model()
    print(summary)
