from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def train_and_evaluate(X_train, y_train, X_test, y_test, name):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(name, "Train Accuracy:", round(train_accuracy, 4))
    print(name, "Test Accuracy:", round(test_accuracy, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    return model, scaler


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    train_and_evaluate(X_train, y_train, X_test, y_test, "Logistic Regression")


if __name__ == "__main__":
    main()
