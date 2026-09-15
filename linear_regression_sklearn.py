from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

    features = ["Hour", "Month", "DayOfWeek", "IsWeekend"]
    target = "AEP_MW"

    X = data[features]
    y = data[target]
    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def train_and_evaluate(X_train, y_train, X_test, y_test, name):
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5

    print(name, "Train R^2:", round(train_r2, 4))
    print(name, "Test R^2:", round(test_r2, 4))
    print(name, "Train MAE:", round(train_mae, 4))
    print(name, "Test MAE:", round(test_mae, 4))
    print(name, "Test RMSE:", round(test_rmse, 4))

    return model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    train_and_evaluate(X_train, y_train, X_test, y_test, "Linear Regression")


if __name__ == "__main__":
    main()
