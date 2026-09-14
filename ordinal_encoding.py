from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "AEP_hourly.csv"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime", "AEP_MW"]).copy()

    df["Hour"] = df["Datetime"].dt.hour
    df["Month"] = df["Datetime"].dt.month
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    return df


df = load_and_prepare_data()
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("\nTraining data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

# Ordered categories for the AEP dataset
month_order = list(range(1, 13))
day_order = list(range(7))

ord_enc = OrdinalEncoder(
    categories=[day_order, month_order],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

ordinal_cols = ["DayOfWeek", "Month"]

train_encoded = ord_enc.fit_transform(train_df[ordinal_cols])
test_encoded = ord_enc.transform(test_df[ordinal_cols])

train_df[["DayOfWeek_enc", "Month_enc"]] = train_encoded
test_df[["DayOfWeek_enc", "Month_enc"]] = test_encoded

print("\nTraining data sample:")
print(train_df[["Datetime", "DayOfWeek", "DayOfWeek_enc", "Month", "Month_enc"]].head())

print("\nTesting data sample:")
print(test_df[["Datetime", "DayOfWeek", "DayOfWeek_enc", "Month", "Month_enc"]].head())