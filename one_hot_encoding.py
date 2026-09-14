from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "AEP_hourly.csv"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime", "AEP_MW"]).copy()

    df["Hour"] = df["Datetime"].dt.hour
    df["Month"] = df["Datetime"].dt.month
    df["DayOfWeek"] = df["Datetime"].dt.day_name()
    df["IsWeekend"] = df["Datetime"].dt.dayofweek.ge(5).astype(str)
    return df


df = load_and_prepare_data()
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

nominal_cols = ["DayOfWeek", "IsWeekend"]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("\nTraining data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

encoder = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore",
)

train_ohe = encoder.fit_transform(train_df[nominal_cols])
test_ohe = encoder.transform(test_df[nominal_cols])

encoded_cols = encoder.get_feature_names_out(nominal_cols)

train_ohe_df = pd.DataFrame(train_ohe, columns=encoded_cols, index=train_df.index)
test_ohe_df = pd.DataFrame(test_ohe, columns=encoded_cols, index=test_df.index)

print("\nNumber of original categorical columns:", len(nominal_cols))
print("\nGenerated One-Hot Columns:")
for col in encoded_cols:
    print(col)

print("\nFirst 5 rows of encoded training data:")
print(train_ohe_df.head())

print("\nFirst 5 rows of encoded testing data:")
print(test_ohe_df.head())