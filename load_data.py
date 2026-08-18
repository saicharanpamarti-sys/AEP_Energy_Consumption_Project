import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "AEP_hourly.csv")

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the American Electric Power hourly energy consumption dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    # Read the full CSV dataset (121,275 rows is small enough to load fully)
    data = pd.read_csv(path)
    return data

def get_data_summary() -> dict:
    df = load_data()
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": df.head(10).to_dict("records"),
    }
    return summary

if __name__ == "__main__":
    print(get_data_summary())
