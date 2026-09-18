from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "AEP_hourly.csv"
PREPROCESSED_DATA_PATH = DATA_DIR / "AEP_hourly_preprocessed.csv"


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    file_path = Path(path) if path is not None else RAW_DATA_PATH
    if not file_path.is_absolute():
        file_path = (BASE_DIR / file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    return pd.read_csv(file_path)


def load_preprocessed_data(path: str | Path | None = None) -> pd.DataFrame:
    return load_data(path or PREPROCESSED_DATA_PATH)


def get_data_summary() -> dict:
    df = load_data()
    return {
        "project_name": "AEP Energy Consumption Project",
        "dataset_name": RAW_DATA_PATH.name,
        "dataset_path": str(RAW_DATA_PATH),
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "duplicate_count": int(df.duplicated().sum()),
        "missing_total": int(df.isnull().sum().sum()),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": df.head(10).to_dict("records"),
    }


if __name__ == "__main__":
    print(get_data_summary())
