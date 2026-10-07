"""
Tree and Ensemble Modeling Suite for AEP Energy Consumption Project.
Adheres strictly to the syllabus requirements:
  - Decision Tree (DecisionTreeRegressor / DecisionTreeClassifier)
  - Random Forest (RandomForestRegressor / RandomForestClassifier with oob_score=True)
  - AdaBoost (AdaBoostRegressor / AdaBoostClassifier)
  - Gradient Boosting (GradientBoostingRegressor / GradientBoostingClassifier)
  - XGBoost (xgb.XGBRegressor / xgb.XGBClassifier)
  - LightGBM (lgb.LGBMRegressor / lgb.LGBMClassifier)
Provides full evaluation:
  - Regression: RMSE, MAE, R² score
  - Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix
  - Training execution time (seconds), Inference execution time (seconds)
  - Random Forest Out-Of-Bag (OOB) score
"""

import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "AEP_hourly.csv"
PREPROCESSED_DATA_PATH = DATA_DIR / "AEP_hourly_preprocessed.csv"

FEATURE_NAMES = ["Hour", "Month", "DayOfWeek", "IsWeekend"]

MODEL_CONFIGS = [
    {
        "key": "decision_tree",
        "title": "Decision Tree",
        "category": "Decision Tree",
        "create_regressor": lambda: DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=10, random_state=42
        ),
        "create_classifier": lambda: DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10, random_state=42
        ),
    },
    {
        "key": "random_forest",
        "title": "Random Forest",
        "category": "Bagging Ensemble",
        "create_regressor": lambda: RandomForestRegressor(
            n_estimators=50, max_depth=12, oob_score=True, random_state=42, n_jobs=-1
        ),
        "create_classifier": lambda: RandomForestClassifier(
            n_estimators=50, max_depth=12, oob_score=True, random_state=42, n_jobs=-1
        ),
    },
    {
        "key": "adaboost",
        "title": "AdaBoost",
        "category": "Boosting",
        "create_regressor": lambda: AdaBoostRegressor(
            n_estimators=50, learning_rate=0.1, random_state=42
        ),
        "create_classifier": lambda: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=50,
            learning_rate=0.5,
            random_state=42,
        ),
    },
    {
        "key": "gradient_boosting",
        "title": "Gradient Boosting",
        "category": "Boosting",
        "create_regressor": lambda: GradientBoostingRegressor(
            n_estimators=60, learning_rate=0.08, max_depth=4, random_state=42
        ),
        "create_classifier": lambda: GradientBoostingClassifier(
            n_estimators=60, learning_rate=0.08, max_depth=4, random_state=42
        ),
    },
    {
        "key": "xgboost",
        "title": "XGBoost",
        "category": "Boosting",
        "create_regressor": lambda: (
            XGBRegressor(
                n_estimators=60,
                learning_rate=0.08,
                max_depth=4,
                random_state=42,
                n_jobs=-1,
            )
            if XGBRegressor is not None
            else None
        ),
        "create_classifier": lambda: (
            XGBClassifier(
                n_estimators=60,
                learning_rate=0.08,
                max_depth=4,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
            if XGBClassifier is not None
            else None
        ),
    },
    {
        "key": "lightgbm",
        "title": "LightGBM",
        "category": "Boosting",
        "create_regressor": lambda: (
            LGBMRegressor(
                n_estimators=60,
                learning_rate=0.08,
                max_depth=4,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            if LGBMRegressor is not None
            else None
        ),
        "create_classifier": lambda: (
            LGBMClassifier(
                n_estimators=60,
                learning_rate=0.08,
                max_depth=4,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            if LGBMClassifier is not None
            else None
        ),
    },
]


def load_dataset() -> pd.DataFrame:
    """Load or generate preprocessed AEP energy data."""
    if PREPROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PREPROCESSED_DATA_PATH)
    else:
        df = pd.read_csv(RAW_DATA_PATH)
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime", "AEP_MW"]).copy()
        df["Hour"] = df["Datetime"].dt.hour
        df["Month"] = df["Datetime"].dt.month
        df["DayOfWeek"] = df["Datetime"].dt.dayofweek
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
        df["HighDemand"] = (df["AEP_MW"] >= df["AEP_MW"].median()).astype(int)
        df.to_csv(PREPROCESSED_DATA_PATH, index=False)

    if "HighDemand" not in df.columns:
        df["HighDemand"] = (df["AEP_MW"] >= df["AEP_MW"].median()).astype(int)

    return df


def prepare_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Prepares train/test split for both regression and classification."""
    df = load_dataset()
    X = df[FEATURE_NAMES]
    y_reg = df["AEP_MW"]
    y_clf = df["HighDemand"].astype(int)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


class TreeEnsembleSuite:
    """Singleton service to train, benchmark, cache and predict with the 6 approved models."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.fitted_regressors: Dict[str, Any] = {}
        self.fitted_classifiers: Dict[str, Any] = {}
        self.regression_results: List[Dict[str, Any]] = []
        self.classification_results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, str]] = []
        self.split_info: Dict[str, Any] = {}
        self.median_demand: float = 15000.0
        self.last_run_timestamp: Optional[str] = None
        self._is_trained = False

    @classmethod
    def get_instance(cls) -> "TreeEnsembleSuite":
        with cls._lock:
            if cls._instance is None:
                cls._instance = TreeEnsembleSuite()
            return cls._instance

    def train_and_benchmark(self, force: bool = False) -> Dict[str, Any]:
        """Trains all 6 models for Regression and Classification, benchmarking them."""
        with self._lock:
            if self._is_trained and not force:
                return self.get_summary()

            self.regression_results = []
            self.classification_results = []
            self.errors = []
            self.fitted_regressors = {}
            self.fitted_classifiers = {}

            try:
                X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = (
                    prepare_data()
                )
            except Exception as e:
                self.errors.append({"stage": "data_preparation", "error": str(e)})
                return self.get_summary()

            self.median_demand = float(y_reg_train.median())
            self.split_info = {
                "total_rows": int(len(X_train) + len(X_test)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "features": FEATURE_NAMES,
                "median_demand_threshold": round(self.median_demand, 2),
            }

            for config in MODEL_CONFIGS:
                key = config["key"]
                title = config["title"]
                category = config["category"]

                # --- 1. Train Regression Model ---
                try:
                    reg_model = config["create_regressor"]()
                    if reg_model is None:
                        raise ImportError(f"{title} library is not available.")

                    t0 = time.time()
                    reg_model.fit(X_train, y_reg_train)
                    reg_train_time = time.time() - t0

                    t1 = time.time()
                    y_reg_test_pred = reg_model.predict(X_test)
                    reg_inf_time = time.time() - t1

                    y_reg_train_pred = reg_model.predict(X_train)

                    reg_test_rmse = math.sqrt(mean_squared_error(y_reg_test, y_reg_test_pred))
                    reg_test_mae = mean_absolute_error(y_reg_test, y_reg_test_pred)
                    reg_test_r2 = r2_score(y_reg_test, y_reg_test_pred)
                    reg_train_r2 = r2_score(y_reg_train, y_reg_train_pred)

                    oob_score = getattr(reg_model, "oob_score_", None)
                    oob_score_val = (
                        round(float(oob_score), 4) if oob_score is not None else None
                    )

                    self.fitted_regressors[key] = reg_model
                    self.regression_results.append({
                        "key": key,
                        "title": title,
                        "category": category,
                        "train_rmse": round(
                            math.sqrt(mean_squared_error(y_reg_train, y_reg_train_pred)), 2
                        ),
                        "test_rmse": round(reg_test_rmse, 2),
                        "train_mae": round(
                            mean_absolute_error(y_reg_train, y_reg_train_pred), 2
                        ),
                        "test_mae": round(reg_test_mae, 2),
                        "train_r2": round(reg_train_r2, 4),
                        "test_r2": round(reg_test_r2, 4),
                        "train_time_sec": round(reg_train_time, 3),
                        "inference_time_sec": round(reg_inf_time, 4),
                        "oob_score": oob_score_val,
                        "sample_predictions": [
                            round(float(p), 2) for p in y_reg_test_pred[:5]
                        ],
                    })
                except Exception as e:
                    self.errors.append({
                        "model": title,
                        "task": "Regression",
                        "error": str(e),
                    })

                # --- 2. Train Classification Model ---
                try:
                    clf_model = config["create_classifier"]()
                    if clf_model is None:
                        raise ImportError(f"{title} library is not available.")

                    t0 = time.time()
                    clf_model.fit(X_train, y_clf_train)
                    clf_train_time = time.time() - t0

                    t1 = time.time()
                    y_clf_test_pred = clf_model.predict(X_test)
                    clf_inf_time = time.time() - t1

                    y_clf_train_pred = clf_model.predict(X_train)

                    train_acc = accuracy_score(y_clf_train, y_clf_train_pred)
                    test_acc = accuracy_score(y_clf_test, y_clf_test_pred)
                    test_prec = precision_score(y_clf_test, y_clf_test_pred, zero_division=0)
                    test_rec = recall_score(y_clf_test, y_clf_test_pred, zero_division=0)
                    test_f1 = f1_score(y_clf_test, y_clf_test_pred, zero_division=0)
                    cm = confusion_matrix(y_clf_test, y_clf_test_pred).tolist()

                    oob_score = getattr(clf_model, "oob_score_", None)
                    oob_score_val = (
                        round(float(oob_score), 4) if oob_score is not None else None
                    )

                    self.fitted_classifiers[key] = clf_model
                    self.classification_results.append({
                        "key": key,
                        "title": title,
                        "category": category,
                        "train_accuracy": round(train_acc, 4),
                        "test_accuracy": round(test_acc, 4),
                        "precision": round(test_prec, 4),
                        "recall": round(test_rec, 4),
                        "f1_score": round(test_f1, 4),
                        "train_time_sec": round(clf_train_time, 3),
                        "inference_time_sec": round(clf_inf_time, 4),
                        "oob_score": oob_score_val,
                        "confusion_matrix": cm,
                    })
                except Exception as e:
                    self.errors.append({
                        "model": title,
                        "task": "Classification",
                        "error": str(e),
                    })

            self.last_run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self._is_trained = True
            return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Returns the summary data dictionary."""
        return {
            "regression_results": self.regression_results,
            "classification_results": self.classification_results,
            "errors": self.errors,
            "split_info": self.split_info,
            "last_run": self.last_run_timestamp,
            "models_available": list(self.fitted_regressors.keys()),
        }

    def predict(
        self,
        task: str,
        model_key: str,
        hour: int,
        month: int,
        day_of_week: int,
        is_weekend: int,
    ) -> Dict[str, Any]:
        """Makes an interactive prediction with the selected model."""
        if not self._is_trained:
            self.train_and_benchmark()

        input_df = pd.DataFrame(
            [[hour, month, day_of_week, is_weekend]],
            columns=FEATURE_NAMES,
        )

        model_title = next(
            (c["title"] for c in MODEL_CONFIGS if c["key"] == model_key), model_key
        )

        if task == "regression":
            model = self.fitted_regressors.get(model_key)
            if model is None:
                raise ValueError(
                    f"Regression model '{model_key}' is not fitted or available."
                )
            pred = float(model.predict(input_df)[0])
            is_high = pred >= self.median_demand
            return {
                "task": "regression",
                "model_key": model_key,
                "model_title": model_title,
                "inputs": {
                    "Hour": hour,
                    "Month": month,
                    "DayOfWeek": day_of_week,
                    "IsWeekend": is_weekend,
                },
                "predicted_mw": round(pred, 2),
                "high_demand_threshold": round(self.median_demand, 2),
                "is_high_demand": is_high,
            }
        else:
            model = self.fitted_classifiers.get(model_key)
            if model is None:
                raise ValueError(
                    f"Classification model '{model_key}' is not fitted or available."
                )
            pred_class = int(model.predict(input_df)[0])
            prob_high = None
            if hasattr(model, "predict_proba"):
                try:
                    prob_high = round(float(model.predict_proba(input_df)[0][1]), 4)
                except Exception:
                    prob_high = None

            return {
                "task": "classification",
                "model_key": model_key,
                "model_title": model_title,
                "inputs": {
                    "Hour": hour,
                    "Month": month,
                    "DayOfWeek": day_of_week,
                    "IsWeekend": is_weekend,
                },
                "predicted_class": pred_class,
                "class_label": "High Demand (>= Median)" if pred_class == 1 else "Normal Demand (< Median)",
                "high_demand_probability": prob_high,
            }


# Module level convenience functions
suite = TreeEnsembleSuite.get_instance()


def run_tree_ensemble_benchmark(force: bool = False) -> Dict[str, Any]:
    return suite.train_and_benchmark(force=force)


def predict_demand(
    task: str,
    model_key: str,
    hour: int,
    month: int,
    day_of_week: int,
    is_weekend: int,
) -> Dict[str, Any]:
    return suite.predict(task, model_key, hour, month, day_of_week, is_weekend)


if __name__ == "__main__":
    print("Executing Tree & Ensemble Suite Benchmark...")
    results = run_tree_ensemble_benchmark(force=True)
    print("\n--- REGRESSION BENCHMARK ---")
    for r in results["regression_results"]:
        print(
            f"{r['title']:<20} | RMSE: {r['test_rmse']:<8} | MAE: {r['test_mae']:<8} | "
            f"R2: {r['test_r2']:<7} | Fit: {r['train_time_sec']}s | OOB: {r['oob_score']}"
        )
    print("\n--- CLASSIFICATION BENCHMARK ---")
    for c in results["classification_results"]:
        print(
            f"{c['title']:<20} | Acc: {c['test_accuracy']:<7} | Prec: {c['precision']:<7} | "
            f"Rec: {c['recall']:<7} | F1: {c['f1_score']:<7} | Fit: {c['train_time_sec']}s | OOB: {c['oob_score']}"
        )
