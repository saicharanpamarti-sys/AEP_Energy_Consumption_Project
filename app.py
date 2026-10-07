from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from linear_regression_sklearn import train_and_evaluate, load_data as load_regression_data, split_data
from load_data import get_data_summary, load_data
from logistic_regression import (
    load_data as load_logistic_data,
    split_data as split_logistic_data,
    train_and_evaluate as train_logistic_and_evaluate,
)
from tree_ensemble_suite import (
    run_tree_ensemble_benchmark,
    predict_demand,
    MODEL_CONFIGS,
    FEATURE_NAMES,
)

app = Flask(__name__)


def get_preprocessing_summary():
    df = load_data()
    original_rows = len(df)
    original_columns = list(df.columns)
    original_missing = int(df.isna().sum().sum())

    cleaned = df.copy()
    cleaned["Datetime"] = pd.to_datetime(cleaned["Datetime"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Datetime", "AEP_MW"]).copy()
    cleaned["Hour"] = cleaned["Datetime"].dt.hour
    cleaned["Month"] = cleaned["Datetime"].dt.month
    cleaned["DayOfWeek"] = cleaned["Datetime"].dt.dayofweek
    cleaned["IsWeekend"] = (cleaned["DayOfWeek"] >= 5).astype(int)
    if "HighDemand" not in cleaned.columns:
        cleaned["HighDemand"] = (cleaned["AEP_MW"] >= cleaned["AEP_MW"].median()).astype(int)

    # Save preprocessed data to CSV
    import os
    preprocessed_path = os.path.join(os.path.dirname(__file__), "data", "AEP_hourly_preprocessed.csv")
    cleaned.to_csv(preprocessed_path, index=False)

    return {
        "original_rows": original_rows,
        "processed_rows": len(cleaned),
        "dropped_rows": original_rows - len(cleaned),
        "original_columns": original_columns,
        "processed_columns": list(cleaned.columns),
        "original_missing": original_missing,
        "processed_missing": int(cleaned.isna().sum().sum()),
        "new_features": ["Hour", "Month", "DayOfWeek", "IsWeekend"],
        "preview": cleaned.head(5).to_dict("records"),
        "preprocessed_path": preprocessed_path,
    }


@app.route("/")
def index():
    """Landing dashboard page."""
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    """Loads the dataset and renders the summary into the page."""
    error = None
    summary = None
    try:
        summary = get_data_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
    )


@app.route("/preprocessing")
def preprocessing():
    """Shows the preprocessing workflow for the AEP energy dataset."""
    error = None
    summary = None
    try:
        summary = get_preprocessing_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        summary=summary,
        error=error,
    )


@app.route("/linear-regression")
def linear_regression():
    error = None
    model_summary = None
    try:
        X, y = load_regression_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_and_evaluate(X_train, y_train, X_test, y_test, "Linear Regression")
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        model_summary = {
            "model_name": "Linear Regression Baseline",
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_r2": float(model.score(X_train, y_train)),
            "test_r2": float(model.score(X_test, y_test)),
            "train_mae": float((abs(y_train - train_pred)).mean()),
            "test_mae": float((abs(y_test - test_pred)).mean()),
            "test_rmse": float(((y_test - test_pred) ** 2).mean() ** 0.5),
        }
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="linear-regression",
        model_summary=model_summary,
        error=error,
    )


@app.route("/logistic-regression")
def logistic_regression():
    error = None
    model_summary = None
    try:
        X, y = load_logistic_data()
        X_train, X_test, y_train, y_test = split_logistic_data(X, y)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, confusion_matrix

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        model_summary = {
            "model_name": "Logistic Regression Baseline",
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        }
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="logistic-regression",
        logistic_summary=model_summary,
        error=error,
    )


@app.route("/models")
@app.route("/tree-ensemble")
@app.route("/decision-tree")
@app.route("/boosting")
def models_suite():
    """
    Renders the unified Tree & Ensemble Model Benchmark Suite.
    Compares the 6 syllabus approved models:
      1. Decision Tree (Regressor / Classifier)
      2. Random Forest (Regressor / Classifier with oob_score=True)
      3. AdaBoost (Regressor / Classifier)
      4. Gradient Boosting (Regressor / Classifier)
      5. XGBoost (Regressor / Classifier)
      6. LightGBM (Regressor / Classifier)
    """
    error = None
    benchmark_data = None
    force_retrain = request.args.get("retrain", "false").lower() == "true"

    try:
        benchmark_data = run_tree_ensemble_benchmark(force=force_retrain)
    except Exception as e:
        error = f"Error training/evaluating models suite: {e}"

    return render_template(
        "index.html",
        active="models",
        benchmark=benchmark_data,
        model_configs=MODEL_CONFIGS,
        error=error,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """
    Interactive prediction endpoint.
    Accepts Hour, Month, DayOfWeek, IsWeekend, Model, and Task.
    Provides instant demand forecast (MW) or high-demand classification probability.
    """
    error = None
    prediction_result = None

    # Default form values
    form_values = {
        "hour": 14,
        "month": 7,
        "day_of_week": 2,
        "is_weekend": 0,
        "task": "regression",
        "model_key": "random_forest",
    }

    if request.method == "POST":
        try:
            form_values["hour"] = int(request.form.get("hour", 14))
            form_values["month"] = int(request.form.get("month", 7))
            form_values["day_of_week"] = int(request.form.get("day_of_week", 2))
            form_values["is_weekend"] = int(request.form.get("is_weekend", 0))
            form_values["task"] = request.form.get("task", "regression")
            form_values["model_key"] = request.form.get("model_key", "random_forest")

            # Validate ranges
            form_values["hour"] = max(0, min(23, form_values["hour"]))
            form_values["month"] = max(1, min(12, form_values["month"]))
            form_values["day_of_week"] = max(0, min(6, form_values["day_of_week"]))
            form_values["is_weekend"] = 1 if form_values["is_weekend"] == 1 or form_values["day_of_week"] >= 5 else 0

            prediction_result = predict_demand(
                task=form_values["task"],
                model_key=form_values["model_key"],
                hour=form_values["hour"],
                month=form_values["month"],
                day_of_week=form_values["day_of_week"],
                is_weekend=form_values["is_weekend"],
            )
        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template(
        "index.html",
        active="prediction",
        form_values=form_values,
        prediction_result=prediction_result,
        model_configs=MODEL_CONFIGS,
        error=error,
    )


@app.route("/eda")
def eda():
    """Runs exploratory data analysis and renders results."""
    error = None
    eda_output = None
    try:
        from aep_eda import run_eda
        eda_output = run_eda()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)
