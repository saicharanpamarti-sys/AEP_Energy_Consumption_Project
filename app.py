import importlib

from flask import Flask, render_template
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from decision_tree_model import load_data as load_decision_tree_data, train_decision_tree_model
from linear_regression_sklearn import train_and_evaluate, load_data as load_regression_data, split_data
from load_data import get_data_summary, load_data
from logistic_regression import load_data as load_logistic_data, split_data as split_logistic_data, train_and_evaluate as train_logistic_and_evaluate
# workflow
app = Flask(__name__)


def _build_boosting_summary(model_name, model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "model_name": model_name,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }


def get_boosting_model_results():
    """Train the seven boosting variants for the AEP dataset."""
    modules = [
        ("adaboost", "AdaBoost", "ada_boosting"),
        ("gradient-boosting", "Gradient Boosting", "gradient_boosting_machine"),
        ("hist-gradient-boosting", "HistGradientBoosting", "hist_gradient_boosting"),
        ("xgboost", "XGBoost", "xgboost_model"),
        ("lightgbm", "LightGBM", "lightgbm_model"),
        ("catboost", "CatBoost", "catboost_model"),
        ("logitboost", "LogitBoost", "logitboost_model"),
    ]

    summaries = []
    errors = []

    for slug, title, module_name in modules:
        try:
            module = importlib.import_module(module_name)
            result = module.train_model()
            summaries.append({
                "slug": slug,
                "title": title,
                "result": result,
            })
        except Exception as exc:
            errors.append({"slug": slug, "title": title, "error": str(exc)})

    return summaries, errors


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
    # Landing page, no section selected yet
    return render_template("index.html", active="none")

@app.route("/data-loading")
def data_loading():
    """Loads the dataset (server-side) and renders the summary into the page."""
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
            "model_name": "Linear Regression",
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
        _, _ = train_logistic_and_evaluate(X_train, y_train, X_test, y_test, "Logistic Regression")
        model = __import__('sklearn.linear_model', fromlist=['LogisticRegression']).LogisticRegression(max_iter=1000)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        from sklearn.metrics import accuracy_score, confusion_matrix
        model_summary = {
            "model_name": "Logistic Regression",
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


@app.route("/decision-tree")
def decision_tree():
    error = None
    model_summary = None
    boosting_summaries = []
    boosting_errors = []

    try:
        model_summary = train_decision_tree_model()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    try:
        boosting_summaries, boosting_errors = get_boosting_model_results()
    except Exception as e:
        boosting_errors = [{"slug": "dashboard", "title": "Boosting models", "error": str(e)}]

    return render_template(
        "index.html",
        active="decision-tree",
        decision_tree_summary=model_summary,
        boosting_summaries=boosting_summaries,
        boosting_errors=boosting_errors,
        error=error,
    )


@app.route("/eda")
def eda():
    """Runs exploratory data analysis and renders results."""
    error = None
    eda_output = None
    try:
        from aep_eda import run_eda
        eda_output = run_eda()   # call energy EDA function
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

