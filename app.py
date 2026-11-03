from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd

from linear_regression_sklearn import get_linear_regression_comparison
from load_data import get_data_summary, get_preprocessing_summary
from logistic_regression import get_logistic_regression_summary
from decision_tree_model import (
    train_regression as train_dt_regression,
    train_classification as train_dt_classification,
    get_depth_impact_table,
)
from random_forest_model import (
    train_regression as train_rf_regression,
    train_classification as train_rf_classification,
)
from tree_ensemble_suite import (
    run_tree_ensemble_benchmark,
    predict_demand,
    MODEL_CONFIGS,
)
from evaluation_calibration import compute_metrics_and_calibration
from tuning_comparison import compute_tuning_and_significance

app = Flask(__name__)


# -------------------------------------------------------------
# CORE DASHBOARD
# -------------------------------------------------------------
@app.route("/")
def dashboard():
    """Home summary page with dataset high-level stats and architecture overview."""
    return render_template("dashboard.html", active="dashboard")


# -------------------------------------------------------------
# SUPERVISED LEARNING
# -------------------------------------------------------------
@app.route("/load-data")
@app.route("/data-loading")
def load_data_route():
    """Data Loading view: Row count 121,273, raw types, first 10 rows."""
    error = None
    summary = None
    try:
        summary = get_data_summary()
    except Exception as e:
        error = f"Error loading dataset: {e}"

    return render_template(
        "load_data.html",
        active="load-data",
        summary=summary,
        error=error,
    )


@app.route("/preprocessing")
def preprocessing():
    """Preprocessing view: Cleaning steps, zero null confirmation, created features."""
    error = None
    summary = None
    try:
        summary = get_preprocessing_summary()
    except Exception as e:
        error = f"Error executing preprocessing pipeline: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        summary=summary,
        error=error,
    )


@app.route("/eda")
def eda():
    """EDA view: 15 comprehensive diagnostic figures with analytical captions & caching."""
    error = None
    eda_output = None
    try:
        from eda import generate_15_eda_charts
        eda_output = generate_15_eda_charts()
    except Exception as e:
        error = f"Error executing EDA: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        charts=eda_output.get("charts", []) if eda_output else [],
        error=error,
    )


@app.route("/linear-regression")
@app.route("/linear_regression")
def linear_regression():
    """Linear Regression view: Single-feature vs multi-feature comparison table & test metrics."""
    error = None
    comparison_data = None
    try:
        from linear_regression_sklearn import get_linear_regression_comparison
        comparison_data = get_linear_regression_comparison()
    except Exception as e:
        error = f"Error evaluating linear regression: {e}"

    models = comparison_data["models"] if comparison_data else []
    calendar_baseline = comparison_data.get("calendar_baseline", {}) if comparison_data else {}
    autoregressive_model = comparison_data.get("autoregressive_model", {}) if comparison_data else {}

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        models=models,
        calendar_baseline=calendar_baseline,
        autoregressive_model=autoregressive_model,
        primary_model=calendar_baseline,
        best_model=autoregressive_model,
        error=error,
    )


@app.route("/logistic-regression")
@app.route("/logistic_regression")
def logistic_regression():
    """Logistic Regression view: Dual Model Architecture & High-Demand classification diagnostics."""
    error = None
    logistic_summary = None
    try:
        from logistic_regression import get_logistic_regression_summary
        logistic_summary = get_logistic_regression_summary()
    except Exception as e:
        error = f"Error evaluating logistic regression: {e}"

    model_a = logistic_summary.get("model_a", {}) if logistic_summary else {}
    model_b = logistic_summary.get("model_b", {}) if logistic_summary else {}

    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        summary=logistic_summary.get("summary", {}) if logistic_summary else {},
        model_a=model_a,
        model_b=model_b,
        error_reduction=logistic_summary.get("error_reduction", 0.0) if logistic_summary else 0.0,
        error=error,
    )


@app.route("/decision-tree")
@app.route("/decision_tree")
def decision_tree_route():
    """Dedicated Decision Tree page: Regressor, Classifier, depth impact table & feature importance chart."""
    error = None
    reg = {}
    clf = {}
    depth_table = []
    try:
        reg = train_dt_regression()
        clf = train_dt_classification()
        depth_table = get_depth_impact_table()
    except Exception as e:
        error = f"Error evaluating Decision Tree: {e}"

    return render_template(
        "decision_tree.html",
        active="decision-tree",
        regression=reg,
        classification=clf,
        depth_table=depth_table,
        error=error,
    )


@app.route("/bagging")
def bagging_route():
    """Dedicated Bagging page: RandomForest with oob_score=True, OOB curve, & variance reduction."""
    error = None
    reg = {}
    clf = {}
    try:
        reg = train_rf_regression()
        clf = train_rf_classification()
    except Exception as e:
        error = f"Error evaluating Bagging ensemble: {e}"

    return render_template(
        "bagging.html",
        active="bagging",
        regression=reg,
        classification=clf,
        error=error,
    )


@app.route("/boosting")
def boosting_route():
    """Dedicated Boosting page: Tabbed interface for AdaBoost, Gradient Boosting, XGBoost, LightGBM."""
    error = None
    benchmark = None
    boosting_models = []
    try:
        benchmark = run_tree_ensemble_benchmark()
        # Filter for the 4 boosting algorithms
        boosting_keys = {"adaboost", "gradient_boosting", "xgboost", "lightgbm"}
        boosting_models = [
            m for m in benchmark.get("models_detail", []) if m["key"] in boosting_keys
        ]
    except Exception as e:
        error = f"Error evaluating Boosting suite: {e}"

    return render_template(
        "boosting.html",
        active="boosting",
        boosting_models=boosting_models,
        error=error,
    )


@app.route("/evaluation")
def evaluation_route():
    """Metrics & Calibration page: Regression Outlier Gap, Imbalance Metrics, & Platt/Isotonic Calibration."""
    error = None
    eval_data = None
    force_run = request.args.get("recompute", "false").lower() == "true"
    try:
        eval_data = compute_metrics_and_calibration(force=force_run)
    except Exception as e:
        error = f"Error computing metrics & calibration: {e}"

    return render_template(
        "evaluation.html",
        active="evaluation",
        data=eval_data,
        error=error,
    )


@app.route("/tuning-comparison")
@app.route("/tuning_comparison")
def tuning_comparison_route():
    """Tuning & Significance page: GridSearchCV vs RandomizedSearchCV, Diagnostic curves, & McNemar / Bootstrap tests."""
    error = None
    tuning_data = None
    force_run = request.args.get("recompute", "false").lower() == "true"
    try:
        tuning_data = compute_tuning_and_significance(force=force_run)
    except Exception as e:
        error = f"Error computing tuning and significance tests: {e}"

    return render_template(
        "tuning_comparison.html",
        active="tuning-comparison",
        data=tuning_data,
        error=error,
    )


@app.route("/predictor", methods=["GET", "POST"], endpoint="predictor_route")
@app.route("/predict", methods=["GET", "POST"], endpoint="predict_page")
def predictor_route():
    """Live interactive energy demand predictor simulation."""
    error = None
    prediction_result = None

    form_values = {
        "hour": 14,
        "month": 7,
        "day_of_week": 2,
        "is_weekend": 0,
        "lag_1": 15200.0,
        "lag_24": 15100.0,
        "task": "regression",
        "model_key": "random_forest",
    }

    if request.method == "POST":
        try:
            form_values["hour"] = int(request.form.get("hour", 14))
            form_values["month"] = int(request.form.get("month", 7))
            form_values["day_of_week"] = int(request.form.get("day_of_week", 2))
            form_values["is_weekend"] = int(request.form.get("is_weekend", 0))
            form_values["lag_1"] = float(request.form.get("lag_1", 15200.0))
            form_values["lag_24"] = float(request.form.get("lag_24", 15100.0))
            form_values["task"] = request.form.get("task", "regression")
            form_values["model_key"] = request.form.get("model_key", "random_forest")

            form_values["hour"] = max(0, min(23, form_values["hour"]))
            form_values["month"] = max(1, min(12, form_values["month"]))
            form_values["day_of_week"] = max(0, min(6, form_values["day_of_week"]))
            form_values["is_weekend"] = (
                1 if form_values["is_weekend"] == 1 or form_values["day_of_week"] >= 5 else 0
            )

            prediction_result = predict_demand(
                task=form_values["task"],
                model_key=form_values["model_key"],
                hour=form_values["hour"],
                month=form_values["month"],
                day_of_week=form_values["day_of_week"],
                is_weekend=form_values["is_weekend"],
                lag_1=form_values["lag_1"],
                lag_24=form_values["lag_24"],
            )
        except Exception as e:
            error = f"Prediction computation failed: {e}"

    return render_template(
        "predict.html",
        active="predictor",
        form_values=form_values,
        prediction_result=prediction_result,
        model_configs=MODEL_CONFIGS,
        error=error,
    )


@app.route("/tree-ensembles")
@app.route("/tree_ensembles")
@app.route("/models")
def tree_ensembles():
    """Full 6-model unified benchmark table."""
    error = None
    benchmark_data = None
    try:
        benchmark_data = run_tree_ensemble_benchmark()
    except Exception as e:
        error = f"Error loading tree ensemble suite: {e}"

    return render_template(
        "tree_ensembles.html",
        active="tree-ensembles",
        benchmark=benchmark_data,
        model_configs=MODEL_CONFIGS,
        error=error,
    )


# -------------------------------------------------------------
# UNSUPERVISED LEARNING
# -------------------------------------------------------------
@app.route("/clustering")
def clustering_route():
    """Comprehensive Unsupervised Clustering Suite: 4 Paradigms & 3 K-Selection Methods."""
    error = None
    summary = None
    force_run = request.args.get("recompute", "false").lower() == "true"
    try:
        from clustering_model import get_clustering_summary
        summary = get_clustering_summary(force=force_run)
    except Exception as e:
        error = f"Error evaluating clustering suite: {e}"

    return render_template(
        "clustering.html",
        active="clustering",
        summary=summary,
        error=error,
    )


@app.route("/pca")
def pca_route():
    """PCA Dimensionality Reduction architecture and variance decomposition."""
    error = None
    pca_data = None
    try:
        from pca_model import compute_pca_summary
        pca_data = compute_pca_summary()
    except Exception as e:
        error = f"Error evaluating PCA: {e}"

    return render_template("pca.html", active="pca", pca_data=pca_data, error=error)


@app.route("/api/clustering/kmeans")
def api_clustering_kmeans():
    """Returns diurnal profiles chart for interactive K explorer."""
    try:
        k = int(request.args.get("k", 3))
        k = max(2, min(8, k))
        from clustering_model import get_kmeans_diurnal_data
        result = get_kmeans_diurnal_data(k=k)
        return jsonify({"status": "success", "k": k, "chart_url": url_for("static", filename=result["chart_file"])})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/clustering/dbscan")
def api_clustering_dbscan():
    """Runs interactive DBSCAN with user eps and min_samples."""
    try:
        eps = float(request.args.get("eps", 0.85))
        min_samples = int(request.args.get("min_samples", 15))
        from clustering_model import run_dbscan_interactive
        result = run_dbscan_interactive(eps=eps, min_samples=min_samples)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)
