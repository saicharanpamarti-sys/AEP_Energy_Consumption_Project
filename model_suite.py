import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "AEP_hourly.csv"
PREPROCESSED_DATA_PATH = DATA_DIR / "AEP_hourly_preprocessed.csv"
CACHE_PATH = DATA_DIR / "model_explorer_cache.json"

ACTIVE_FEATURES = [
    "hour", "dayofweek", "month", "is_weekend", "quarter",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24h", "rolling_std_24h",
]

MODEL_METADATA_CONFIG = {
    "decision-tree": {
        "slug": "decision-tree",
        "title": "Decision Tree",
        "class_name": "DecisionTreeRegressor / DecisionTreeClassifier",
        "key_strength": "Fast, interpretable white-box baseline that captures non-linear step-function thresholds across diurnal peak-demand hours with zero feature scaling required.",
        "feature_selection_strategy": "Greedy split over all 10 features at each tree node, selecting the split that maximizes variance reduction.",
        "standout_metric_name": "Fastest Single Tree",
        "accent_color": "#2563eb",
    },
    "random-forest": {
        "slug": "random-forest",
        "title": "Random Forest",
        "class_name": "RandomForestRegressor / RandomForestClassifier (oob_score=True)",
        "key_strength": "Bagging ensemble averaging decorrelated trees with Out-Of-Bag (OOB) internal cross-validation; highly resilient against noise, extreme outliers, and overfitting.",
        "feature_selection_strategy": "Random feature subsampling (sqrt/p) at each split, decorrelating individual decision trees.",
        "standout_metric_name": "Best OOB Score & Generalization",
        "accent_color": "#059669",
    },
    "adaboost": {
        "slug": "adaboost",
        "title": "AdaBoost",
        "class_name": "AdaBoostRegressor / AdaBoostClassifier",
        "key_strength": "Adaptive sequential boosting that iteratively reweights hardest-to-predict extreme peak load timestamps, converting weak base learners into an agile forecaster.",
        "feature_selection_strategy": "Global feature evaluation weighted by sample pseudo-residuals across successive decision trees.",
        "standout_metric_name": "Adaptive Peak-Demand Reweighting",
        "accent_color": "#d97706",
    },
    "gradient-boosting": {
        "slug": "gradient-boosting",
        "title": "Gradient Boosting",
        "class_name": "GradientBoostingRegressor / GradientBoostingClassifier",
        "key_strength": "Stage-wise residual gradient descent optimization directly minimizing MSE loss; captures subtle seasonal transitions and continuous daily load curvature.",
        "feature_selection_strategy": "Greedy coordinate descent in function space over all features, fitting subsequent trees to pseudo-residuals.",
        "standout_metric_name": "Smooth Residual Minimization",
        "accent_color": "#7c3aed",
    },
    "xgboost": {
        "slug": "xgboost",
        "title": "XGBoost",
        "class_name": "xgb.XGBRegressor / xgb.XGBClassifier",
        "key_strength": "Regularized tree boosting with 2nd-order Taylor loss expansion, weighted quantile sketches, and extreme cache-optimized parallel execution.",
        "feature_selection_strategy": "Histogram-based column subsampling per tree/split combined with exact L1/L2 leaf weight regularization.",
        "standout_metric_name": "Lowest Test RMSE & Top R²",
        "accent_color": "#dc2626",
    },
    "lightgbm": {
        "slug": "lightgbm",
        "title": "LightGBM",
        "class_name": "lgb.LGBMRegressor / lgb.LGBMClassifier",
        "key_strength": "Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) with leaf-wise tree growth, achieving blazing fast training and top-tier accuracy.",
        "feature_selection_strategy": "Column subsampling + Histogram binning of continuous features into discrete bins with Exclusive Feature Bundling (EFB).",
        "standout_metric_name": "Fastest Training & Leaf-wise Speed",
        "accent_color": "#0891b2",
    },
}


def prepare_feature_data(save_csv: bool = True) -> pd.DataFrame:
    """Loads raw AEP hourly data, computes temporal, lag, and rolling features chronologically,

    and exports AEP_hourly_preprocessed.csv.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset missing at: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime", "AEP_MW"])
    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).reset_index(drop=True)

    # 1. Temporal features
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["quarter"] = df["Datetime"].dt.quarter

    # 2. Autoregressive lags (strictly causal using past observations)
    df["lag_1"] = df["AEP_MW"].shift(1)
    df["lag_24"] = df["AEP_MW"].shift(24)
    df["lag_168"] = df["AEP_MW"].shift(168)

    # 3. Rolling statistics (strictly causal using shifted historical window)
    df["rolling_mean_24h"] = df["AEP_MW"].shift(1).rolling(24).mean()
    df["rolling_std_24h"] = df["AEP_MW"].shift(1).rolling(24).std()

    # Binary high demand target for classification metrics
    median_val = float(df["AEP_MW"].median())
    df["HighDemand"] = (df["AEP_MW"] >= median_val).astype(int)

    # Drop NaNs created by shifts (first 168 rows)
    df = df.dropna().reset_index(drop=True)

    if save_csv:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PREPROCESSED_DATA_PATH, index=False)

    return df


def train_and_evaluate_all_models(force_retrain: bool = False) -> dict:
    """Trains the 6 strictly defined tree/ensemble models,

    computes Train vs Test R2, Accuracy, RMSE, MAE, training time,
    and returns rich explorer metadata.
    """
    if not force_retrain and CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if "models" in cached and len(cached["models"]) == 6:
                return cached
        except Exception:
            pass

    df = prepare_feature_data(save_csv=True)
    median_threshold = float(df["AEP_MW"].median())

    # Chronological split: 80% train, 20% test (preserving time series causality)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[ACTIVE_FEATURES]
    y_train = train_df["AEP_MW"]
    y_train_clf = train_df["HighDemand"]

    X_test = test_df[ACTIVE_FEATURES]
    y_test = test_df["AEP_MW"]
    y_test_clf = test_df["HighDemand"]

    results = []

    # 1. Decision Tree
    t0 = time.time()
    dt = DecisionTreeRegressor(max_depth=12, random_state=42)
    dt.fit(X_train, y_train)
    t_dt = time.time() - t0
    train_pred_dt = dt.predict(X_train)
    test_pred_dt = dt.predict(X_test)
    results.append(_format_model_result("decision-tree", dt, t_dt, y_train, train_pred_dt, y_test, test_pred_dt, test_df, median_threshold))

    # 2. Random Forest (with oob_score=True)
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=50, max_depth=16, oob_score=True, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    train_pred_rf = rf.predict(X_train)
    test_pred_rf = rf.predict(X_test)
    results.append(_format_model_result("random-forest", rf, t_rf, y_train, train_pred_rf, y_test, test_pred_rf, test_df, median_threshold, oob_score=float(rf.oob_score_)))

    # 3. AdaBoost
    t0 = time.time()
    ada = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=6), n_estimators=30, learning_rate=0.05, random_state=42)
    ada.fit(X_train, y_train)
    t_ada = time.time() - t0
    train_pred_ada = ada.predict(X_train)
    test_pred_ada = ada.predict(X_test)
    results.append(_format_model_result("adaboost", ada, t_ada, y_train, train_pred_ada, y_test, test_pred_ada, test_df, median_threshold))

    # 4. Gradient Boosting
    t0 = time.time()
    gb = GradientBoostingRegressor(n_estimators=60, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    t_gb = time.time() - t0
    train_pred_gb = gb.predict(X_train)
    test_pred_gb = gb.predict(X_test)
    results.append(_format_model_result("gradient-boosting", gb, t_gb, y_train, train_pred_gb, y_test, test_pred_gb, test_df, median_threshold))

    # 5. XGBoost
    if xgb is not None:
        t0 = time.time()
        xg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, random_state=42, n_jobs=4)
        xg.fit(X_train, y_train)
        t_xg = time.time() - t0
        train_pred_xg = xg.predict(X_train)
        test_pred_xg = xg.predict(X_test)
        results.append(_format_model_result("xgboost", xg, t_xg, y_train, train_pred_xg, y_test, test_pred_xg, test_df, median_threshold))
    else:
        results.append(_format_fallback_result("xgboost", "xgboost library not installed"))

    # 6. LightGBM
    if lgb is not None:
        t0 = time.time()
        lg = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, random_state=42, verbose=-1)
        lg.fit(X_train, y_train)
        t_lg = time.time() - t0
        train_pred_lg = lg.predict(X_train)
        test_pred_lg = lg.predict(X_test)
        results.append(_format_model_result("lightgbm", lg, t_lg, y_train, train_pred_lg, y_test, test_pred_lg, test_df, median_threshold))
    else:
        results.append(_format_fallback_result("lightgbm", "lightgbm library not installed"))

    # Assign dynamic standout metrics based on actual evaluation
    # Lowest Test RMSE, Best OOB Score, Fastest Training
    best_rmse_model = min(results, key=lambda m: m["metrics"]["test_rmse"])
    fastest_model = min(results, key=lambda m: m["metrics"]["training_time_sec"])
    best_r2_model = max(results, key=lambda m: m["metrics"]["test_r2"])

    for m in results:
        slug = m["slug"]
        if slug == "random-forest":
            m["standout_metric"] = f"Top OOB Score: {m['metrics'].get('oob_score', 0):.4f}"
        elif slug == best_rmse_model["slug"]:
            m["standout_metric"] = f"Lowest Test RMSE: {m['metrics']['test_rmse']:.2f} MW"
        elif slug == fastest_model["slug"]:
            m["standout_metric"] = f"Fastest Training: {m['metrics']['training_time_sec']:.2f}s"
        elif slug == best_r2_model["slug"]:
            m["standout_metric"] = f"Highest Test R²: {m['metrics']['test_r2']:.4f}"
        elif slug == "decision-tree":
            m["standout_metric"] = f"Instant Single-Tree ({m['metrics']['training_time_sec']:.2f}s)"
        elif slug == "adaboost":
            m["standout_metric"] = "Iterative Hard-Sample Reweighting"
        elif slug == "gradient-boosting":
            m["standout_metric"] = f"Stage-Wise Gradient Residuals (R² {m['metrics']['test_r2']:.4f})"

    # Global leaderboard sorted by test R2 descending
    ranked_models = sorted(results, key=lambda m: m["metrics"]["test_r2"], reverse=True)
    for rank, m in enumerate(ranked_models, 1):
        m["rank"] = rank

    summary = {
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_used": ACTIVE_FEATURES,
        "feature_count": len(ACTIVE_FEATURES),
        "temporal_features": ["hour", "dayofweek", "month", "is_weekend", "quarter"],
        "lag_features": ["lag_1", "lag_24", "lag_168"],
        "rolling_features": ["rolling_mean_24h", "rolling_std_24h"],
        "target_variable": "AEP_MW",
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": results,
        "ranked_models": ranked_models,
    }

    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save model cache: {e}")

    return summary


def _format_model_result(slug, model, train_time, y_train, train_pred, y_test, test_pred, test_df, median_val, oob_score=None):
    cfg = MODEL_METADATA_CONFIG[slug]

    train_r2 = float(r2_score(y_train, train_pred))
    test_r2 = float(r2_score(y_test, test_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    train_mae = float(mean_absolute_error(y_train, train_pred))
    test_mae = float(mean_absolute_error(y_test, test_pred))

    # Binary High Demand threshold accuracy
    train_acc = float(accuracy_score(y_train >= median_val, train_pred >= median_val))
    test_acc = float(accuracy_score(y_test >= median_val, test_pred >= median_val))

    # Sample preview of 6 test observations
    sample_indices = [len(test_df) - 6 + i for i in range(6)]
    sample_preview = []
    for idx in sample_indices:
        dt_val = str(test_df.iloc[idx]["Datetime"])
        actual_val = float(y_test.iloc[idx])
        pred_val = float(test_pred[idx])
        err = float(abs(actual_val - pred_val))
        sample_preview.append({
            "datetime": dt_val,
            "actual_mw": round(actual_val, 1),
            "predicted_mw": round(pred_val, 1),
            "abs_error": round(err, 1),
            "pct_error": round((err / actual_val) * 100, 2) if actual_val > 0 else 0.0,
        })

    metrics = {
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        "train_accuracy": round(train_acc * 100, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "train_rmse": round(train_rmse, 2),
        "test_rmse": round(test_rmse, 2),
        "train_mae": round(train_mae, 2),
        "test_mae": round(test_mae, 2),
        "training_time_sec": round(train_time, 2),
    }
    if oob_score is not None:
        metrics["oob_score"] = round(oob_score, 4)

    return {
        "slug": slug,
        "title": cfg["title"],
        "class_name": cfg["class_name"],
        "key_strength": cfg["key_strength"],
        "feature_selection_strategy": cfg["feature_selection_strategy"],
        "standout_metric": cfg["standout_metric_name"],
        "features_used": ACTIVE_FEATURES,
        "metrics": metrics,
        "sample_preview": sample_preview,
        "accent_color": cfg["accent_color"],
    }


def _format_fallback_result(slug, error_msg):
    cfg = MODEL_METADATA_CONFIG[slug]
    return {
        "slug": slug,
        "title": cfg["title"],
        "class_name": cfg["class_name"],
        "key_strength": cfg["key_strength"],
        "feature_selection_strategy": cfg["feature_selection_strategy"],
        "standout_metric": "Unavailable",
        "features_used": ACTIVE_FEATURES,
        "metrics": {
            "train_r2": 0.0,
            "test_r2": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "train_rmse": 0.0,
            "test_rmse": 0.0,
            "train_mae": 0.0,
            "test_mae": 0.0,
            "training_time_sec": 0.0,
        },
        "sample_preview": [],
        "accent_color": cfg["accent_color"],
        "error": error_msg,
    }


if __name__ == "__main__":
    print("Training and benchmarking all 6 models...")
    data = train_and_evaluate_all_models(force_retrain=True)
    print(f"Successfully trained {len(data['models'])} models.")
    for m in data["models"]:
        print(f"- {m['title']}: Test R2={m['metrics']['test_r2']}, Test RMSE={m['metrics']['test_rmse']} MW, Time={m['metrics']['training_time_sec']}s | Standout: {m['standout_metric']}")
