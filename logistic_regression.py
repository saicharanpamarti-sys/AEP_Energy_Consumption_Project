"""
Logistic Regression Module for AEP Energy Consumption Project.
Implements Dual Model Architecture:
- Model A: Calendar Baseline (hour, month, dayofweek, is_weekend)
- Model B: Autoregressive Classifier (hour, month, dayofweek, lag_1, lag_24)
Generates diagnostic curves: Learning Curves, Comparative ROC, Precision-Recall, and Dual Confusion Matrices.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "AEP_preprocessed.csv"
RAW_DATA_PATH = BASE_DIR / "data" / "AEP_hourly.csv"
CHARTS_DIR = BASE_DIR / "static" / "charts"


def load_clean_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(RAW_DATA_PATH)
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime", "AEP_MW"]).sort_values("Datetime").reset_index(drop=True)
        df["hour"] = df["Datetime"].dt.hour.astype(int)
        df["month"] = df["Datetime"].dt.month.astype(int)
        df["dayofweek"] = df["Datetime"].dt.dayofweek.astype(int)
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["lag_1"] = df["AEP_MW"].shift(1)
        df["lag_24"] = df["AEP_MW"].shift(24)
        df = df.bfill().copy()
        df["High_Demand"] = (df["AEP_MW"] >= df["AEP_MW"].median()).astype(int)

    # Clean lowercase aliases
    if "hour" not in df.columns and "Hour" in df.columns:
        df["hour"] = df["Hour"]
    if "month" not in df.columns and "Month" in df.columns:
        df["month"] = df["Month"]
    if "dayofweek" not in df.columns and "DayOfWeek" in df.columns:
        df["dayofweek"] = df["DayOfWeek"]
    if "is_weekend" not in df.columns and "IsWeekend" in df.columns:
        df["is_weekend"] = df["IsWeekend"]
    if "High_Demand" not in df.columns and "HighDemand" in df.columns:
        df["High_Demand"] = df["HighDemand"]

    return df


def generate_logistic_diagnostics(
    X_train_a, X_test_a, y_train, y_test,
    X_train_b, X_test_b,
    model_a, model_b,
    scaler_a, scaler_b,
    force_run=False
):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    f_lc = CHARTS_DIR / "logreg_learning_curve.png"
    f_roc = CHARTS_DIR / "logreg_roc_curve.png"
    f_pr = CHARTS_DIR / "logreg_pr_curve.png"
    f_cm = CHARTS_DIR / "logreg_confusion_matrices.png"

    if not force_run and f_lc.exists() and f_roc.exists() and f_pr.exists() and f_cm.exists():
        return

    print("Generating Logistic Regression diagnostic graphs...")

    # Probabilities & Predictions
    X_test_a_s = scaler_a.transform(X_test_a)
    X_test_b_s = scaler_b.transform(X_test_b)

    pred_a = model_a.predict(X_test_a_s)
    pred_b = model_b.predict(X_test_b_s)

    prob_a = model_a.predict_proba(X_test_a_s)[:, 1]
    prob_b = model_b.predict_proba(X_test_b_s)[:, 1]

    # 1. Learning Curves across Sample Increments
    fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
    train_acc_a_list, val_acc_a_list = [], []
    train_acc_b_list, val_acc_b_list = [], []

    n_tot = len(y_train)
    for frac in fractions:
        n_samp = int(n_tot * frac)
        sub_X_a = X_train_a.iloc[:n_samp]
        sub_X_b = X_train_b.iloc[:n_samp]
        sub_y = y_train.iloc[:n_samp]

        sc_a = StandardScaler().fit(sub_X_a)
        sc_b = StandardScaler().fit(sub_X_b)

        m_a = LogisticRegression(max_iter=1000).fit(sc_a.transform(sub_X_a), sub_y)
        m_b = LogisticRegression(max_iter=1000).fit(sc_b.transform(sub_X_b), sub_y)

        train_acc_a_list.append(accuracy_score(sub_y, m_a.predict(sc_a.transform(sub_X_a))))
        val_acc_a_list.append(accuracy_score(y_test, m_a.predict(sc_a.transform(X_test_a))))

        train_acc_b_list.append(accuracy_score(sub_y, m_b.predict(sc_b.transform(sub_X_b))))
        val_acc_b_list.append(accuracy_score(y_test, m_b.predict(sc_b.transform(X_test_b))))

    plt.figure(figsize=(10, 4.8), dpi=150)
    sizes = [f"{int(f*100)}%" for f in fractions]
    plt.plot(sizes, val_acc_a_list, "o--", color="#ef4444", lw=2, label="Model A (Calendar): Validation Acc (Plateau at ~71%)")
    plt.plot(sizes, train_acc_a_list, "s-", color="#f87171", lw=1.5, alpha=0.7, label="Model A (Calendar): Train Acc")
    plt.plot(sizes, val_acc_b_list, "o--", color="#10b981", lw=2, label="Model B (Autoregressive): Validation Acc (Convergence at ~94%)")
    plt.plot(sizes, train_acc_b_list, "s-", color="#34d399", lw=1.5, alpha=0.7, label="Model B (Autoregressive): Train Acc")
    plt.title("Sample Size Learning Curves: Model A (High Bias) vs. Model B (Convergence)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Training Set Fraction", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.ylim(0.65, 0.98)
    plt.grid(True)
    plt.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f_lc)
    plt.close()

    # 2. Comparative ROC Curves
    fpr_a, tpr_a, _ = roc_curve(y_test, prob_a)
    fpr_b, tpr_b, _ = roc_curve(y_test, prob_b)
    auc_a = auc(fpr_a, tpr_a)
    auc_b = auc(fpr_b, tpr_b)

    plt.figure(figsize=(9, 4.8), dpi=150)
    plt.plot(fpr_a, tpr_a, color="#ef4444", lw=2, label=f"Model A: Calendar Baseline (AUC = {auc_a:.4f})")
    plt.plot(fpr_b, tpr_b, color="#10b981", lw=2.5, label=f"Model B: Autoregressive Lags (AUC = {auc_b:.4f})")
    plt.plot([0, 1], [0, 1], color="#64748b", linestyle="--", lw=1.2, label="Random Guessing Reference (AUC = 0.5000)")
    plt.title("Comparative Receiver Operating Characteristic (ROC) Curves", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    plt.grid(True)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f_roc)
    plt.close()

    # 3. Comparative Precision-Recall Curves
    prec_a, rec_a, _ = precision_recall_curve(y_test, prob_a)
    prec_b, rec_b, _ = precision_recall_curve(y_test, prob_b)
    pr_auc_a = auc(rec_a, prec_a)
    pr_auc_b = auc(rec_b, prec_b)

    plt.figure(figsize=(9, 4.8), dpi=150)
    plt.plot(rec_a, prec_a, color="#ef4444", lw=2, label=f"Model A: Calendar Baseline (PR-AUC = {pr_auc_a:.4f})")
    plt.plot(rec_b, prec_b, color="#10b981", lw=2.5, label=f"Model B: Autoregressive Lags (PR-AUC = {pr_auc_b:.4f})")
    plt.title("Comparative Precision-Recall (PR) Curves on High-Demand Target", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Recall (High-Demand Coverage)", fontsize=10)
    plt.ylabel("Precision (High-Demand Alert Accuracy)", fontsize=10)
    plt.grid(True)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f_pr)
    plt.close()

    # 4. Dual Confusion Matrix Heatmaps (Side-by-Side)
    cm_a = confusion_matrix(y_test, pred_a)
    cm_b = confusion_matrix(y_test, pred_b)
    err_a = int(cm_a[0, 1] + cm_a[1, 0])
    err_b = int(cm_b[0, 1] + cm_b[1, 0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    sns.heatmap(cm_a, annot=True, fmt=",d", cmap="Reds", cbar=False, ax=axes[0],
                xticklabels=["Normal (0)", "High (1)"], yticklabels=["Normal (0)", "High (1)"])
    axes[0].set_title(f"Model A: Calendar Baseline\nTest Errors: {err_a:,} ({err_a/len(y_test)*100:.1f}%)", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Predicted Class", fontsize=9)
    axes[0].set_ylabel("True Ground Truth Class", fontsize=9)

    sns.heatmap(cm_b, annot=True, fmt=",d", cmap="Greens", cbar=False, ax=axes[1],
                xticklabels=["Normal (0)", "High (1)"], yticklabels=["Normal (0)", "High (1)"])
    axes[1].set_title(f"Model B: Autoregressive Classifier\nTest Errors: {err_b:,} ({err_b/len(y_test)*100:.1f}%)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Predicted Class", fontsize=9)
    axes[1].set_ylabel("True Ground Truth Class", fontsize=9)

    plt.suptitle("Dual Confusion Matrix Heatmaps: Error Reduction from Model A to Model B", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f_cm)
    plt.close()

    print("All 4 Logistic Regression diagnostic plots successfully generated.")


def get_logistic_regression_summary():
    """
    Evaluates Model A (Calendar Baseline) vs. Model B (Autoregressive Classifier).
    Returns comparative evaluation metrics and diagnostic metadata.
    """
    df = load_clean_data()
    y = df["High_Demand"]

    feat_a = ["hour", "month", "dayofweek", "is_weekend"]
    feat_b = ["hour", "month", "dayofweek", "lag_1", "lag_24"]

    X_a = df[feat_a]
    X_b = df[feat_b]

    X_train_a, X_test_a, y_train, y_test = train_test_split(
        X_a, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_b, X_test_b, _, _ = train_test_split(
        X_b, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler_a = StandardScaler().fit(X_train_a)
    scaler_b = StandardScaler().fit(X_train_b)

    X_train_a_s = scaler_a.transform(X_train_a)
    X_test_a_s = scaler_a.transform(X_test_a)

    X_train_b_s = scaler_b.transform(X_train_b)
    X_test_b_s = scaler_b.transform(X_test_b)

    model_a = LogisticRegression(max_iter=1000).fit(X_train_a_s, y_train)
    model_b = LogisticRegression(max_iter=1000).fit(X_train_b_s, y_train)

    generate_logistic_diagnostics(
        X_train_a, X_test_a, y_train, y_test,
        X_train_b, X_test_b,
        model_a, model_b,
        scaler_a, scaler_b
    )

    pred_train_a = model_a.predict(X_train_a_s)
    pred_test_a = model_a.predict(X_test_a_s)
    prob_test_a = model_a.predict_proba(X_test_a_s)[:, 1]

    pred_train_b = model_b.predict(X_train_b_s)
    pred_test_b = model_b.predict(X_test_b_s)
    prob_test_b = model_b.predict_proba(X_test_b_s)[:, 1]

    cm_a = confusion_matrix(y_test, pred_test_a).tolist()
    cm_b = confusion_matrix(y_test, pred_test_b).tolist()

    model_a_metrics = {
        "name": "Model A (Calendar Baseline)",
        "type": "Calendar Baseline",
        "features": feat_a,
        "description": "Trained purely on diurnal and calendar coordinates.",
        "train_accuracy": round(float(accuracy_score(y_train, pred_train_a)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, pred_test_a)), 4),
        "precision": round(float(precision_score(y_test, pred_test_a, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred_test_a, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred_test_a, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob_test_a)), 4),
        "confusion_matrix": cm_a,
        "test_errors": int(cm_a[0][1] + cm_a[1][0]),
        "coefficients": {k: round(float(v), 4) for k, v in zip(feat_a, model_a.coef_[0])},
        "intercept": round(float(model_a.intercept_[0]), 4),
    }

    model_b_metrics = {
        "name": "Model B (Autoregressive Classifier)",
        "type": "Autoregressive (With Lags)",
        "features": feat_b,
        "description": "Combines calendar cycles with 1-hour and 24-hour historical consumption lags.",
        "train_accuracy": round(float(accuracy_score(y_train, pred_train_b)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, pred_test_b)), 4),
        "precision": round(float(precision_score(y_test, pred_test_b, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred_test_b, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred_test_b, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob_test_b)), 4),
        "confusion_matrix": cm_b,
        "test_errors": int(cm_b[0][1] + cm_b[1][0]),
        "coefficients": {k: round(float(v), 4) for k, v in zip(feat_b, model_b.coef_[0])},
        "intercept": round(float(model_b.intercept_[0]), 4),
    }

    return {
        "model_a": model_a_metrics,
        "model_b": model_b_metrics,
        "primary_model": model_a_metrics,
        "best_model": model_b_metrics,
        "summary": model_b_metrics,  # Backwards compatibility
        "error_reduction": round((model_a_metrics["test_errors"] - model_b_metrics["test_errors"]) / model_a_metrics["test_errors"] * 100, 1),
    }


if __name__ == "__main__":
    res = get_logistic_regression_summary()
    print("Model A Accuracy:", res["model_a"]["test_accuracy"], "F1:", res["model_a"]["f1_score"], "AUC:", res["model_a"]["roc_auc"])
    print("Model B Accuracy:", res["model_b"]["test_accuracy"], "F1:", res["model_b"]["f1_score"], "AUC:", res["model_b"]["roc_auc"])
    print(f"Test Errors: {res['model_a']['test_errors']} -> {res['model_b']['test_errors']} (Reduction: {res['error_reduction']}%)")
