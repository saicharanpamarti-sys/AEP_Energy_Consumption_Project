import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import holidays
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_preprocessed_data

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PREPROCESSED_DATA_PATH = DATA_DIR / "AEP_hourly_preprocemssed.csv"


def add_holiday_feature(df: pd.DataFrame, date_col: str = 'Datetime') -> pd.DataFrame:
    """Add US holiday feature to the dataframe."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    us_holidays = holidays.US()
    df['is_holiday'] = df[date_col].apply(lambda x: int(x in us_holidays))
    
    return df


def engineer_features(df: pd.DataFrame, date_col: str = 'Datetime') -> pd.DataFrame:
    """
    Engineer calendar features (hour, weekday, month, holiday) and lag features.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Calendar features
    df['hour'] = df[date_col].dt.hour
    df['weekday'] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df[date_col].dt.month
    
    # Holiday feature
    us_holidays = holidays.US()
    df['is_holiday'] = df[date_col].apply(lambda x: int(x in us_holidays))
    
    # Lag features (if not already present)
    if 'lag_1' not in df.columns:
        df['lag_1'] = df['AEP_MW'].shift(1)
    if 'lag_24' not in df.columns:
        df['lag_24'] = df['AEP_MW'].shift(24)
    if 'lag_48' not in df.columns:
        df['lag_48'] = df['AEP_MW'].shift(48)
    
    # Rolling statistics
    df['rolling_mean_24'] = df['AEP_MW'].rolling(window=24, min_periods=1).mean()
    df['rolling_std_24'] = df['AEP_MW'].rolling(window=24, min_periods=1).std()
    
    # Fill missing lag values with forward fill
    df = df.ffill().bfill()
    
    return df


def prepare_regression_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare features and target for regression forecasting."""
    feature_cols = ['hour', 'weekday', 'month', 'is_holiday', 
                    'lag_1', 'lag_24', 'lag_48', 
                    'rolling_mean_24', 'rolling_std_24']
    
    X = df[feature_cols].copy()
    y = df['AEP_MW'].copy()
    
    return X, y


def train_regression_model(X: pd.DataFrame, y: pd.DataFrame, 
                          test_size: float = 0.2) -> Tuple[Any, Dict[str, Any]]:
    """
    Train Random Forest regressor for hourly demand forecasting.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_r2': r2_score(y_test, y_test_pred),
    }
    
    return model, metrics


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect anomalous consumption periods using Isolation Forest.
    """
    feature_cols = ['AEP_MW', 'hour', 'weekday', 'month', 'is_holiday',
                    'lag_1', 'lag_24', 'rolling_mean_24', 'rolling_std_24']
    
    X = df[feature_cols].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest for anomaly detection
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    df_result = df.copy()
    df_result['anomaly'] = np.where(anomaly_labels == -1, 1, 0)
    df_result['anomaly_score'] = anomaly_scores
    
    return df_result


def analyze_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze detected anomalies by time periods."""
    anomalies = df[df['anomaly'] == 1]
    
    analysis = {
        'total_anomalies': len(anomalies),
        'anomaly_percentage': (len(anomalies) / len(df)) * 100,
        'mean_anomaly_demand': anomalies['AEP_MW'].mean(),
        'std_anomaly_demand': anomalies['AEP_MW'].std(),
        'anomalies_by_hour': anomalies['hour'].value_counts().to_dict(),
        'anomalies_by_month': anomalies['month'].value_counts().to_dict(),
        'anomalies_by_weekday': anomalies['weekday'].value_counts().to_dict(),
        'anomalies_on_holidays': anomalies['is_holiday'].sum(),
    }
    
    return analysis


def visualize_forecast(y_true: pd.Series, y_pred: np.ndarray, 
                      title: str = "Electricity Demand Forecast"):
    """Visualize forecast results."""
    plt.figure(figsize=(14, 6))
    
    # Plot first 500 points for clarity
    n_points = min(500, len(y_true))
    plt.plot(y_true.values[:n_points], label='Actual', alpha=0.7)
    plt.plot(y_pred[:n_points], label='Predicted', alpha=0.7)
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Energy Demand (MW)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def visualize_anomalies(df: pd.DataFrame, n_points: int = 1000):
    """Visualize anomalies in the time series."""
    plt.figure(figsize=(14, 6))
    
    plot_df = df.head(n_points).copy()
    
    # Plot normal points
    normal = plot_df[plot_df['anomaly'] == 0]
    anomalies = plot_df[plot_df['anomaly'] == 1]
    
    plt.scatter(normal.index, normal['AEP_MW'], 
                c='blue', label='Normal', s=10, alpha=0.5)
    plt.scatter(anomalies.index, anomalies['AEP_MW'], 
                c='red', label='Anomaly', s=30, alpha=0.8)
    
    plt.xlabel('Time Index')
    plt.ylabel('Energy Demand (MW)')
    plt.title('Anomaly Detection in Electricity Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def run_full_pipeline() -> Dict[str, Any]:
    """
    Run complete pipeline: feature engineering, regression forecasting, and anomaly detection.
    """
    print("Loading preprocessed data...")
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("\n" + "="*60)
    print("REGRESSION: Hourly Demand Forecasting")
    print("="*60)
    
    X, y = prepare_regression_data(df)
    model, metrics = train_regression_model(X, y)
    
    print(f"\nTraining Metrics:")
    print(f"  MAE: {metrics['train_mae']:.2f} MW")
    print(f"  RMSE: {metrics['train_rmse']:.2f} MW")
    print(f"  R²: {metrics['train_r2']:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  MAE: {metrics['test_mae']:.2f} MW")
    print(f"  RMSE: {metrics['test_rmse']:.2f} MW")
    print(f"  R²: {metrics['test_r2']:.4f}")
    
    # Generate predictions for visualization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    y_test_pred = model.predict(X_test)
    
    print("\n" + "="*60)
    print("ANOMALY DETECTION: Identifying Unusual Consumption")
    print("="*60)
    
    df_with_anomalies = detect_anomalies(df)
    anomaly_analysis = analyze_anomalies(df_with_anomalies)
    
    print(f"\nTotal Anomalies Detected: {anomaly_analysis['total_anomalies']}")
    print(f"Anomaly Percentage: {anomaly_analysis['anomaly_percentage']:.2f}%")
    print(f"Mean Demand During Anomalies: {anomaly_analysis['mean_anomaly_demand']:.2f} MW")
    print(f"Anomalies on Holidays: {anomaly_analysis['anomalies_on_holidays']}")
    
    print(f"\nAnomalies by Hour (Top 5):")
    for hour, count in sorted(anomaly_analysis['anomalies_by_hour'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Hour {hour}: {count} anomalies")
    
    print(f"\nAnomalies by Month:")
    for month, count in sorted(anomaly_analysis['anomalies_by_month'].items()):
        print(f"  Month {month}: {count} anomalies")
    
    # Feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Regression Model)")
    print("="*60)
    for feature, importance in sorted_importance:
        print(f"  {feature}: {importance:.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    fig1 = visualize_forecast(y_test, y_test_pred, "Test Set: Actual vs Predicted Demand")
    fig1.savefig(DATA_DIR / "forecast_results.png", dpi=150, bbox_inches='tight')
    
    fig2 = visualize_anomalies(df_with_anomalies, n_points=2000)
    fig2.savefig(DATA_DIR / "anomaly_detection.png", dpi=150, bbox_inches='tight')
    
    print(f"\nVisualizations saved to {DATA_DIR}")
    
    results = {
        'regression_metrics': metrics,
        'feature_importance': dict(sorted_importance),
        'anomaly_analysis': anomaly_analysis,
        'model': model,
        'data_with_anomalies': df_with_anomalies
    }
    
    return results


if __name__ == "__main__":
    results = run_full_pipeline()
