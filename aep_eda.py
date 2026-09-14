import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from load_data import load_data

# Configurations
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
CACHE_FILE = os.path.join(CHARTS_DIR, "results_cache.json")
EXPECTED_CHARTS = {
    "missing_values.png", "energy_boxplot.png", "energy_trend.png",
    "monthly_seasonal_trend.png", "daily_consumption_trend.png",
    "hourly_load_profile.png", "energy_distribution.png",
    "annual_energy_growth.png", "weekday_weekend_comparison.png",
    "seasonal_boxplot.png", "daily_peaks_mins.png", "weekly_pattern_heatmap.png",
    "rolling_demand_variability.png", "top_demand_days.png",
}

def run_eda(force_run=False):
    """
    Executes the 13 Energy Consumption EDA tasks and generates charts.
    Uses JSON caching to make subsequent loads instant.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # CHECK CACHE
    if not force_run and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached_results = json.load(f)
            
            charts_list = cached_results.get("charts", [])
            cached_filenames = {
                c["filename"] for c in charts_list if isinstance(c, dict) and "filename" in c
            }
            all_exist = True
            for c in charts_list:
                fname = c["filename"] if isinstance(c, dict) else c
                if not os.path.exists(os.path.join(CHARTS_DIR, fname)):
                    all_exist = False
                    break
            
            is_new_format = len(charts_list) > 0 and isinstance(charts_list[0], dict)
            
            if all_exist and is_new_format and cached_filenames == EXPECTED_CHARTS:
                print("\n========== RETURNING CACHED EDA RESULTS INSTANTLY ==========")
                return cached_results
        except Exception as e:
            print("Cache read failed, running EDA fresh:", e)

    print("\n========== AEP ENERGY EDA STARTED ==========")

    # Load data
    data = load_data()
    print("=" * 80)
    print("Data loaded. Shape:", data.shape)
    print("=" * 80)

    charts = []
    original_columns = set(data.columns)

    # Ensure Datetime conversion
    if "Datetime" in data.columns:
        data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")

    # Helper function to save plots safely
    def _save(filename):
        plt.tight_layout()
        path = os.path.join(CHARTS_DIR, filename)
        plt.savefig(path, bbox_inches="tight", dpi=100)
        plt.close("all")

    # 1. MISSING VALUES ANALYSIS
    print("\n" + "=" * 80)
    print("1. MISSING VALUES ANALYSIS")
    print("=" * 80)
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_pct"] > 0].sort_values(by="missing_count", ascending=False)
    print("Missing columns:\n", missing_df)

    # Even if no values are missing, save a blank/success placeholder plot to satisfy the task
    plt.figure(figsize=(10, 5))
    if not missing_df.empty:
        sns.barplot(x=missing_df.index, y=missing_df["missing_pct"], palette="Reds_r")
        plt.ylabel("Percentage of Missing Values")
        plt.title("Missing Values by Column")
    else:
        plt.text(0.5, 0.5, "No Missing Values Found\n(100% Complete Dataset)", 
                 ha="center", va="center", fontsize=14, color="green", weight="bold")
        plt.title("Dataset Completeness Check")
    _save("missing_values.png")
    charts.append("missing_values.png")

    # 2. DUPLICATE ROWS CHECK
    print("\n" + "=" * 80)
    print("2. DUPLICATE ROWS CHECK")
    print("=" * 80)
    duplicate_count = int(data.duplicated().sum())
    print("Duplicate rows count:", duplicate_count)

    # 3. ENERGY CONSUMPTION BASIC STATISTICS (AEP_MW)
    print("\n" + "=" * 80)
    print("3. ENERGY CONSUMPTION BASIC STATISTICS")
    print("=" * 80)
    target_counts = {}
    if "AEP_MW" in data.columns:
        stats = data["AEP_MW"].describe()
        print(stats)
        target_counts = {
            "Min": float(stats["min"]),
            "25%": float(stats["25%"]),
            "50% (Median)": float(stats["50%"]),
            "75%": float(stats["75%"]),
            "Max": float(stats["max"]),
            "Mean": float(stats["mean"])
        }
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(y="AEP_MW", data=data, color="#f43f5e") # Coral Rose
        plt.title("Energy Consumption (AEP_MW) Boxplot")
        plt.ylabel("Megawatts (MW)")
        _save("energy_boxplot.png")
        charts.append("energy_boxplot.png")

    # 4. OVERALL TIME SERIES TREND (DAILY AGGREGATE)
    print("\n" + "=" * 80)
    print("4. OVERALL TIME SERIES TREND")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        # Group by date for a smooth time trend
        daily_data = data.groupby(data["Datetime"].dt.date)["AEP_MW"].mean()
        plt.figure(figsize=(12, 6))
        plt.plot(daily_data.index, daily_data.values, color="#3b82f6", linewidth=1) # Royal Blue
        plt.title("American Electric Power (AEP) Daily Average Energy Trend")
        plt.xlabel("Timeline")
        plt.ylabel("Average Megawatts (MW)")
        plt.grid(True, linestyle="--", alpha=0.3)
        _save("energy_trend.png")
        charts.append("energy_trend.png")

    # 5. MONTHLY CONSUMPTION TREND (SEASONALITY)
    print("\n" + "=" * 80)
    print("5. MONTHLY CONSUMPTION TREND (SEASONALITY)")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        data["MonthNum"] = data["Datetime"].dt.month
        data["Month"] = data["Datetime"].dt.month_name()
        monthly_avg = data.groupby(["MonthNum", "Month"])["AEP_MW"].mean().reset_index()
        print(monthly_avg)

        plt.figure(figsize=(10, 5))
        sns.barplot(x="Month", y="AEP_MW", data=monthly_avg, palette="coolwarm")
        plt.title("Average Energy Consumption by Month (Seasonal Peaks)")
        plt.xlabel("Month")
        plt.ylabel("Average Megawatts (MW)")
        plt.xticks(rotation=30)
        _save("monthly_seasonal_trend.png")
        charts.append("monthly_seasonal_trend.png")

    # 6. DAILY CONSUMPTION TREND (DAY OF WEEK)
    print("\n" + "=" * 80)
    print("6. DAILY CONSUMPTION TREND (DAY OF WEEK)")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        data["DayOfWeekNum"] = data["Datetime"].dt.dayofweek
        data["DayOfWeek"] = data["Datetime"].dt.day_name()
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_avg = data.groupby(["DayOfWeekNum", "DayOfWeek"])["AEP_MW"].mean().reset_index()
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x="DayOfWeek", y="AEP_MW", data=weekly_avg, order=days_order, palette="Spectral")
        plt.title("Average Energy Consumption by Day of Week (Weekday vs Weekend)")
        plt.xlabel("Day of Week")
        plt.ylabel("Average Megawatts (MW)")
        _save("daily_consumption_trend.png")
        charts.append("daily_consumption_trend.png")

    # 7. HOURLY CONSUMPTION TREND (HOURLY LOAD PROFILE)
    print("\n" + "=" * 80)
    print("7. HOURLY CONSUMPTION TREND (HOURLY LOAD PROFILE)")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        data["Hour"] = data["Datetime"].dt.hour
        hourly_avg = data.groupby("Hour")["AEP_MW"].mean()

        plt.figure(figsize=(10, 5))
        plt.plot(hourly_avg.index, hourly_avg.values, marker="o", color="#ec4899", linewidth=2) # Rosepink
        plt.title("Hourly Load Profile (Average Energy Demand by Hour of Day)")
        plt.xlabel("Hour of Day (0-23)")
        plt.ylabel("Average Megawatts (MW)")
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle="--", alpha=0.3)
        _save("hourly_load_profile.png")
        charts.append("hourly_load_profile.png")

    # 8. DISTRIBUTION PROFILE OF ENERGY USAGE
    print("\n" + "=" * 80)
    print("8. DISTRIBUTION PROFILE OF ENERGY USAGE")
    print("=" * 80)
    if "AEP_MW" in data.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data["AEP_MW"], kde=True, color="#3b82f6", bins=50)
        plt.title("Probability Distribution Profile of AEP Energy Consumption")
        plt.xlabel("Megawatts (MW)")
        plt.ylabel("Frequency")
        _save("energy_distribution.png")
        charts.append("energy_distribution.png")

    # 9. ANNUAL CONSUMPTION GROWTH
    print("\n" + "=" * 80)
    print("9. ANNUAL CONSUMPTION GROWTH")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        annual_avg = data.groupby(data["Datetime"].dt.year)["AEP_MW"].mean()
        plt.figure(figsize=(10, 5))
        plt.plot(annual_avg.index, annual_avg.values, marker="o", color="#16a34a", linewidth=2)
        plt.title("Year-over-Year Average Energy Consumption")
        plt.xlabel("Year")
        plt.ylabel("Average Megawatts (MW)")
        plt.grid(True, linestyle="--", alpha=0.3)
        _save("annual_energy_growth.png")
        charts.append("annual_energy_growth.png")

    # 10. WEEKDAY VERSUS WEEKEND DEMAND
    print("\n" + "=" * 80)
    print("10. WEEKDAY VERSUS WEEKEND DEMAND")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        comparison = pd.DataFrame({
            "AEP_MW": data["AEP_MW"],
            "Day Type": np.where(data["Datetime"].dt.dayofweek < 5, "Weekday", "Weekend"),
        })
        plt.figure(figsize=(8, 5))
        sns.violinplot(x="Day Type", y="AEP_MW", data=comparison, hue="Day Type", legend=False,
                       palette={"Weekday": "#2563eb", "Weekend": "#f97316"})
        plt.title("Energy Demand Distribution: Weekday vs Weekend")
        plt.xlabel("")
        plt.ylabel("Megawatts (MW)")
        _save("weekday_weekend_comparison.png")
        charts.append("weekday_weekend_comparison.png")

    # 11. SEASONAL DEMAND SPREAD
    print("\n" + "=" * 80)
    print("11. SEASONAL DEMAND SPREAD")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                      5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall",
                      10: "Fall", 11: "Fall"}
        seasonal = data[["Datetime", "AEP_MW"]].copy()
        seasonal["Season"] = seasonal["Datetime"].dt.month.map(season_map)
        plt.figure(figsize=(9, 5))
        sns.boxplot(x="Season", y="AEP_MW", data=seasonal,
                    order=["Winter", "Spring", "Summer", "Fall"], color="#60a5fa")
        plt.title("Energy Consumption Spread by Season")
        plt.xlabel("Season")
        plt.ylabel("Megawatts (MW)")
        _save("seasonal_boxplot.png")
        charts.append("seasonal_boxplot.png")

    # 12. DAILY PEAK AND MINIMUM DEMAND
    print("\n" + "=" * 80)
    print("12. DAILY PEAK AND MINIMUM DEMAND")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        daily_extremes = data.groupby(data["Datetime"].dt.date)["AEP_MW"].agg(["max", "min"])
        plt.figure(figsize=(12, 5))
        plt.plot(daily_extremes.index, daily_extremes["max"], label="Daily Peak", color="#dc2626")
        plt.plot(daily_extremes.index, daily_extremes["min"], label="Daily Minimum", color="#0891b2")
        plt.title("Daily Maximum and Minimum Energy Demand")
        plt.xlabel("Date")
        plt.ylabel("Megawatts (MW)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        _save("daily_peaks_mins.png")
        charts.append("daily_peaks_mins.png")

    # 13. WEEKLY PATTERN HEATMAP
    print("\n" + "=" * 80)
    print("13. WEEKLY PATTERN HEATMAP")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        heatmap_data = data.assign(
            Day=data["Datetime"].dt.day_name(), Hour=data["Datetime"].dt.hour
        ).pivot_table(index="Day", columns="Hour", values="AEP_MW", aggfunc="mean")
        heatmap_data = heatmap_data.reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                                             "Friday", "Saturday", "Sunday"])
        plt.figure(figsize=(14, 5))
        sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={"label": "Average MW"})
        plt.title("Average Energy Demand by Hour and Day of Week")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")
        _save("weekly_pattern_heatmap.png")
        charts.append("weekly_pattern_heatmap.png")

    # 14. ROLLING DEMAND VARIABILITY
    print("\n" + "=" * 80)
    print("14. ROLLING DEMAND VARIABILITY")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        daily_avg = data.set_index("Datetime")["AEP_MW"].resample("D").mean()
        rolling_std = daily_avg.rolling(7).std()
        plt.figure(figsize=(12, 5))
        plt.plot(rolling_std.index, rolling_std.values, color="#7c3aed", linewidth=1.5)
        plt.title("7-Day Rolling Variability of Daily Energy Demand")
        plt.xlabel("Date")
        plt.ylabel("Rolling Standard Deviation (MW)")
        plt.grid(True, linestyle="--", alpha=0.3)
        _save("rolling_demand_variability.png")
        charts.append("rolling_demand_variability.png")

    # 15. TOP DEMAND DAYS
    print("\n" + "=" * 80)
    print("15. TOP 10 HIGHEST-DEMAND DAYS")
    print("=" * 80)
    if "Datetime" in data.columns and "AEP_MW" in data.columns:
        top_days = data.groupby(data["Datetime"].dt.date)["AEP_MW"].mean().nlargest(10).sort_values()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_days.values, y=[str(day) for day in top_days.index], color="#ea580c")
        plt.title("Top 10 Days by Average Energy Demand")
        plt.xlabel("Average Megawatts (MW)")
        plt.ylabel("Date")
        _save("top_demand_days.png")
        charts.append("top_demand_days.png")

    print("\n========== EDA COMPLETED ==========")
    print("Charts generated:", len(charts))

    # Map raw chart names to human-readable titles
    chart_title_map = {
        "missing_values.png": "Missing Values Analysis Chart",
        "energy_boxplot.png": "Megawatts (MW) Summary Statistics Boxplot",
        "energy_trend.png": "Daily Average Energy Demand Trend (Historical)",
        "monthly_seasonal_trend.png": "Average Demand by Month (Seasonal Analysis)",
        "daily_consumption_trend.png": "Average Demand by Day of Week",
        "hourly_load_profile.png": "24-Hour Average Energy Load Profile",
        "annual_energy_growth.png": "Year-over-Year Energy Consumption Growth",
        "weekday_weekend_comparison.png": "Demand Density Profile: Weekday vs. Weekend",
        "seasonal_boxplot.png": "Energy Consumption Spread by Season",
        "daily_peaks_mins.png": "Daily Max (Peak) vs. Min Demand Gap Trend",
        "weekly_pattern_heatmap.png": "Load Heatmap: Hour of Day vs. Day of Week",
        "energy_distribution.png": "Probability Distribution Profile of Energy Consumption",
        "rolling_demand_variability.png": "7-Day Rolling Variability of Daily Demand",
        "top_demand_days.png": "Top 10 Highest-Demand Days"
    }

    formatted_charts = []
    for fname in charts:
        formatted_charts.append({
            "filename": fname,
            "title": chart_title_map.get(fname, fname.replace(".png", "").replace("_", " ").title())
        })

    results = {
        "n_rows": len(data),
        "n_cols": len(original_columns),
        "duplicate_count": duplicate_count,
        "missing": {col: int(cnt) for col, cnt in missing.items() if cnt > 0},
        "target_counts": target_counts,
        "charts": formatted_charts,
    }

    # Save to cache
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(results, f)
        print("EDA results cached successfully.")
    except Exception as e:
        print("Failed to save EDA results to cache:", e)

    return results

if __name__ == "__main__":
    run_eda(force_run=True)
