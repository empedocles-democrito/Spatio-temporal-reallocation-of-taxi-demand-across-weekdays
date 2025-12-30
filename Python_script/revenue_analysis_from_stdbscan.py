#!/usr/bin/env python3
# ============================================================
# Revenue analysis from ST-DBSCAN clusters (FULL WEEK)
#
# Input:
#   Taxi_clusters_STDBSCAN_fullweek.csv
#
# Output:
#   - clusters_revenue_t30.csv
#   - daily_revenue.csv
#   - weekday_revenue_summary.csv
#   - revenue_time_series.png
#   - revenue_by_weekday_boxplot.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
INPUT_CSV = (
    "Taxi_clusters_STDBSCAN_fullweek.csv"
)

OUT_DIR = Path(
    "/data/processed"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load data (robust CSV parsing)
# ------------------------------------------------------------
print(f"📥 Leyendo: {INPUT_CSV}")

df = pd.read_csv(
    INPUT_CSV,
    sep=None,          # autodetect delimiter
    engine="python"
)

print("🔎 Columnas:", df.columns.tolist())

# ------------------------------------------------------------
# Fix fare_amount (CRITICAL FIX)
# ------------------------------------------------------------
df["fare_amount"] = (
    df["fare_amount"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")

# ------------------------------------------------------------
# Parse datetime + derive fields
# ------------------------------------------------------------
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

df["pickup_date"] = df["pickup_datetime"].dt.date.astype(str)
df["year"] = df["pickup_datetime"].dt.year
df["weekday"] = df["pickup_datetime"].dt.day_name()

# ------------------------------------------------------------
# Filter valid clustered points
# ------------------------------------------------------------
df = df[
    (df["cluster"] != -1) &
    (~df["fare_amount"].isna()) &
    (~df["pickup_datetime"].isna())
].copy()

print(f"✅ Registros válidos: {len(df)}")

if len(df) == 0:
    raise RuntimeError("❌ No hay datos válidos después de limpieza")

# ------------------------------------------------------------
# Revenue by cluster × interval
# ------------------------------------------------------------
clusters_revenue = (
    df
    .groupby(["cluster", "interval30"], as_index=False)
    .agg(
        cluster_revenue=("fare_amount", "sum"),
        n_trips=("fare_amount", "size")
    )
)

clusters_revenue.to_csv(
    OUT_DIR / "clusters_revenue_t30.csv",
    index=False
)

print("💾 clusters_revenue_t30.csv guardado")

# ------------------------------------------------------------
# Daily revenue
# ------------------------------------------------------------
daily_revenue = (
    df
    .groupby(["pickup_date", "weekday", "year"], as_index=False)
    .agg(
        daily_revenue=("fare_amount", "sum"),
        daily_trips=("fare_amount", "size")
    )
)

daily_revenue.to_csv(
    OUT_DIR / "daily_revenue.csv",
    index=False
)

print("💾 daily_revenue.csv guardado")

# ------------------------------------------------------------
# Weekday summary
# ------------------------------------------------------------
weekday_summary = (
    daily_revenue
    .groupby("weekday", as_index=False)
    .agg(
        avg_daily_revenue=("daily_revenue", "mean"),
        sd_daily_revenue=("daily_revenue", "std"),
        avg_daily_trips=("daily_trips", "mean"),
        n_days=("daily_revenue", "size")
    )
)

weekday_summary.to_csv(
    OUT_DIR / "weekday_revenue_summary.csv",
    index=False
)

print("💾 weekday_revenue_summary.csv guardado")

# ------------------------------------------------------------
# Figure 1: Revenue time series
# ------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(
    pd.to_datetime(daily_revenue["pickup_date"]),
    daily_revenue["daily_revenue"],
    lw=1
)
plt.xlabel("Date")
plt.ylabel("Daily revenue")
plt.title("Daily ride-hailing revenue (Manhattan)")
plt.tight_layout()

plt.savefig(OUT_DIR / "revenue_time_series.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Figure 2: Revenue by weekday (boxplot)
# ------------------------------------------------------------
order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

data = [
    daily_revenue.loc[daily_revenue["weekday"] == d, "daily_revenue"].values
    for d in order
    if d in daily_revenue["weekday"].unique()
]

labels = [d for d in order if d in daily_revenue["weekday"].unique()]

if len(data) > 0:
    plt.figure(figsize=(8, 4))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("Daily revenue")
    plt.title("Distribution of daily revenue by weekday")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "revenue_by_weekday_boxplot.png", dpi=300)
    plt.close()
else:
    print("⚠ No hay datos suficientes para boxplot")

print("🖼 Figuras guardadas")
print("✅ Revenue analysis COMPLETADO")
