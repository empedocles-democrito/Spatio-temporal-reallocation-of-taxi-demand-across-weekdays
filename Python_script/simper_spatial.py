#!/usr/bin/env python3
# ============================================================
# CAPA A: Hotspots canónicos globales (todos los días juntos)
# - Detecta automáticamente columnas lat/lon
# - DBSCAN espacial por intervalo + tracking por centroides
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from math import cos, radians, sqrt
from sklearn.cluster import DBSCAN

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = Path(
        "Taxi_manhattan_correct_datetime_numbers_with_day.csv"
)
OUTDIR = INPUT_CSV.parent

T_MIN = 30
N_INTERVALS = int(24 * 60 / T_MIN)

EPS_ESPACIAL_M = 350.0
MIN_SAMPLES = 5
MAX_MATCH_DIST = 400.0
TOP_K = 20

# Estas columnas sí deben existir (si no, se detecta/avisa)
CAND_LAT = ["pickup_latitude", "pickup_lat", "lat", "latitude", "PICKUP_LATITUDE", "PICKUP_LAT", "LAT", "LATITUDE"]
CAND_LON = ["pickup_longitude", "pickup_lon", "lon", "longitude", "PICKUP_LONGITUDE", "PICKUP_LON", "LON", "LONGITUDE"]
CAND_DATE = ["pickup_date", "date", "fecha", "PICKUP_DATE", "FECHA"]
CAND_TIME = ["pickup_time", "time", "hora", "PICKUP_TIME", "HORA"]

def pick_col(cols, candidates, name):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def latlon_to_xy(lat, lon, lat0, lon0):
    m_lat = 111_320.0
    m_lon = 111_320.0 * cos(radians(lat0))
    x = (lon - lon0) * m_lon
    y = (lat - lat0) * m_lat
    return x, y

def dist_m(p, q):
    return sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

print("📥 Leyendo datos...")
# Lectura robusta: detecta ; o ,
try:
    df = pd.read_csv(INPUT_CSV, sep=";", decimal=",")
    if "pickup_latitude" not in df.columns:
        raise ValueError
except Exception:
    df = pd.read_csv(INPUT_CSV, sep=",", decimal=".")


# Detectar nombres reales
COL_LAT = pick_col(df.columns, CAND_LAT, "lat")
COL_LON = pick_col(df.columns, CAND_LON, "lon")
COL_DATE = pick_col(df.columns, CAND_DATE, "date")
COL_TIME = pick_col(df.columns, CAND_TIME, "time")

if COL_LAT is None or COL_LON is None or COL_DATE is None or COL_TIME is None:
    print("\n❌ No pude detectar columnas clave.")
    print("Columnas encontradas en el CSV:")
    print(df.columns.tolist())
    raise SystemExit(1)

print(f"✅ Usando columnas: lat={COL_LAT}, lon={COL_LON}, date={COL_DATE}, time={COL_TIME}")

# Tipos numéricos
df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")
df = df.dropna(subset=[COL_LAT, COL_LON, COL_DATE, COL_TIME]).copy()

# Datetime + intervalos intra-día
df["datetime"] = pd.to_datetime(df[COL_DATE].astype(str) + " " + df[COL_TIME].astype(str), errors="coerce")
df = df.dropna(subset=["datetime"]).copy()
df["minutes_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
df["interval"] = (df["minutes_day"] // T_MIN).astype(int)

# Proyección plana global
lat0_all = df[COL_LAT].mean()
lon0_all = df[COL_LON].mean()

xy = df.apply(lambda r: latlon_to_xy(r[COL_LAT], r[COL_LON], lat0_all, lon0_all), axis=1)
df["x"] = [p[0] for p in xy]
df["y"] = [p[1] for p in xy]

print(f"✅ Registros válidos: {len(df):,}")
print(f"✅ Intervalos: {N_INTERVALS} (T_MIN={T_MIN})")

# 1) DBSCAN por intervalo → centroides
print("📍 DBSCAN espacial por intervalo → centroides...")
centroides = []

for intervalo in range(N_INTERVALS):
    df_i = df[df["interval"] == intervalo].copy()
    if len(df_i) < MIN_SAMPLES:
        continue

    labels = DBSCAN(eps=EPS_ESPACIAL_M, min_samples=MIN_SAMPLES).fit_predict(df_i[["x", "y"]])
    df_i["cluster_local"] = labels

    for cid, g in df_i.groupby("cluster_local"):
        if cid == -1:
            continue
        centroides.append({
            "interval": int(intervalo),
            "cluster_local": int(cid),
            "lat": float(g[COL_LAT].mean()),
            "lon": float(g[COL_LON].mean()),
            "x": float(g["x"].mean()),
            "y": float(g["y"].mean()),
            "size": int(len(g))
        })

cent = pd.DataFrame(centroides).sort_values(["interval"]).reset_index(drop=True)
if cent.empty:
    raise RuntimeError("No se encontraron clusters. Prueba bajar MIN_SAMPLES o subir EPS_ESPACIAL_M.")

print(f"✅ Centroides generados: {len(cent):,}")

# 2) Tracking → hotspot_id canónico
print("⏱ Tracking temporal → hotspot_id...")
cent["hotspot_id"] = -1
next_id = 0
prev = {}

for intervalo in range(N_INTERVALS):
    current = cent[cent["interval"] == intervalo]
    if current.empty:
        prev = {}
        continue

    curr = {}
    for idx, r in current.iterrows():
        best = None
        best_d = 1e12
        for hid, p in prev.items():
            d = dist_m((r.x, r.y), (p.x, p.y))
            if d < MAX_MATCH_DIST and d < best_d:
                best = hid
                best_d = d

        if best is None:
            cent.loc[idx, "hotspot_id"] = next_id
            curr[next_id] = r
            next_id += 1
        else:
            cent.loc[idx, "hotspot_id"] = best
            curr[best] = r
    prev = curr

print(f"✅ Hotspots canónicos (K): {next_id}")

# 3) Resumen por hotspot
dur = cent.groupby("hotspot_id")["interval"].agg(
    first_interval="min", last_interval="max", duration_intervals="count"
).reset_index()

agg_size = cent.groupby("hotspot_id")["size"].sum().reset_index().rename(columns={"size": "total_points"})
agg_pos = cent.groupby("hotspot_id")[["lat", "lon"]].mean().reset_index().rename(columns={"lat": "mean_lat", "lon": "mean_lon"})

resumen = dur.merge(agg_size, on="hotspot_id").merge(agg_pos, on="hotspot_id")
resumen = resumen.sort_values(["duration_intervals", "total_points"], ascending=[False, False]).reset_index(drop=True)
topk = resumen.head(TOP_K).copy()

# SAVE
out_cent = OUTDIR / f"hotspots_global_centroides_t{T_MIN}.csv"
out_res  = OUTDIR / f"hotspots_global_resumen_t{T_MIN}.csv"
out_topk = OUTDIR / f"hotspots_global_top{TOP_K}_t{T_MIN}.csv"

cent.to_csv(out_cent, index=False)
resumen.to_csv(out_res, index=False)
topk.to_csv(out_topk, index=False)

print("\n✅ Listo (CAPA A). Guardado:")
print(" -", out_cent)
print(" -", out_res)
print(" -", out_topk)
