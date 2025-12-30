#!/usr/bin/env python3
# ============================================================
# stdbscan_final_calibrated.py
#
# Ejecuta ST-DBSCAN usando parámetros calibrados
# SOBRE TODA LA SEMANA (sin filtrar días).
#
# Input:
#   - Taxi_manhattan_correct_datetime_numbers_with_day.csv
#   - stdbscan_best_config.json
#
# Output:
#   - Taxi_clusters_STDBSCAN_fullweek.csv
# ============================================================

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path("")
DATA_DIR = BASE_DIR / ""

INPUT_CSV = DATA_DIR / "Taxi_manhattan_correct_datetime_numbers_with_day.csv"
CONFIG_JSON = DATA_DIR / "stdbscan_best_config.json"
OUTPUT_CSV = DATA_DIR / "Taxi_clusters_STDBSCAN_fullweek.csv"

# ============================================================
# ST-DBSCAN (espacial + temporal por interval30)
# ============================================================

def st_dbscan(coords, times, eps_spatial, eps_temporal, min_samples):
    n = len(coords)
    labels = -1 * np.ones(n, dtype=int)

    nn = NearestNeighbors(radius=eps_spatial, metric="haversine")
    nn.fit(coords)

    neighbors = nn.radius_neighbors(coords, return_distance=False)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue

        # vecinos espaciales + filtro temporal
        neigh = [
            j for j in neighbors[i]
            if abs(times[j] - times[i]) <= eps_temporal
        ]

        if len(neigh) < min_samples:
            continue

        labels[i] = cluster_id
        stack = list(neigh)

        while stack:
            j = stack.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
                neigh_j = [
                    k for k in neighbors[j]
                    if abs(times[k] - times[j]) <= eps_temporal
                ]
                if len(neigh_j) >= min_samples:
                    stack.extend(neigh_j)

        cluster_id += 1

    return labels

# ============================================================
# MAIN
# ============================================================

def main():

    print("📥 Leyendo base completa:")
    print(f"  {INPUT_CSV}")

# --------------------------------------------------------
# Lectura robusta CSV (auto-detecta delimitador)
# --------------------------------------------------------
try:
    df = pd.read_csv(INPUT_CSV)
    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(INPUT_CSV, sep=";")
except Exception:
    df = pd.read_csv(INPUT_CSV, sep=";")

    print(f"✅ Registros: {len(df)}")
    print(f"🔎 Columnas: {list(df.columns)}")

    # --------------------------------------------------------
    # Reconstruir pickup_datetime si no existe
    # --------------------------------------------------------
    if "pickup_datetime" not in df.columns:
        if not {"pickup_date", "pickup_time"}.issubset(df.columns):
            raise ValueError("❌ No se encontró pickup_datetime ni (pickup_date, pickup_time)")

        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_date"] + " " + df["pickup_time"],
            errors="coerce"
        )
        print("🕒 pickup_datetime reconstruido")

    df = df.dropna(subset=["pickup_datetime"])
    print(f"🧹 Registros válidos: {len(df)}")

    # --------------------------------------------------------
    # Intervalo temporal (30 min)
    # --------------------------------------------------------
    df["interval30"] = (
        df["pickup_datetime"].astype("int64") // (30 * 60 * 1e9)
    ).astype(int)

    # --------------------------------------------------------
    # Coordenadas en radianes
    # --------------------------------------------------------
    coords = np.radians(
        df[["pickup_latitude", "pickup_longitude"]].values
    )

    times = df["interval30"].values

    # --------------------------------------------------------
    # Cargar configuración calibrada
    # --------------------------------------------------------
    if not CONFIG_JSON.exists():
        raise FileNotFoundError("❌ Falta stdbscan_best_config.json")

    with open(CONFIG_JSON) as f:
        cfg = json.load(f)

    eps_spatial = cfg["eps"]
    min_samples = cfg["min_samples"]

    # eps_temporal en intervalos de 30 min (1 = ±30 min)
    eps_temporal = 1

    print("⚙ Configuración ST-DBSCAN:")
    print(f"   eps espacial (m): {eps_spatial}")
    print(f"   eps temporal (interval30): {eps_temporal}")
    print(f"   min_samples: {min_samples}")

    # --------------------------------------------------------
    # Ejecutar ST-DBSCAN
    # --------------------------------------------------------
    print("🧠 Ejecutando ST-DBSCAN sobre semana completa...")

    labels = st_dbscan(
        coords,
        times,
        eps_spatial / 6371000,  # metros → radianes
        eps_temporal,
        min_samples
    )

    df["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_in_clusters = (labels != -1).sum()

    print(f"✅ Clusters detectados: {n_clusters}")
    print(f"✅ Puntos en clusters: {n_in_clusters} / {len(df)}")

    # --------------------------------------------------------
    # Guardar salida
    # --------------------------------------------------------
    df.to_csv(OUTPUT_CSV, index=False)

    print("💾 Guardado:")
    print(f"  {OUTPUT_CSV}")
    print("🎯 ST-DBSCAN FULL WEEK COMPLETADO")

# ============================================================

if __name__ == "__main__":
    main()

print("🎯 ST-DBSCAN FINALIZADO CORRECTAMENTE")
print(f"📦 Archivo generado: {OUTPUT_CSV}")

df_out = pd.read_csv(OUTPUT_CSV)

print(f"🧮 Registros totales: {len(df_out)}")
print(f"🧠 Clusters únicos: {df_out['cluster'].nunique()}")
print(f"🔵 Puntos en clusters: {(df_out['cluster'] != -1).sum()}")


