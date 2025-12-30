#!/usr/bin/env python3
# ============================================================
# figure_and_simper_weekdays.py
#
# Inputs:
#   daily_vectors_t30.csv   (salida del pipeline PERMANOVA)
#
# Outputs:
#   - centroides_weekday_t30.png
#   - simper_weekday_t30.csv
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import jensenshannon

# ------------------------------------------------------------
# Parámetros
# ------------------------------------------------------------
T_MIN = 30
INPUT = Path(
        "daily_vectors_t30.csv"
)
OUTDIR = INPUT.parent

# ------------------------------------------------------------
# Leer datos (formato Numbers-friendly)
# ------------------------------------------------------------
df = pd.read_csv(INPUT, sep=";", decimal=",")

interval_cols = [c for c in df.columns if c.startswith("interval_")]

# ------------------------------------------------------------
# Normalizar (forma temporal)
# ------------------------------------------------------------
X = df[interval_cols].to_numpy(float)
X = X / X.sum(axis=1, keepdims=True)

df_norm = pd.concat([df[["day"]], pd.DataFrame(X, columns=interval_cols)], axis=1)

# ------------------------------------------------------------
# 1) CENTROIDES TEMPORALES POR DÍA
# ------------------------------------------------------------
centroids = (
    df_norm
    .groupby("day")[interval_cols]
    .mean()
)

# Reordenar días
order = ["monday", "tuesday", "wednesday", "thursday",
         "friday", "saturday", "sunday"]
centroids = centroids.loc[order]

# ------------------------------------------------------------
# FIGURA: perfiles intra-día
# ------------------------------------------------------------
hours = np.arange(len(interval_cols)) * T_MIN / 60

plt.figure(figsize=(10, 6))
for day in centroids.index:
    plt.plot(hours, centroids.loc[day], label=day.capitalize())

plt.xlabel("Hour of day")
plt.ylabel("Share of daily trips")
plt.title("Intra-day demand profiles by weekday (Manhattan )")
plt.legend(ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()

fig_path = OUTDIR / "centroides_weekday_t30.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"✔ Figura guardada en: {fig_path}")

# ------------------------------------------------------------
# 2) SIMPER-LIKE (contribución por intervalo)
# ------------------------------------------------------------
# Comparación: weekday vs weekend
weekday = centroids.loc[["monday", "tuesday", "wednesday", "thursday", "friday"]].mean()
weekend = centroids.loc[["saturday", "sunday"]].mean()

# Contribución JS por intervalo (aprox. local)
eps = 1e-12
p = weekday.to_numpy() + eps
q = weekend.to_numpy() + eps
m = 0.5 * (p + q)

js_local = 0.5 * (p * np.log(p / m) + q * np.log(q / m))
js_local = js_local / js_local.sum()  # normalizar contribuciones

simper = pd.DataFrame({
    "interval": np.arange(len(p)),
    "hour_start": hours,
    "contribution": js_local
}).sort_values("contribution", ascending=False)

simper["cum_contribution"] = simper["contribution"].cumsum()

simper_path = OUTDIR / "simper_weekday_t30.csv"
simper.to_csv(simper_path, index=False, sep=";", decimal=",")

print(f"✔ SIMPER-like guardado en: {simper_path}")

print("\nTop intervalos explicativos:")
print(simper.head(10))
