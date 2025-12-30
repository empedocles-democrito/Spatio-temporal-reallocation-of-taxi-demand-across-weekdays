#!/usr/bin/env python3
# ============================================================
# CAPA B DEFINITIVA
# Comparación espacial entre días de la semana usando
# hotspots canónicos (CAPA A)
#
# Outputs:
#  - spatial_daily_vectors.csv
#  - spatial_permanova_global.csv
#  - spatial_permanova_pairwise.csv
#  - spatial_simper_weekday_weekend.csv
#  - mapa_hotspots_simper.html
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import jensenshannon
import folium

# ============================================================
# CONFIG
# ============================================================
DATA = Path(
    
    "Taxi_manhattan_correct_datetime_numbers_with_day.csv"
)

HOTSPOTS = Path(
        "hotspots_global_resumen_t30.csv"
)

OUTDIR = DATA.parent
MAX_DIST = 500.0      # metros
PERMUTATIONS = 999
SEED = 123

# ============================================================
# LECTURA ROBUSTA
# ============================================================
try:
    df = pd.read_csv(DATA, sep=";", decimal=",")
    if "pickup_latitude" not in df.columns:
        raise ValueError
except Exception:
    df = pd.read_csv(DATA, sep=",", decimal=".")

for c in ["pickup_latitude", "pickup_longitude"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["pickup_latitude", "pickup_longitude"]).copy()
df["date"] = df["pickup_date"].astype(str)

hs = pd.read_csv(HOTSPOTS)

# ============================================================
# PROYECCIÓN PLANA (m)
# ============================================================
lat0, lon0 = hs.mean_lat.mean(), hs.mean_lon.mean()
m_lat = 111_320.0
m_lon = 111_320.0 * np.cos(np.radians(lat0))

def xy(lat, lon):
    return np.c_[(lon - lon0) * m_lon, (lat - lat0) * m_lat]

hs[["x", "y"]] = xy(hs.mean_lat, hs.mean_lon)
df[["x", "y"]] = xy(df.pickup_latitude, df.pickup_longitude)

# ============================================================
# ASIGNACIÓN A HOTSPOT CANÓNICO
# ============================================================
HXY = hs[["hotspot_id", "x", "y"]].set_index("hotspot_id")

def assign_hotspot(row):
    d = np.sqrt((HXY.x - row.x)**2 + (HXY.y - row.y)**2)
    j = d.idxmin()
    return j if d.min() <= MAX_DIST else -1

df["hotspot_id"] = df.apply(assign_hotspot, axis=1)
df = df[df.hotspot_id >= 0]

# ============================================================
# VECTORES ESPACIALES DIARIOS (NORMALIZADOS)
# ============================================================
daily = (
    df.groupby(["date", "day", "hotspot_id"])
    .size()
    .unstack(fill_value=0)
)

daily_norm = daily.div(daily.sum(axis=1), axis=0)
daily_norm.reset_index(inplace=True)

daily_norm.to_csv(
    OUTDIR / "spatial_daily_vectors.csv",
    index=False, sep=";", decimal=","
)

# ============================================================
# DISTANCIA JENSEN–SHANNON
# ============================================================
X = daily_norm.drop(columns=["date", "day"]).to_numpy()
labels = daily_norm["day"].to_numpy()

D = np.zeros((len(X), len(X)))
for i in range(len(X)):
    for j in range(i + 1, len(X)):
        D[i, j] = D[j, i] = jensenshannon(X[i], X[j])

# ============================================================
# PERMANOVA (misma lógica que CAPA temporal)
# ============================================================
def permanova(D, groups, permutations=999, seed=123):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)

    def ss_between(lbls):
        return sum(
            np.mean(D[np.ix_(lbls == g, lbls != g)])
            for g in unique
        )

    F_obs = ss_between(groups)

    count = 0
    for _ in range(permutations):
        perm = rng.permutation(groups)
        if ss_between(perm) >= F_obs:
            count += 1

    p = (count + 1) / (permutations + 1)
    return F_obs, p

F_global, p_global = permanova(D, labels, PERMUTATIONS, SEED)

pd.DataFrame([{
    "pseudo_F": F_global,
    "p_value": p_global,
    "n_days": len(daily_norm)
}]).to_csv(
    OUTDIR / "spatial_permanova_global.csv",
    index=False, sep=";", decimal=","
)

# ============================================================
# PERMANOVA PAIRWISE
# ============================================================
rows = []
for a, b in combinations(np.unique(labels), 2):
    idx = (labels == a) | (labels == b)
    F, p = permanova(D[np.ix_(idx, idx)], labels[idx], PERMUTATIONS, SEED)
    rows.append({"day_A": a, "day_B": b, "pseudo_F": F, "p_value": p})

pairwise = pd.DataFrame(rows)
pairwise.to_csv(
    OUTDIR / "spatial_permanova_pairwise.csv",
    index=False, sep=";", decimal=","
)

# ============================================================
# SIMPER ESPACIAL (WEEKDAY vs WEEKEND)
# ============================================================
weekday = daily_norm[daily_norm.day.isin(
    ["monday","tuesday","wednesday","thursday","friday"]
)].drop(columns=["date","day"]).mean()

weekend = daily_norm[daily_norm.day.isin(
    ["saturday","sunday"]
)].drop(columns=["date","day"]).mean()

diff = np.abs(weekday - weekend)
simper = pd.DataFrame({
    "hotspot_id": diff.index,
    "contribution": diff.values
}).sort_values("contribution", ascending=False)

simper["cum_contribution"] = simper.contribution.cumsum() / simper.contribution.sum()

simper.to_csv(
    OUTDIR / "spatial_simper_weekday_weekend.csv",
    index=False, sep=";", decimal=","
)

# ============================================================
# MAPA: hotspots coloreados por contribución SIMPER
# ============================================================
m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles="cartodbpositron")

hs_map = hs.merge(simper, on="hotspot_id", how="left").fillna(0)

for _, r in hs_map.iterrows():
    folium.CircleMarker(
        location=[r.mean_lat, r.mean_lon],
        radius=6 + 20 * r.contribution,
        color="red",
        fill=True,
        fill_opacity=0.7,
        popup=f"Hotspot {r.hotspot_id}<br>SIMPER={r.contribution:.3f}"
    ).add_to(m)

m.save(OUTDIR / "mapa_hotspots_simper.html")

print("✅ CAPA B completa")
print("PERMANOVA global:", F_global, "p =", p_global)
print("Archivos guardados en:", OUTDIR)
