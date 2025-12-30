#!/usr/bin/env python3
# ============================================================
# weekday_temporal_permanova_pipeline.py
#
# Entrada (CSV):
#   Columnas esperadas: pickup_date, pickup_time, day
#   (opcionalmente otras columnas)
#
# Hace:
#   1) Construye histogramas intra-día por día calendario:
#        x_d[t] = #viajes en intervalo t (t-minutos)
#      y normaliza a distribución (suma=1) por día.
#   2) PERMANOVA global (7 días semana) sobre distancias (JS).
#   3) PERMANOVA pairwise + corrección Holm y FDR(BH).
#   4) Validación temporal:
#        - PERMANOVA estratificada por año (permuta dentro de año).
#        - Bootstrap por bloques de año (estabilidad del p-value).
#   5) JS weekday–weekend (distancia entre centroides temporales).
#
# Salidas (misma carpeta del input):
#   - daily_vectors_t{t}.csv
#   - permanova_global_t{t}.csv
#   - permanova_pairwise_t{t}.csv
#   - permanova_stratified_by_year_t{t}.csv
#   - bootstrap_by_year_t{t}.csv
# ============================================================

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


# ---------------------------
# Utilidades: distancias
# ---------------------------
def js_distance_matrix(P: np.ndarray) -> np.ndarray:
    """Matriz NxN de distancias Jensen–Shannon (scipy usa base 2 por defecto)."""
    n = P.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(jensenshannon(P[i], P[j]))
            D[i, j] = d
            D[j, i] = d
    return D


# ---------------------------
# PERMANOVA (one-way) desde matriz de distancias (Anderson 2001)
# Gower centering + permutación simple o estratificada por bloques
# ---------------------------
def permanova_oneway(
    D: np.ndarray,
    groups: np.ndarray,
    permutations: int = 999,
    seed: int = 123,
    strata: np.ndarray | None = None,
):
    """
    PERMANOVA one-way:
      - D: matriz NxN de distancias
      - groups: etiquetas de grupo (N)
      - strata: si se entrega, permuta 'groups' SOLO dentro de cada estrato
    Retorna: dict con pseudo-F, R2, p-value, SS, df.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("D debe ser NxN y consistente con groups.")

    # Gower centering: A = -0.5 * D^2 ; G = H A H
    A = -0.5 * (D ** 2)
    H = np.eye(n) - np.ones((n, n)) / n
    G = H @ A @ H

    def pseudo_F_for_labels(lbls: np.ndarray):
        ss_total = float(np.trace(G))

        ss_within = 0.0
        unique = pd.unique(lbls)
        k = len(unique)

        for g in unique:
            idx = np.where(lbls == g)[0]
            ng = len(idx)
            if ng <= 1:
                continue
            Gg = G[np.ix_(idx, idx)]
            Hg = np.eye(ng) - np.ones((ng, ng)) / ng
            ss_within += float(np.trace(Hg @ Gg @ Hg))

        ss_among = ss_total - ss_within
        df_among = k - 1
        df_within = n - k

        ms_among = ss_among / df_among if df_among > 0 else np.nan
        ms_within = ss_within / df_within if df_within > 0 else np.nan
        F = ms_among / ms_within if (np.isfinite(ms_within) and ms_within > 0) else np.nan

        R2 = ss_among / ss_total if ss_total > 0 else np.nan
        return F, R2, ss_total, ss_among, ss_within, df_among, df_within, k

    def permute_labels(lbls: np.ndarray) -> np.ndarray:
        if strata is None:
            return rng.permutation(lbls)
        out = lbls.copy()
        for s in pd.unique(strata):
            idx = np.where(strata == s)[0]
            out[idx] = rng.permutation(out[idx])
        return out

    F_obs, R2_obs, ss_total, ss_among, ss_within, df_a, df_w, k = pseudo_F_for_labels(groups)

    count_ge = 0
    for _ in range(permutations):
        g_perm = permute_labels(groups)
        F_perm, *_ = pseudo_F_for_labels(g_perm)
        if np.isfinite(F_perm) and np.isfinite(F_obs) and (F_perm >= F_obs):
            count_ge += 1

    p_value = (count_ge + 1) / (permutations + 1)

    return {
        "pseudo_F": F_obs,
        "R2": R2_obs,
        "p_value": p_value,
        "permutations": permutations,
        "ss_total": ss_total,
        "ss_among": ss_among,
        "ss_within": ss_within,
        "df_among": df_a,
        "df_within": df_w,
        "n": n,
        "k": k,
        "stratified": strata is not None,
    }


# ---------------------------
# Correcciones múltiples
# ---------------------------
def p_adjust_holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        val = max(val, prev)  # monotonicidad
        adj[idx] = min(val, 1.0)
        prev = adj[idx]
    return adj


def p_adjust_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        val = p[idx] * m / rank
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


# ---------------------------
# Construcción de vectores diarios
# ---------------------------
def build_daily_vectors(df: pd.DataFrame, t_minutes: int) -> pd.DataFrame:
    """
    Retorna DataFrame con una fila por fecha (día calendario),
    columnas interval_0..interval_{T-1}, más metadatos: date, day, year, total_trips.
    """
    df = df.copy()

    dt = pd.to_datetime(
        df["pickup_date"].astype(str) + " " + df["pickup_time"].astype(str),
        errors="coerce",
    )
    df["pickup_datetime"] = dt
    df = df.dropna(subset=["pickup_datetime"])

    df["date"] = df["pickup_datetime"].dt.date.astype(str)
    df["year"] = df["pickup_datetime"].dt.year.astype(int)

    minutes = df["pickup_datetime"].dt.hour * 60 + df["pickup_datetime"].dt.minute
    df["interval"] = (minutes // t_minutes).astype(int)

    T = (24 * 60) // t_minutes

    g = (
        df.groupby(["date", "day", "year", "interval"])
        .size()
        .reset_index(name="count")
    )

    piv = g.pivot_table(
        index=["date", "day", "year"],
        columns="interval",
        values="count",
        fill_value=0,
    )

    piv = piv.reindex(columns=list(range(T)), fill_value=0)
    piv.columns = [f"interval_{c}" for c in piv.columns]

    piv = piv.reset_index()
    piv["total_trips"] = piv.filter(like="interval_").sum(axis=1).astype(int)
    return piv


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return mat / s


# ---------------------------
# Bootstrap por bloques de año
# ---------------------------
def bootstrap_by_year(
    D_full: np.ndarray,
    groups: np.ndarray,
    years: np.ndarray,
    B: int,
    permutations: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    unique_years = pd.unique(years)
    idx_by_year = {y: np.where(years == y)[0] for y in unique_years}

    results = []
    for b in range(B):
        sampled_years = rng.choice(unique_years, size=len(unique_years), replace=True)

        boot_idx = []
        for y in sampled_years:
            idx = idx_by_year[y]
            boot_idx.extend(rng.choice(idx, size=len(idx), replace=True).tolist())

        boot_idx = np.array(boot_idx, dtype=int)

        D = D_full[np.ix_(boot_idx, boot_idx)]
        g = groups[boot_idx]
        y = years[boot_idx]

        res = permanova_oneway(
            D, g,
            permutations=permutations,
            seed=int(rng.integers(1, 10**9)),
            strata=y,
        )
        results.append({"bootstrap": b + 1, **res})

    return pd.DataFrame(results)


def infer_default_input(script_path: Path) -> Path | None:
    """Busca un CSV plausible en la carpeta del script si no pasas --input."""
    candidates = [
        "Taxi_manhattan_correct_datetime_numbers_with_day.csv",
        "taxi_manhattan_correct_datetime_numbers_with_day.csv",
        "data.csv",
    ]
    for name in candidates:
        p = script_path.parent / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="ruta al CSV de entrada")
    ap.add_argument("--t", type=int, default=30, help="minutos por intervalo (ej: 30)")
    ap.add_argument("--permutations", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--bootstrap_B", type=int, default=200)
    args = ap.parse_args()

    if args.input is None:
        default_path = infer_default_input(Path(__file__).resolve())
        if default_path is None:
            raise SystemExit(
                "ERROR: no pasaste --input y no encontré un CSV por defecto en la carpeta del script.\n"
                "Ejemplo:\n"
                "  python weekday_temporal_permanova_pipeline.py --input /ruta/a/tu.csv --t 30\n"
            )
        input_path = default_path
        print(f"Usando input por defecto: {input_path}")
    else:
        input_path = Path(args.input)

    if not input_path.exists():
        raise SystemExit(f"ERROR: no existe el archivo: {input_path}")

    out_dir = input_path.parent

    df = pd.read_csv(input_path, sep=";", decimal=",")

    required = {"pickup_date", "pickup_time", "day"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: faltan columnas: {missing}. Columnas disponibles: {df.columns.tolist()}")

    # 1) Vectores diarios
    daily = build_daily_vectors(df, t_minutes=args.t)

    X = daily.filter(like="interval_").to_numpy(dtype=float)
    P = normalize_rows(X)

    # 2) Distancias + PERMANOVA global
    D = js_distance_matrix(P)

    groups = daily["day"].astype(str).to_numpy()
    years = daily["year"].to_numpy()

    res_global = permanova_oneway(
        D, groups,
        permutations=args.permutations,
        seed=args.seed,
        strata=None,
    )

    res_strat = permanova_oneway(
        D, groups,
        permutations=args.permutations,
        seed=args.seed,
        strata=years,
    )

    # 3) Pairwise + correcciones
    days = sorted(pd.unique(groups))
    pair_rows = []
    for a, b in combinations(days, 2):
        idx = np.where((groups == a) | (groups == b))[0]
        D_ab = D[np.ix_(idx, idx)]
        g_ab = groups[idx]
        y_ab = years[idx]

        res_ab = permanova_oneway(
            D_ab, g_ab,
            permutations=args.permutations,
            seed=args.seed,
            strata=y_ab,  # estratificado por año (más conservador)
        )

        pair_rows.append({
            "day_A": a,
            "day_B": b,
            "pseudo_F": res_ab["pseudo_F"],
            "R2": res_ab["R2"],
            "p_value": res_ab["p_value"],
            "n": res_ab["n"],
            "stratified_by_year": True,
        })

    pair_df = pd.DataFrame(pair_rows)
    if len(pair_df) > 0:
        pair_df["p_holm"] = p_adjust_holm(pair_df["p_value"].to_numpy())
        pair_df["p_fdr_bh"] = p_adjust_bh(pair_df["p_value"].to_numpy())

    # 4) Bootstrap por bloques de año
    boot_df = bootstrap_by_year(
        D_full=D,
        groups=groups,
        years=years,
        B=args.bootstrap_B,
        permutations=max(199, args.permutations // 5),
        seed=args.seed + 999,
    )

    # 5) JS weekday–weekend (centroides temporales)
        # --------------------------------------------------
    # Mapear day a numérico (robusto a strings)
    # --------------------------------------------------
    day_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    day_numeric = (
        daily["day"]
        .astype(str)
        .str.lower()
        .map(day_map)
        .to_numpy()
    )

    if np.any(pd.isna(day_numeric)):
        raise ValueError("Hay valores de 'day' no reconocidos tras el mapeo.")

    weekday_mask = np.isin(day_numeric, [0, 1, 2, 3, 4])
    weekend_mask = np.isin(day_numeric, [5, 6])

    weekday_profile = P[weekday_mask].mean(axis=0)
    weekend_profile = P[weekend_mask].mean(axis=0)

    weekday_profile /= weekday_profile.sum()
    weekend_profile /= weekend_profile.sum()

    js_weekday_weekend = float(jensenshannon(weekday_profile, weekend_profile))

    print("--------------------------------------------------")
    print(f"JS weekday–weekend (temporal centroids) = {js_weekday_weekend:.3f}")
    print("--------------------------------------------------")

    # Añadir JS a la tabla global (para trazabilidad)
    res_global["js_weekday_weekend_centroids"] = js_weekday_weekend
    res_strat["js_weekday_weekend_centroids"] = js_weekday_weekend

    # ------------------
    # Guardar salidas
    # ------------------
    daily_out = out_dir / f"daily_vectors_t{args.t}.csv"
    global_out = out_dir / f"permanova_global_t{args.t}.csv"
    pair_out = out_dir / f"permanova_pairwise_t{args.t}.csv"
    strat_out = out_dir / f"permanova_stratified_by_year_t{args.t}.csv"
    boot_out = out_dir / f"bootstrap_by_year_t{args.t}.csv"

    daily.to_csv(daily_out, index=False, sep=";", decimal=",")
    pd.DataFrame([res_global]).to_csv(global_out, index=False, sep=";", decimal=",")
    pair_df.to_csv(pair_out, index=False, sep=";", decimal=",")
    pd.DataFrame([res_strat]).to_csv(strat_out, index=False, sep=";", decimal=",")
    boot_df.to_csv(boot_out, index=False, sep=";", decimal=",")

    print("\n=== PERMANOVA GLOBAL (no estratificada) ===")
    print(res_global)
    print("\n=== PERMANOVA GLOBAL (estratificada por año) ===")
    print(res_strat)

    if len(pair_df) > 0:
        print("\n=== TOP diferencias (pairwise, ordenado por p_fdr_bh) ===")
        print(pair_df.sort_values("p_fdr_bh").head(10))

    print("\nArchivos guardados en:")
    print(daily_out)
    print(global_out)
    print(pair_out)
    print(strat_out)
    print(boot_out)


if __name__ == "__main__":
    main()
