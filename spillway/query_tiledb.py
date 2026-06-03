#!/usr/bin/env python3
# ============================================================
# query_tiledb.py
# Consultas analíticas sobre el array TileDB de Triton.
#
# Procesa paso a paso (no carga todo el array en RAM):
# ~360 MB pico por paso vs ~28 GB si se leyera todo a la vez.
#
#   1. Estadísticas globales por paso temporal
#   2. Área inundada (H > FLOOD_THRESHOLD)
#   3. Máximos globales: dónde está la mayor profundidad
#   4. Serie temporal: evolución de H en el dominio
#
# Uso:
#   python query_tiledb.py --dataset datos1
#   python query_tiledb.py --dataset datos2
# ============================================================
from __future__ import annotations

import argparse
import heapq
import json
import time

import numpy as np
import tiledb
from config import BASE_URI, resolve_dataset

FLOOD_THRESHOLD = 0.011   # m
TOP_N           = 10


def row_col_to_xy(row, col, transform):
    x = transform[2] + col * transform[0]
    y = transform[5] + row * transform[4]
    return x, y


def sep(title=""):
    w = 65
    if title:
        print(f"\n{'─'*3} {title} {'─'*(w - len(title) - 5)}")
    else:
        print("─" * w)


def query_all(uri: str) -> None:
    t0 = time.perf_counter()

    with tiledb.open(uri, mode="r") as A:
        meta       = dict(A.meta)
        transform  = json.loads(meta["transform"])
        time_steps = json.loads(meta["time_steps"])

    cell_area = abs(transform[0] * transform[4])
    n_total   = int(meta["width"]) * int(meta["height"])

    print(f"\n{'═'*65}")
    print(f"  CONSULTAS ANALÍTICAS — TileDB Triton")
    print(f"{'═'*65}")
    print(f"  Fuente:     {uri}")
    print(f"  CRS:        {meta.get('crs', '?')}")
    print(f"  Resolución: {meta['width']} x {meta['height']} px  ({cell_area:.0f} m²/celda)")
    print(f"  Pasos:      {time_steps}")

    # Acumuladores (datos agregados, no arrays completos)
    step_stats  = []   # sección 1
    flood_data  = []   # sección 2
    time_series = []   # sección 4

    # Sección 3: máximos globales y top-N (min-heap de tamaño TOP_N)
    g_max_h  = {"val": -np.inf, "row": 0, "col": 0, "t": 0, "mh": 0.0}
    g_max_mh = {"val": -np.inf, "row": 0, "col": 0, "t": 0}
    top_heap  = []   # (h_val, t_idx, row, col, mh_val)

    with tiledb.open(uri, mode="r") as A:
        for t_idx, step in enumerate(time_steps):
            res  = A.query(attrs=["H", "QX", "QY", "MH"])[t_idx, :, :]
            H    = res["H"]
            MH   = res["MH"]
            rows = res["row"]
            cols = res["col"]
            n    = len(H)

            # ── Sección 1 ──────────────────────────────────────────
            if n > 0:
                step_stats.append({
                    "step": step, "n": n,
                    "h_min": float(H.min()), "h_max": float(H.max()),
                    "h_mean": float(H.mean()), "h_std": float(H.std()),
                })
            else:
                step_stats.append({"step": step, "n": 0})

            # ── Sección 2 ──────────────────────────────────────────
            mask_flood = H > FLOOD_THRESHOLD
            n_flood    = int(mask_flood.sum())
            mh_mean    = float(MH[mask_flood].mean()) if n_flood > 0 else 0.0
            flood_data.append({
                "step": step, "n_flood": n_flood,
                "area_km2": n_flood * cell_area / 1e6,
                "pct": n_flood / n_total * 100,
                "mh_mean": mh_mean,
            })

            # ── Sección 3 ──────────────────────────────────────────
            if n > 0:
                idx_h = int(np.argmax(H))
                if H[idx_h] > g_max_h["val"]:
                    g_max_h = {"val": float(H[idx_h]), "row": int(rows[idx_h]),
                               "col": int(cols[idx_h]), "t": t_idx, "mh": float(MH[idx_h])}

                idx_mh = int(np.argmax(MH))
                if MH[idx_mh] > g_max_mh["val"]:
                    g_max_mh = {"val": float(MH[idx_mh]), "row": int(rows[idx_mh]),
                                "col": int(cols[idx_mh]), "t": t_idx}

                # Top-N: solo los TOP_N mejores candidatos del paso entran al heap
                top_local = np.argsort(H)[-TOP_N:]
                for k in top_local:
                    entry = (float(H[k]), t_idx, int(rows[k]), int(cols[k]), float(MH[k]))
                    if len(top_heap) < TOP_N:
                        heapq.heappush(top_heap, entry)
                    elif float(H[k]) > top_heap[0][0]:
                        heapq.heapreplace(top_heap, entry)

            # ── Sección 4 ──────────────────────────────────────────
            if n > 0:
                vol = float((H * cell_area).sum())
                time_series.append({
                    "step": step, "n": n,
                    "h_max": float(H.max()), "h_mean": float(H.mean()),
                    "vol_mm3": vol / 1e6,
                })

            del res, H, MH, rows, cols   # liberar RAM inmediatamente

    t_query = time.perf_counter() - t0
    print(f"  Tiempo de carga: {t_query:.2f}s")

    # ── Imprimir sección 1 ────────────────────────────────────
    sep("1. ESTADÍSTICAS GLOBALES POR PASO TEMPORAL")
    print(f"  {'Paso':<8}  {'Celdas mojadas':>15}  {'H min':>7}  {'H max':>7}  {'H media':>8}  {'H std':>7}")
    print(f"  {'─'*8}  {'─'*15}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*7}")
    for s in step_stats:
        if s["n"] == 0:
            print(f"  {s['step']:<8}  {'— sin datos —':>15}")
        else:
            print(f"  {s['step']:<8}  {s['n']:>15,}  "
                  f"{s['h_min']:>7.3f}  {s['h_max']:>7.3f}  "
                  f"{s['h_mean']:>8.4f}  {s['h_std']:>7.4f}")

    # ── Imprimir sección 2 ────────────────────────────────────
    sep(f"2. ÁREA INUNDADA  (H > {FLOOD_THRESHOLD} m)")
    print(f"  {'Paso':<8}  {'Celdas':>12}  {'Área (km²)':>11}  {'% dominio':>10}  {'MH media':>9}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*11}  {'─'*10}  {'─'*9}")
    for f in flood_data:
        print(f"  {f['step']:<8}  {f['n_flood']:>12,}  {f['area_km2']:>11.2f}  "
              f"{f['pct']:>9.2f}%  {f['mh_mean']:>9.4f}")

    # ── Imprimir sección 3 ────────────────────────────────────
    sep("3. MÁXIMOS GLOBALES")
    xh,  yh  = row_col_to_xy(g_max_h["row"],  g_max_h["col"],  transform)
    xmh, ymh = row_col_to_xy(g_max_mh["row"], g_max_mh["col"], transform)
    print(f"  Máxima profundidad instantánea (H):")
    print(f"    H = {g_max_h['val']:.4f} m  |  paso {time_steps[g_max_h['t']]}")
    print(f"    Fila {g_max_h['row']}, Col {g_max_h['col']}  →  X={xh:.1f}, Y={yh:.1f}")
    print()
    print(f"  Máxima profundidad histórica (MH):")
    print(f"    MH = {g_max_mh['val']:.4f} m  |  paso {time_steps[g_max_mh['t']]}")
    print(f"    Fila {g_max_mh['row']}, Col {g_max_mh['col']}  →  X={xmh:.1f}, Y={ymh:.1f}")
    print()
    top_sorted = sorted(top_heap, key=lambda e: e[0], reverse=True)
    print(f"  Top {TOP_N} celdas con mayor H:")
    print(f"  {'#':<4}  {'H (m)':>7}  {'MH (m)':>7}  {'paso':<8}  {'X':>10}  {'Y':>10}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*10}  {'─'*10}")
    for rank, (h_val, t_idx, row, col, mh_val) in enumerate(top_sorted, 1):
        xk, yk = row_col_to_xy(row, col, transform)
        print(f"  {rank:<4}  {h_val:>7.3f}  {mh_val:>7.3f}  "
              f"{time_steps[t_idx]:<8}  {xk:>10.1f}  {yk:>10.1f}")

    # ── Imprimir sección 4 ────────────────────────────────────
    sep("4. SERIE TEMPORAL — EVOLUCIÓN DE H EN EL DOMINIO")
    print(f"  {'Paso':<8}  {'Células activas':>16}  {'H max':>7}  {'H media':>8}  {'Vol total (Mm³)':>16}")
    print(f"  {'─'*8}  {'─'*16}  {'─'*7}  {'─'*8}  {'─'*16}")
    for ts in time_series:
        print(f"  {ts['step']:<8}  {ts['n']:>16,}  {ts['h_max']:>7.3f}  "
              f"{ts['h_mean']:>8.4f}  {ts['vol_mm3']:>16.4f}")

    sep()
    print(f"  Tiempo total: {time.perf_counter() - t0:.2f}s")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Consultas analíticas TileDB/Triton")
    parser.add_argument("--dataset", required=True,
                        help="Dataset a consultar")
    args = parser.parse_args()
    query_all(f"{BASE_URI}/{resolve_dataset(args.dataset)}")


if __name__ == "__main__":
    main()
