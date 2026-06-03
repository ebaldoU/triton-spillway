#!/usr/bin/env python3
"""
comparar_datasets.py — Comparativa de métricas clave entre N escenarios TileDB.

Métricas calculadas por dataset:
  - Celdas húmedas totales almacenadas
  - Extensión inundada máxima y hora en que se alcanza
  - Extensión en hora 60 (paso 10_00, referencia)
  - Peligro Russo adultos en hora 60 (H > 0.5 m ó Q_mod > 0.5 m²/s)
  - Volumen máximo de agua (Mm³)
  - Curva de extensión y peligro paso a paso

Uso:
  python comparar_datasets.py                              # auto-descubre todos
  python comparar_datasets.py --datasets datos1 datos3    # subconjunto
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tiledb
from config import resolve_dataset

from config import BASE_URI as _BASE_URI_IMPORT
BASE_URI = _BASE_URI_IMPORT
CELL_AREA = 100.0   # m²  (10×10 m)
STEP_H    = 6       # h por paso
H_ADULTO  = 0.50
HV_ADULTO = 0.50
REF_STEP  = "10_00"


def discover_datasets() -> list[str]:
    try:
        return sorted(d for d in os.listdir(BASE_URI)
                      if os.path.isdir(os.path.join(BASE_URI, d)))
    except FileNotFoundError:
        return []


def analizar(dataset: str) -> dict:
    uri = f"{BASE_URI}/{resolve_dataset(dataset)}"
    t0  = time.perf_counter()

    with tiledb.open(uri, mode="r") as A:
        meta       = dict(A.meta)
        time_steps = json.loads(meta["time_steps"])

    n_steps     = len(time_steps)
    horas       = np.array([(i + 1) * STEP_H for i in range(n_steps)])
    areas       = np.zeros(n_steps)
    peligros    = np.zeros(n_steps)
    vols        = np.zeros(n_steps)
    n_wet_total = 0

    with tiledb.open(uri, mode="r") as A:
        for t_idx, step in enumerate(time_steps):
            res   = A.query(attrs=["H", "QX", "QY"])[t_idx, :, :]
            H     = res["H"]
            Q_mod = np.sqrt(res["QX"]**2 + res["QY"]**2)
            n     = len(H)
            n_wet_total        += n
            areas[t_idx]        = n * CELL_AREA / 1e6
            vols[t_idx]         = float(H.sum()) * CELL_AREA / 1e6   # Mm³
            peligros[t_idx]     = int(((H > H_ADULTO) | (Q_mod > HV_ADULTO)).sum()) * CELL_AREA / 1e6
            del res, H, Q_mod

    idx_pico = int(np.argmax(areas))
    ref_idx  = time_steps.index(REF_STEP) if REF_STEP in time_steps else 9

    return {
        "dataset":         dataset,
        "n_steps":         n_steps,
        "n_wet_total":     n_wet_total,
        "areas_km2":       areas,
        "peligros_km2":    peligros,
        "vols_mm3":        vols,
        "horas":           horas,
        "area_max_km2":    float(areas.max()),
        "hora_pico":       int(horas[idx_pico]),
        "area_ref_km2":    float(areas[ref_idx]),
        "peligro_ref_km2": float(peligros[ref_idx]),
        "vol_max_mm3":     float(vols.max()),
        "elapsed_s":       time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparativa multi-escenario TileDB Triton")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets a comparar (por defecto: todos los de triton_results/)")
    args = parser.parse_args()

    datasets = args.datasets or discover_datasets()
    if not datasets:
        print("No se encontraron datasets en", BASE_URI)
        return

    print(f"\n{'═'*78}")
    print(f"  COMPARATIVA MULTI-ESCENARIO — TileDB Triton")
    print(f"{'═'*78}")
    print(f"  Datasets: {', '.join(datasets)}\n")

    resultados = {}
    for ds in datasets:
        print(f"  Analizando {ds}...", end=" ", flush=True)
        r = analizar(ds)
        resultados[ds] = r
        print(f"{r['elapsed_s']:.1f}s")

    rs = [resultados[ds] for ds in datasets]
    col_w = 13

    # ── Tabla resumen ──────────────────────────────────────────────────────
    header = f"  {'Métrica':<38}" + "".join(f"  {ds:>{col_w}}" for ds in datasets)
    print(f"\n{header}")
    print(f"  {'─'*38}" + ("  " + "─"*col_w) * len(datasets))

    def fmt_row(nombre, vals):
        return f"  {nombre:<38}" + "".join(f"  {v:>{col_w}}" for v in vals)

    metricas = [
        ("Pasos temporales",
         [str(r["n_steps"]) for r in rs]),
        ("Celdas húmedas totales (M)",
         [f"{r['n_wet_total']/1e6:.1f} M" for r in rs]),
        ("Extensión máxima (km²)",
         [f"{r['area_max_km2']:.0f}" for r in rs]),
        ("Hora del pico (h)",
         [f"{r['hora_pico']}" for r in rs]),
        ("Extensión en hora 60 (km²)",
         [f"{r['area_ref_km2']:.0f}" for r in rs]),
        ("Peligro adultos Russo h60 (km²)",
         [f"{r['peligro_ref_km2']:.0f}" for r in rs]),
        ("Volumen máximo (Mm³)",
         [f"{r['vol_max_mm3']:.1f}" for r in rs]),
    ]

    for nombre, vals in metricas:
        print(fmt_row(nombre, vals))

    # ── Tabla paso a paso ─────────────────────────────────────────────────
    print(f"\n  {'Hora':>5}" +
          "".join(f"  {ds+' área':>{col_w}}" for ds in datasets) +
          "".join(f"  {ds+' peligro':>{col_w}}" for ds in datasets))
    print(f"  {'─'*5}" + ("  " + "─"*col_w) * len(datasets) * 2)

    n_pasos = min(r["n_steps"] for r in rs)
    for i in range(n_pasos):
        h    = int(rs[0]["horas"][i])
        areas_str   = "".join(f"  {r['areas_km2'][i]:>{col_w}.0f}" for r in rs)
        peligro_str = "".join(f"  {r['peligros_km2'][i]:>{col_w}.0f}" for r in rs)
        print(f"  {h:>5}{areas_str}{peligro_str}")

    print(f"\n{'═'*78}\n")


if __name__ == "__main__":
    main()
