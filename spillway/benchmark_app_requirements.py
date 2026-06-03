#!/usr/bin/env python3
"""
benchmark_app_requirements.py
Mide los requerimientos de la app para despliegue:
  - Tiempo frío y caliente por query
  - RAM pico (tracemalloc) y RSS proceso (psutil)
  - CPU user+sys (psutil)
  - Disco leído (psutil io_counters)
  - Almacenamiento: TileDB, venv, scripts

Uso:
  python benchmark_app_requirements.py --dataset datos1
  python benchmark_app_requirements.py --dataset datos1 --reps 3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil
import tiledb
from config import resolve_dataset

from config import BASE_URI as _BASE_URI_IMPORT
BASE_URI = _BASE_URI_IMPORT
SCRIPTS  = Path("/home/ebald/spillway")
VENV     = Path("/home/ebald/venvs/tiledb_env")
H_WET    = 0.01
H_ADULTO = 0.50
HV_ADULTO= 0.50
H_NINO   = 0.25
HV_NINO  = 0.15
H_VEHICULO   = 0.30
H_EMERGENCIA = 0.60
STEP_H   = 6


def dir_mb(path: str | Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024**2


def medir(fn, reps: int) -> dict:
    proc = psutil.Process()
    gc.collect()

    # --- caché fría ---
    io_before = proc.io_counters()
    cpu_before = proc.cpu_times()
    rss_before = proc.memory_info().rss
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    t_frio = time.perf_counter() - t0
    _, pico_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = proc.memory_info().rss
    cpu_after = proc.cpu_times()
    io_after  = proc.io_counters()

    ram_pico_mb = pico_tm / 1024**2
    rss_delta_mb = (rss_after - rss_before) / 1024**2
    cpu_s = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
    disco_mb = (io_after.read_bytes - io_before.read_bytes) / 1024**2

    # --- caché caliente ---
    tiempos_cal = []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        tiempos_cal.append(time.perf_counter() - t0)

    return {
        "t_frio":   t_frio,
        "t_cal":    float(np.mean(tiempos_cal)),
        "ram_pico": ram_pico_mb,
        "rss_delta": rss_delta_mb,
        "cpu_s":    cpu_s,
        "disco_mb": disco_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reps", type=int, default=2, help="Repeticiones caché caliente")
    args = parser.parse_args()

    uri = f"{BASE_URI}/{resolve_dataset(args.dataset)}"
    reps = args.reps

    with tiledb.open(uri, mode="r") as A:
        meta_raw  = dict(A.meta)
        time_steps = json.loads(meta_raw["time_steps"])
        nrows = int(meta_raw["height"])
        ncols = int(meta_raw["width"])
        tr    = json.loads(meta_raw["transform"])

    n_steps = len(time_steps)
    cs = abs(float(tr[0]))
    xll = float(tr[2])
    yll = float(tr[5]) + float(tr[4]) * nrows
    cell2 = cs**2 / 1e6

    # BBox central de referencia (50×50 km)
    xc, yc = xll + ncols * cs / 2, yll + nrows * cs / 2
    bx0, by0, bx1, by1 = xc - 25000, yc - 25000, xc + 25000, yc + 25000
    c0 = max(0, int((bx0 - xll) / cs))
    c1 = min(ncols - 1, int((bx1 - xll) / cs))
    r0 = max(0, int((yll + nrows * cs - by1) / cs))
    r1 = min(nrows - 1, int((yll + nrows * cs - by0) / cs))
    ref_step = min(9, n_steps - 1)

    # ── Definición de queries ──────────────────────────────────────────────────

    def q1_calado():
        row = int((yll + nrows * cs - yc) / cs)
        col = int((xc - xll) / cs)
        with tiledb.open(uri, mode="r") as A:
            A.query(attrs=["H"])[ref_step, row, col]

    def q2_serie():
        row = int((yll + nrows * cs - yc) / cs)
        col = int((xc - xll) / cs)
        with tiledb.open(uri, mode="r") as A:
            for t in range(n_steps):
                A.query(attrs=["H"])[t, row, col]

    def q4_umbral():
        with tiledb.open(uri, mode="r") as A:
            A.query(attrs=["H"])[ref_step, r0:r1, c0:c1]

    def q5_evolucion():
        with tiledb.open(uri, mode="r") as A:
            for t in range(n_steps):
                res = A.query(attrs=["H"])[t, r0:r1, c0:c1]
                del res

    def q7_llegada():
        gr = min(1800, r1 - r0)
        gc_ = min(1800, c1 - c0)
        scale = gr / (r1 - r0) if r1 > r0 else 1
        g = np.full((gr, gc_), n_steps, np.int16)
        with tiledb.open(uri, mode="r") as A:
            for t in range(n_steps):
                res = A.query(attrs=["H"])[t, r0:r1, c0:c1]
                H = res["H"]
                rows = ((res["row"] - r0) * scale).astype(np.int32).clip(0, gr - 1)
                cols = ((res["col"] - c0) * scale).astype(np.int32).clip(0, gc_ - 1)
                mask = H >= H_WET
                if mask.any():
                    np.minimum.at(g, (rows[mask], cols[mask]), t)
                del res
        return g

    def q10a_peligro():
        with tiledb.open(uri, mode="r") as A:
            res = A.query(attrs=["H", "QX", "QY"])[ref_step, r0:r1, c0:c1]
        Q = np.sqrt(res["QX"]**2 + res["QY"]**2)
        return int(((res["H"] > H_ADULTO) | (Q > HV_ADULTO)).sum())

    def q12_area():
        with tiledb.open(uri, mode="r") as A:
            for t in range(n_steps):
                res = A.query(attrs=["H"])[t, :, :]
                del res

    def q13_volumen():
        with tiledb.open(uri, mode="r") as A:
            for t in range(n_steps):
                res = A.query(attrs=["H"])[t, :, :]
                _ = float(res["H"].sum()) * cs**2
                del res

    def q16_evacuacion():
        g_lleg = np.full((1800, 1800), n_steps, np.int16)
        g_crit = np.full((1800, 1800), n_steps, np.int16)
        gr, gc_ = 1800, 1800
        scale = gr / max(r1 - r0, 1)
        with tiledb.open(uri, mode="r") as A:
            for t in range(n_steps):
                res = A.query(attrs=["H", "QX", "QY"])[t, r0:r1, c0:c1]
                H = res["H"]
                Q = np.sqrt(res["QX"]**2 + res["QY"]**2)
                rows = ((res["row"] - r0) * scale).astype(np.int32).clip(0, gr - 1)
                cols = ((res["col"] - c0) * scale).astype(np.int32).clip(0, gc_ - 1)
                m_wet = H >= H_WET
                if m_wet.any():
                    np.minimum.at(g_lleg, (rows[m_wet], cols[m_wet]), t)
                m_cri = (H > H_ADULTO) | (Q > HV_ADULTO)
                if m_cri.any():
                    np.minimum.at(g_crit, (rows[m_cri], cols[m_cri]), t)
                del res

    QUERIES = [
        ("Q1  Calado en punto                    ", q1_calado),
        ("Q2  Serie temporal en punto (20 pasos) ", q2_serie),
        ("Q4  Zonas inundadas (1 paso, bbox)     ", q4_umbral),
        ("Q5  Evolución extensión (20 pasos bbox)", q5_evolucion),
        ("Q7  Hora llegada frente (20p, stream)  ", q7_llegada),
        ("Q10a Peligro adultos (1 paso, bbox)    ", q10a_peligro),
        ("Q12 Área inundada (20 pasos, completo) ", q12_area),
        ("Q13 Volumen (20 pasos, completo)       ", q13_volumen),
        ("Q16 Evacuación (20p stream, bbox)      ", q16_evacuacion),
    ]

    # ── Cabecera ──────────────────────────────────────────────────────────────
    W = 115
    print(f"\n{'═'*W}")
    print(f"  REQUERIMIENTOS DE LA APP — dataset: {args.dataset}  |  {reps} reps caché caliente")
    print(f"{'═'*W}")
    print(f"  URI TileDB : {uri}")
    print(f"  Dominio    : {ncols:,} × {nrows:,} celdas · {cs:.0f} m/celda")
    print(f"  BBox ref   : {(bx1-bx0)/1000:.0f} × {(by1-by0)/1000:.0f} km (centrada)")
    print()

    # ── Almacenamiento ────────────────────────────────────────────────────────
    tdb_mb  = dir_mb(uri)
    base_uri_mb = dir_mb(BASE_URI)
    venv_mb = dir_mb(VENV)
    scripts_mb = dir_mb(SCRIPTS)

    print(f"  ALMACENAMIENTO")
    print(f"  {'─'*50}")
    print(f"    TileDB {args.dataset}         : {tdb_mb:>8.0f} MB  ({tdb_mb/1024:.2f} GB)")
    print(f"    TileDB todos los datasets    : {base_uri_mb:>8.0f} MB  ({base_uri_mb/1024:.2f} GB)")
    print(f"    Entorno Python (venv)        : {venv_mb:>8.0f} MB  ({venv_mb/1024:.2f} GB)")
    print(f"    Scripts /claude/             : {scripts_mb:>8.0f} MB")
    print(f"    TOTAL (1 dataset + venv)     : {(tdb_mb + venv_mb + scripts_mb):>8.0f} MB  "
          f"({(tdb_mb + venv_mb + scripts_mb)/1024:.2f} GB)")
    print(f"    TOTAL (todos + venv)         : {(base_uri_mb + venv_mb + scripts_mb):>8.0f} MB  "
          f"({(base_uri_mb + venv_mb + scripts_mb)/1024:.2f} GB)")
    print()

    # ── RAM del proceso en reposo ─────────────────────────────────────────────
    proc = psutil.Process()
    rss_base_mb = proc.memory_info().rss / 1024**2
    print(f"  RAM proceso en reposo (imports cargados): {rss_base_mb:.0f} MB")
    print()

    # ── Queries ───────────────────────────────────────────────────────────────
    print(f"  RENDIMIENTO POR QUERY  ({reps} reps caliente)")
    print(f"  {'─'*W}")
    print(f"  {'Query':<44} {'Frío':>7}  {'Cal':>7}  {'RAM pico':>9}  "
          f"{'RSS +':>8}  {'CPU':>6}  {'Disco':>8}")
    print(f"  {'─'*44} {'─'*7}  {'─'*7}  {'─'*9}  {'─'*8}  {'─'*6}  {'─'*8}")

    resultados = []
    for nombre, fn in QUERIES:
        print(f"  {nombre}", end="  ", flush=True)
        try:
            m = medir(fn, reps)
            print(f"{m['t_frio']:>6.2f}s  {m['t_cal']:>6.2f}s  "
                  f"{m['ram_pico']:>7.0f} MB  {m['rss_delta']:>6.0f} MB  "
                  f"{m['cpu_s']:>5.2f}s  {m['disco_mb']:>6.0f} MB")
            resultados.append((nombre, m))
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    if resultados:
        ram_max = max(m["ram_pico"] for _, m in resultados)
        rss_max = max(m["rss_delta"] for _, m in resultados)
        print(f"\n{'─'*W}")
        print(f"  RAM pico máxima (una query):    {ram_max:.0f} MB")
        print(f"  RSS delta máximo (una query):   {rss_max:.0f} MB")
        rss_total = rss_base_mb + rss_max
        print(f"  RAM recomendada total:          {rss_total:.0f} MB "
              f"≈ {rss_total/1024:.1f} GB  (reposo + query más pesada)")
        print()
        print(f"  REQUERIMIENTOS MÍNIMOS DE DESPLIEGUE:")
        print(f"    CPU        : 2+ núcleos (queries streaming usan 1 núcleo al 100%)")
        print(f"    RAM        : {max(4096, int(rss_total * 1.5) // 512 * 512):,} MB  "
              f"({max(4, int(rss_total * 1.5 / 1024 + 0.5))} GB recomendados con margen)")
        print(f"    Disco      : {(tdb_mb + venv_mb + scripts_mb)/1024:.1f} GB mínimo por dataset "
              f"/ {(base_uri_mb + venv_mb + scripts_mb)/1024:.1f} GB con todos los datasets")
        print(f"    Red        : no necesaria (app local)")
        print(f"    Python     : 3.12+  |  TileDB Python  |  Streamlit  |  Plotly  |  Folium")

    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
