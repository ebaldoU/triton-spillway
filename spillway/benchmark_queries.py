#!/usr/bin/env python3
# ============================================================
# benchmark_queries.py
# Mide el rendimiento de distintos patrones de consulta sobre
# el array TileDB sparse de Triton.
#
# Consultas:
#   Q1  Escaneo completo — 4 atributos
#   Q2  Escaneo completo — 1 atributo (H)
#   Q3  Paso temporal único — 4 atributos
#   Q4  Caja espacial — 4 atributos (20 pasos)
#   Q5  Combinada: paso + caja espacial — 4 atributos
#
# Uso:
#   python benchmark_queries.py --dataset datos1
#   python benchmark_queries.py --dataset datos2
# ============================================================
from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import tiledb
from config import resolve_dataset


# ── Configuración ─────────────────────────────────────────────

REPS        = 1

TARGET_STEP = 9       # paso 10 de 20 (mitad de la simulación)

# Caja espacial: región central de 5 000 × 5 000 celdas
ROW_MIN, ROW_MAX = 14_600, 19_600
COL_MIN, COL_MAX =  9_020, 14_020

ALL_ATTRS   = ["H", "QX", "QY", "MH"]
SINGLE_ATTR = ["H"]


# ── Utilidades ────────────────────────────────────────────────

def result_bytes(result: dict) -> int:
    if "_total_bytes" in result:
        return int(result["_total_bytes"])
    return sum(v.nbytes for v in result.values() if hasattr(v, "nbytes"))


def sep(title=""):
    w = 72
    if title:
        print(f"\n{'─'*3} {title} {'─'*(w - len(title) - 5)}")
    else:
        print("─" * w)


def run_query(fn, reps: int) -> tuple[list[float], int]:
    times, nbytes = [], 0
    for i in range(reps):
        gc.collect()
        t0     = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if i == 0:
            nbytes = result_bytes(result)
        del result
        gc.collect()
    return times, nbytes


def print_row(label: str, times: list[float], nbytes: int) -> None:
    t    = times[0]
    mb   = nbytes / 1e6
    mbps = mb / t if t > 0 else 0
    print(f"  {label:<52}  {t:7.2f}s  {mb:7.0f} MB  {mbps:6.0f} MB/s")


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de consultas TileDB/Triton")
    parser.add_argument("--dataset", required=True,
                        help="Dataset a medir")
    args = parser.parse_args()
    tiledb_uri = f"/home/ebald/TFG/tiledb/triton_results/{resolve_dataset(args.dataset)}"

    print(f"\n{'═'*72}")
    print(f"  BENCHMARK DE CONSULTAS — TileDB Triton  ({args.dataset})")
    print(f"{'═'*72}")
    print(f"  URI:         {tiledb_uri}")
    print(f"  Repeticiones por consulta: {REPS}")
    print(f"  Paso de referencia: índice {TARGET_STEP}")
    print(f"  Caja espacial: filas [{ROW_MIN}:{ROW_MAX}], cols [{COL_MIN}:{COL_MAX}]")

    with tiledb.open(tiledb_uri, mode="r") as A:
        meta       = dict(A.meta)
        time_steps = json.loads(meta["time_steps"])
        n_stored   = int(meta.get("n_cells", 0))

    print(f"  Pasos: {time_steps}")
    if n_stored:
        print(f"  Celdas almacenadas: {n_stored:,}")

    n_steps = len(time_steps)

    sep("RESULTADOS")
    print(f"  {'Consulta':<52}  {'tiempo':>8}  {'datos':>9}   {'thrpt':>8}")
    print(f"  {'─'*52}  {'─'*8}  {'─'*9}   {'─'*8}")

    def q1():
        total_bytes = 0
        with tiledb.open(tiledb_uri, mode="r") as A:
            for s in range(n_steps):
                result = A.query(attrs=ALL_ATTRS)[s, :, :]
                total_bytes += sum(v.nbytes for v in result.values() if hasattr(v, "nbytes"))
                del result
        return {"_total_bytes": total_bytes}

    t, nb = run_query(q1, REPS)
    print_row("Q1  20 pasos × mapa completo  [H,QX,QY,MH]", t, nb)

    def q2():
        total_bytes = 0
        with tiledb.open(tiledb_uri, mode="r") as A:
            for s in range(n_steps):
                result = A.query(attrs=SINGLE_ATTR)[s, :, :]
                total_bytes += sum(v.nbytes for v in result.values() if hasattr(v, "nbytes"))
                del result
        return {"_total_bytes": total_bytes}

    t, nb = run_query(q2, REPS)
    print_row("Q2  20 pasos × mapa completo  [H]", t, nb)

    def q3():
        with tiledb.open(tiledb_uri, mode="r") as A:
            return A.query(attrs=ALL_ATTRS)[TARGET_STEP, :, :]

    t, nb = run_query(q3, REPS)
    print_row(f"Q3  1 paso (10_00) × mapa completo  [H,QX,QY,MH]", t, nb)

    def q4():
        total_bytes = 0
        with tiledb.open(tiledb_uri, mode="r") as A:
            for s in range(n_steps):
                result = A.query(attrs=ALL_ATTRS)[s, ROW_MIN:ROW_MAX, COL_MIN:COL_MAX]
                total_bytes += sum(v.nbytes for v in result.values() if hasattr(v, "nbytes"))
                del result
        return {"_total_bytes": total_bytes}

    t, nb = run_query(q4, REPS)
    print_row("Q4  20 pasos × región 5k×5k  [H,QX,QY,MH]", t, nb)

    def q5():
        with tiledb.open(tiledb_uri, mode="r") as A:
            return A.query(attrs=ALL_ATTRS)[TARGET_STEP, ROW_MIN:ROW_MAX, COL_MIN:COL_MAX]

    t, nb = run_query(q5, REPS)
    print_row("Q5  1 paso (10_00) × región 5k×5k  [H,QX,QY,MH]", t, nb)

    sep()
    print()


if __name__ == "__main__":
    main()
