#!/usr/bin/env python3
# ============================================================
# query_max_depth_tiledb.py
# "¿Dónde hay más profundidad de agua?"
#
# Uso:
#   python query_max_depth_tiledb.py --dataset datos1
#   python query_max_depth_tiledb.py --dataset datos2 --top 10
# ============================================================
from __future__ import annotations

import argparse
import json

import numpy as np
import tiledb
from config import BASE_URI, resolve_dataset


def query_max_depth(uri: str, top_n: int) -> None:
    with tiledb.open(uri, mode="r") as A:
        meta       = dict(A.meta)
        transform  = json.loads(meta["transform"])
        time_steps = json.loads(meta["time_steps"])
        result     = A.query(attrs=["H"])[:]

    a, b, c = transform[0], transform[1], transform[2]
    d, e, f = transform[3], transform[4], transform[5]

    rows   = result["row"]
    cols   = result["col"]
    times  = result["time"]
    H_vals = result["H"]

    idx     = int(np.argmax(H_vals))
    max_h   = float(H_vals[idx])
    max_row = int(rows[idx])
    max_col = int(cols[idx])
    max_t   = int(times[idx])

    x = c + max_col * a
    y = f + max_row * e

    print("=" * 50)
    print("  MÁXIMA PROFUNDIDAD DE AGUA")
    print("=" * 50)
    print(f"  Profundidad:    {max_h:.4f} m")
    print(f"  Paso temporal:  {time_steps[max_t]} (índice {max_t})")
    print(f"  Fila / Columna: {max_row} / {max_col}")
    print(f"  Coordenadas:    X={x:.2f}  Y={y:.2f}  ({meta['crs']})")
    print()
    print(f"  Top {top_n} celdas con mayor profundidad:")
    print(f"  {'#':<4} {'H (m)':>8}  {'paso':<8}  {'fila':>5}  {'col':>5}  {'X':>10}  {'Y':>10}")
    print(f"  {'-'*4} {'-'*8}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}")

    for rank, k in enumerate(np.argsort(H_vals)[-top_n:][::-1], start=1):
        xi = c + int(cols[k]) * a
        yi = f + int(rows[k]) * e
        print(f"  {rank:<4} {H_vals[k]:>8.4f}  {time_steps[int(times[k])]:<8}  "
              f"{rows[k]:>5}  {cols[k]:>5}  {xi:>10.2f}  {yi:>10.2f}")

    print()
    print(f"  Total celdas mojadas: {H_vals.size:,}")
    print(f"  Umbral aplicado:      H >= {meta.get('depth_threshold_m', '?')} m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Localiza la mayor profundidad en TileDB/Triton")
    parser.add_argument("--dataset", required=True,
                        help="Dataset a consultar")
    parser.add_argument("--top", type=int, default=5, metavar="N",
                        help="Número de celdas a mostrar (defecto: 5)")
    args = parser.parse_args()
    query_max_depth(f"{BASE_URI}/{resolve_dataset(args.dataset)}", args.top)


if __name__ == "__main__":
    main()
