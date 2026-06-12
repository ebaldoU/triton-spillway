#!/usr/bin/env python3
# ============================================================
# make_demo_subset.py
# Genera un subconjunto de demostración de un array TileDB
# sparse recortando a un bounding box espacial. El array
# resultante mantiene el esquema y el dominio completos, por lo
# que la app y las consultas funcionan sin cambios.
#
# Bbox por defecto: ventana 50x50 km del benchmark (zona húmeda
# suroeste), filas 28700-33700, columnas 500-5500.
#
# Uso:
#   python make_demo_subset.py --dataset datos1 --out /ruta/demo_datos1
# ============================================================
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tiledb
from config import BASE_URI, resolve_dataset
from geotiff_to_tiledb_sparse import create_sparse_array

# Ventana del benchmark: (685000,4335000)-(735000,4385000) EPSG:32614
# Límites inclusivos, mismos índices que devuelve bbox_a_indices()
ROW0, ROW1 = 28_700, 33_700
COL0, COL1 = 500, 5_500
VARS = ("H", "QX", "QY", "MH")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recorta un array TileDB sparse a un bbox de demostración.")
    parser.add_argument("--dataset", required=True, help="Dataset fuente, p. ej. datos1")
    parser.add_argument("--out", required=True, help="URI del array de salida")
    args = parser.parse_args()

    src_uri = f"{BASE_URI}/{resolve_dataset(args.dataset)}"
    t_start = time.perf_counter()

    with tiledb.open(src_uri, mode="r") as A:
        meta = dict(A.meta)
        dom = A.schema.domain
        n_times = int(dom.dim("time").domain[1]) + 1
        n_rows = int(dom.dim("row").domain[1]) + 1
        n_cols = int(dom.dim("col").domain[1]) + 1

    print(f"Fuente:  {src_uri}")
    print(f"Salida:  {args.out}")
    print(f"Bbox:    filas {ROW0}-{ROW1}, cols {COL0}-{COL1}  ({ROW1-ROW0+1}x{COL1-COL0+1} px, inclusivo)")

    if tiledb.array_exists(args.out):
        tiledb.remove(args.out)
    create_sparse_array(args.out, n_times, n_rows, n_cols)

    total_cells = 0
    with tiledb.open(src_uri, mode="r") as A, tiledb.SparseArray(args.out, mode="w") as B:
        q = A.query(attrs=VARS, coords=True)
        for t in range(n_times):
            t_step = time.perf_counter()
            data = q.multi_index[t, ROW0:ROW1, COL0:COL1]
            n = data["row"].size
            if n > 0:
                B[data["time"], data["row"], data["col"]] = {v: data[v] for v in VARS}
            total_cells += n
            print(f"  [{t+1}/{n_times}] {n:,} celdas  ({time.perf_counter()-t_step:.1f}s)")

    with tiledb.open(args.out, mode="m") as B:
        for k, v in meta.items():
            B.meta[k] = v
        B.meta["demo_subset"] = json.dumps({
            "source_dataset": args.dataset,
            "bbox_rows": [ROW0, ROW1],
            "bbox_cols": [COL0, COL1],
        })

    print()
    print(f"Subset creado: {total_cells:,} celdas en {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
