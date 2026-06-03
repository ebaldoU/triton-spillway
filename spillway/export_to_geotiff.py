#!/usr/bin/env python3
# Exporta datos del array TileDB sparse a GeoTIFF compatible con QGIS.
#
# Uso:
#   python export_to_geotiff.py --dataset datos1 --step 10_00
#   python export_to_geotiff.py --dataset datos2 --step 10_00 --var H
#   python export_to_geotiff.py --dataset datos1 --all-steps --var H
#   python export_to_geotiff.py --dataset datos2 --step 10_00 --bbox 14600 19600 9020 14020

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import tiledb
from config import BASE_URI, OUTPUT_DIR, resolve_dataset
ALL_VARS   = ["H", "QX", "QY", "MH"]
NODATA     = -9999.0


# ── Utilidades ────────────────────────────────────────────────────────

def step_to_index(step_name: str, time_steps: list[str]) -> int:
    try:
        return time_steps.index(step_name)
    except ValueError:
        print(f"Error: paso '{step_name}' no encontrado. Disponibles: {time_steps}")
        sys.exit(1)


def build_transform(meta: dict, row_min: int, col_min: int) -> rasterio.Affine:
    """Construye el transform afín desde los metadatos del array TileDB."""
    tr = json.loads(meta["transform"])   # (a, b, c, d, e, f) formato rasterio
    x0 = tr[2] + col_min * tr[0]
    y0 = tr[5] + row_min * tr[4]
    return from_origin(x0, y0, abs(tr[0]), abs(tr[4]))


def sparse_to_dense(rows: np.ndarray, cols: np.ndarray, values: np.ndarray,
                    row_min: int, col_min: int, nrows: int, ncols: int) -> np.ndarray:
    grid = np.full((nrows, ncols), NODATA, dtype=np.float32)
    r = rows - row_min
    c = cols - col_min
    mask = (r >= 0) & (r < nrows) & (c >= 0) & (c < ncols)
    grid[r[mask], c[mask]] = values[mask]
    return grid


def write_geotiff(path: Path, grid: np.ndarray, transform: rasterio.Affine, crs: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff",
                       height=grid.shape[0], width=grid.shape[1],
                       count=1, dtype=np.float32, crs=crs,
                       transform=transform, nodata=NODATA, compress="deflate") as dst:
        dst.write(grid, 1)
    print(f"  → {path}  ({grid.shape[1]}×{grid.shape[0]} px)")


# ── Exportación ───────────────────────────────────────────────────────

def export_step(tiledb_uri: str, meta: dict, step_idx: int, step_name: str,
                vars_: list[str], row_min: int, row_max: int,
                col_min: int, col_max: int, output_dir: Path) -> None:
    nrows = row_max - row_min
    ncols = col_max - col_min
    transform = build_transform(meta, row_min, col_min)
    crs = meta.get("crs", "EPSG:32614")

    print(f"Exportando paso {step_name} "
          f"[rows {row_min}:{row_max}, cols {col_min}:{col_max}] vars={vars_}")

    with tiledb.open(tiledb_uri, mode="r") as A:
        result = A.query(attrs=vars_)[step_idx, row_min:row_max, col_min:col_max]

    rows = result["row"]
    cols = result["col"]
    for var in vars_:
        grid = sparse_to_dense(rows, cols, result[var], row_min, col_min, nrows, ncols)
        write_geotiff(output_dir / f"{var}_{step_name}.tif", grid, transform, crs)


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta TileDB → GeoTIFF")
    parser.add_argument("--dataset", required=True,
                        help="Dataset a exportar")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--step",       help="Paso a exportar, ej. 10_00")
    group.add_argument("--all-steps",  action="store_true", help="Exportar todos los pasos")
    parser.add_argument("--var",  default="all",
                        help="Variable: H, QX, QY, MH o all (defecto: all)")
    parser.add_argument("--bbox", nargs=4, type=int,
                        metavar=("ROW_MIN", "ROW_MAX", "COL_MIN", "COL_MAX"),
                        help="Subregión espacial (defecto: dominio completo)")
    parser.add_argument("--out", default=str(OUTPUT_DIR),
                        help=f"Directorio de salida (defecto: {OUTPUT_DIR})")
    args = parser.parse_args()

    tiledb_uri = f"{BASE_URI}/{resolve_dataset(args.dataset)}"
    output_dir = Path(args.out)
    vars_ = ALL_VARS if args.var == "all" else [args.var.upper()]

    with tiledb.open(tiledb_uri, mode="r") as A:
        meta = dict(A.meta)
        time_steps = json.loads(meta["time_steps"])
        n_rows = int(meta["height"])
        n_cols = int(meta["width"])

    row_min, row_max = (0, n_rows)
    col_min, col_max = (0, n_cols)
    if args.bbox:
        row_min, row_max, col_min, col_max = args.bbox

    if args.all_steps:
        for idx, name in enumerate(time_steps):
            export_step(tiledb_uri, meta, idx, name, vars_, row_min, row_max, col_min, col_max, output_dir)
    else:
        idx = step_to_index(args.step, time_steps)
        export_step(tiledb_uri, meta, idx, args.step, vars_, row_min, row_max, col_min, col_max, output_dir)

    print(f"\nListo. Archivos en: {output_dir}")


if __name__ == "__main__":
    main()
