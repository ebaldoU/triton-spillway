#!/usr/bin/env python3
# ============================================================
# geotiff_to_tiledb_sparse.py
# Carga GeoTIFFs de Triton (H, QX, QY, MH) en TileDB sparse.
#
# Filtro de compresión: ByteShuffle + ZSTD-9 (elegido tras el
# benchmark exhaustivo: ~8.9% más de compresión que ZSTD-1 sin
# penalización apreciable en lectura).
#
# Lectura por franjas (CHUNK_ROWS filas) para evitar cargar el
# raster completo (~3 GB/banda) en RAM. Memoria pico con 4 bandas:
#   4 × CHUNK_ROWS × 23040 cols × 4 B ≈ 1.8 GB (CHUNK_ROWS=5000)
#
# Uso:
#   python geotiff_to_tiledb_sparse.py --dataset datos1
#   python geotiff_to_tiledb_sparse.py --dataset datos2
# ============================================================
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import re
import time

import numpy as np
import rasterio
import tiledb
from config import BASE_URI, resolve_dataset


DEPTH_THRESHOLD  = 0.01   # 1 cm
CHUNK_ROWS       = 5000   # filas por franja
STEP_RE          = re.compile(r"^H_(\d{2}_\d{2})\.tif$")


# ── Helpers ──────────────────────────────────────────────────

def find_steps(gtiff_dir: Path) -> list[str]:
    return sorted(m.group(1) for p in gtiff_dir.glob("H_*.tif") if (m := STEP_RE.match(p.name)))


def read_meta(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "width":     src.width,
            "height":    src.height,
            "crs":       src.crs.to_string() if src.crs else None,
            "transform": tuple(src.transform),
            "bounds":    tuple(src.bounds),
            "nodata":    src.nodata,
        }


def check_companion_files(gtiff_dir: Path, step: str) -> None:
    for var in ("H", "QX", "QY", "MH"):
        p = gtiff_dir / f"{var}_{step}.tif"
        if not p.exists():
            raise FileNotFoundError(f"Falta {p.name} en {gtiff_dir}")


def create_sparse_array(uri: str, n_times: int, n_rows: int, n_cols: int) -> None:
    filters = tiledb.FilterList([tiledb.ByteShuffleFilter(), tiledb.ZstdFilter(level=9)])
    t_dim = tiledb.Dim("time", domain=(0, max(n_times - 1, 0)), tile=min(n_times, 16),  dtype=np.int32)
    r_dim = tiledb.Dim("row",  domain=(0, n_rows - 1),          tile=min(n_rows, 256),  dtype=np.int32)
    c_dim = tiledb.Dim("col",  domain=(0, n_cols - 1),          tile=min(n_cols, 256),  dtype=np.int32)
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(t_dim, r_dim, c_dim),
        attrs=[
            tiledb.Attr("H",  dtype=np.float32, filters=filters),
            tiledb.Attr("QX", dtype=np.float32, filters=filters),
            tiledb.Attr("QY", dtype=np.float32, filters=filters),
            tiledb.Attr("MH", dtype=np.float32, filters=filters),
        ],
        sparse=True,
        allows_duplicates=False,
        cell_order="row-major",
        tile_order="row-major",
    )
    tiledb.SparseArray.create(uri, schema)


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Carga GeoTIFFs de Triton en TileDB sparse.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset a cargar, p. ej. datos1, datos2, datos3… (resuelve las rutas GTIFF_DIR y TILEDB_URI).")
    parser.add_argument("--gtiff-dir",
                        help="Directorio fuente con los GeoTIFFs. Por defecto: /home/ebald/TFG/{dataset}")
    args = parser.parse_args()

    gtiff_dir  = Path(args.gtiff_dir) if args.gtiff_dir else Path(f"/home/ebald/TFG/{args.dataset}")
    tiledb_uri = f"{BASE_URI}/{resolve_dataset(args.dataset)}"

    t_start = time.perf_counter()

    steps = find_steps(gtiff_dir)
    if not steps:
        raise RuntimeError(f"No se encontraron GeoTIFF en {gtiff_dir}")

    print(f"Dataset: {args.dataset}")
    print(f"  GTIFF_DIR:  {gtiff_dir}")
    print(f"  TILEDB_URI: {tiledb_uri}")
    print(f"Pasos encontrados: {len(steps)}")
    for step in steps:
        check_companion_files(gtiff_dir, step)

    ref_meta = read_meta(gtiff_dir / f"H_{steps[0]}.tif")
    n_rows, n_cols = int(ref_meta["height"]), int(ref_meta["width"])
    print(f"Resolución: {n_cols} x {n_rows} px  |  Franja: {CHUNK_ROWS} filas")

    if tiledb.array_exists(tiledb_uri):
        tiledb.remove(tiledb_uri)
    create_sparse_array(tiledb_uri, len(steps), n_rows, n_cols)

    total_cells = 0

    with tiledb.SparseArray(tiledb_uri, mode="w") as A:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for t_index, step in enumerate(steps):
                t_step     = time.perf_counter()
                step_cells = 0

                acc = {var: [] for var in ("H", "QX", "QY", "MH")}
                acc_rows, acc_cols = [], []

                srcs = {
                    var: rasterio.open(gtiff_dir / f"{var}_{step}.tif")
                    for var in ("H", "QX", "QY", "MH")
                }
                try:
                    for row_off in range(0, n_rows, CHUNK_ROWS):
                        actual_rows = min(CHUNK_ROWS, n_rows - row_off)
                        window = rasterio.windows.Window(0, row_off, n_cols, actual_rows)

                        def _read(var, w=window, s=srcs):
                            return var, s[var].read(1, window=w).astype(np.float32)
                        chunk = dict(pool.map(_read, ("H", "QX", "QY", "MH")))

                        wet_mask      = chunk["H"] >= DEPTH_THRESHOLD
                        local_r, cols = np.nonzero(wet_mask)

                        if local_r.size > 0:
                            acc_rows.append((local_r + row_off).astype(np.int32))
                            acc_cols.append(cols.astype(np.int32))
                            for var in ("H", "QX", "QY", "MH"):
                                acc[var].append(chunk[var][local_r, cols])
                            step_cells += local_r.size
                finally:
                    for src in srcs.values():
                        src.close()

                if step_cells > 0:
                    all_rows = np.concatenate(acc_rows)
                    all_cols = np.concatenate(acc_cols)
                    t_vals   = np.full(step_cells, t_index, dtype=np.int32)
                    A[t_vals, all_rows, all_cols] = {
                        var: np.concatenate(acc[var]) for var in ("H", "QX", "QY", "MH")
                    }

                total_cells += step_cells
                elapsed = time.perf_counter() - t_step
                print(f"  [{t_index+1}/{len(steps)}] paso {step} — "
                      f"{step_cells:,} celdas mojadas  ({elapsed:.1f}s)")

    with tiledb.open(tiledb_uri, mode="m") as A:
        A.meta["crs"]               = ref_meta["crs"] or ""
        A.meta["transform"]         = json.dumps(ref_meta["transform"])
        A.meta["bounds"]            = json.dumps(ref_meta["bounds"])
        A.meta["width"]             = int(ref_meta["width"])
        A.meta["height"]            = int(ref_meta["height"])
        A.meta["nodata"]            = "None" if ref_meta["nodata"] is None else str(ref_meta["nodata"])
        A.meta["time_steps"]        = json.dumps(steps)
        A.meta["source_dir"]        = str(gtiff_dir)
        A.meta["source_dataset"]    = args.dataset
        A.meta["depth_threshold_m"] = float(DEPTH_THRESHOLD)
        A.meta["storage_mode"]      = "sparse_bshuffle_zstd9"
        A.meta["zero_rule"]         = "Si H < threshold, no se almacena; celda ausente = H=QX=QY=MH=0."

    t_total = time.perf_counter() - t_start
    print()
    print(f"Array TileDB sparse creado en:    {tiledb_uri}")
    print(f"Pasos temporales:                  {len(steps)}")
    print(f"Total celdas almacenadas:          {total_cells:,}")
    print(f"Umbral aplicado:                   H >= {DEPTH_THRESHOLD} m")
    print(f"Tiempo total:                      {t_total:.1f}s")


if __name__ == "__main__":
    main()
