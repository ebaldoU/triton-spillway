#!/usr/bin/env python3
"""
benchmark_vs_geotiff.py — Comparativa TileDB sparse vs lectura directa de GeoTIFF.

6 casos de prueba:
  C1: Leer un instante temporal completo (4 variables)
  C2: Leer una ventana espacial concreta en un instante
  C3: Serie temporal de un píxel a través de los 20 instantes
  C4: Calcular peligro para personas (Russo) sobre toda la serie
  C5: Estadísticos (media, máximo, std) en zona de interés
  C6: Exportar subconjunto espaciotemporal a GeoTIFF

Métricas: tiempo frío (1ª lectura), tiempo caliente (media REPS siguientes), RAM pico.

Uso:
  python benchmark_vs_geotiff.py --dataset datos1
  python benchmark_vs_geotiff.py --dataset datos2
"""
from __future__ import annotations

import argparse
import gc
import os
import tempfile
import time
import tracemalloc
from pathlib import Path

import json

import numpy as np
import rasterio
import rasterio.windows
import tiledb
from config import resolve_dataset


# ── Configuración ─────────────────────────────────────────────
VARS     = ["H", "QX", "QY", "MH"]
STEPS    = [f"{i:02d}_00" for i in range(1, 21)]
STEP_IDX = 9    # paso 10_00 (hora 60)
REPS     = 3

XLLCORNER, YLLCORNER = 680_000.0, 4_330_000.0
NROWS, NCOLS, CELLSIZE = 34_200, 23_040, 10.0

# Ventana espacial: 50×50 km en zona suroeste (zona húmeda)
WIN_X0, WIN_Y0 = 685_000.0, 4_335_000.0
WIN_X1, WIN_Y1 = 735_000.0, 4_385_000.0

# Píxel de referencia (dentro de la ventana húmeda)
PIX_X, PIX_Y = 695_000.0, 4_345_000.0


# ── Conversiones ──────────────────────────────────────────────

def load_domain_constants(uri: str) -> None:
    """Lee NROWS, NCOLS, CELLSIZE, XLLCORNER, YLLCORNER desde los metadatos TileDB."""
    global NROWS, NCOLS, CELLSIZE, XLLCORNER, YLLCORNER
    with tiledb.open(uri, mode="r") as A:
        meta = dict(A.meta)
    tr = json.loads(meta["transform"])
    NCOLS      = int(meta["width"])
    NROWS      = int(meta["height"])
    CELLSIZE   = abs(float(tr[0]))
    XLLCORNER  = float(tr[2])
    YLLCORNER  = float(tr[5]) + float(tr[4]) * NROWS


def coord_a_pixel(x, y):
    col = int((x - XLLCORNER) / CELLSIZE)
    row = int((YLLCORNER + NROWS * CELLSIZE - y) / CELLSIZE)
    return row, col


def bbox_a_win(x0, y0, x1, y1):
    c0 = int((x0 - XLLCORNER) / CELLSIZE)
    r0 = int((YLLCORNER + NROWS * CELLSIZE - y1) / CELLSIZE)
    c1 = int((x1 - XLLCORNER) / CELLSIZE)
    r1 = int((YLLCORNER + NROWS * CELLSIZE - y0) / CELLSIZE)
    return rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)


def bbox_a_idx(x0, y0, x1, y1):
    c0 = int((x0 - XLLCORNER) / CELLSIZE)
    c1 = int((x1 - XLLCORNER) / CELLSIZE)
    r0 = int((YLLCORNER + NROWS * CELLSIZE - y1) / CELLSIZE)
    r1 = int((YLLCORNER + NROWS * CELLSIZE - y0) / CELLSIZE)
    return r0, r1, c0, c1


def dir_mb(path) -> float:
    return sum(f.stat().st_size for f in Path(str(path)).rglob("*") if f.is_file()) / 1024**2


# ── Medición ──────────────────────────────────────────────────

def medir(fn) -> tuple[float, float, float]:
    """Ejecuta fn() 1+REPS veces. Retorna (t_frio, t_caliente_media, ram_pico_mb)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    t_frio = time.perf_counter() - t0
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_mb = pico / 1024**2

    tiempos = []
    for _ in range(REPS):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        tiempos.append(time.perf_counter() - t0)
    return t_frio, float(np.mean(tiempos)), ram_mb


# ── Implementaciones TileDB ───────────────────────────────────

def _make_tdb(tiledb_uri: str, gtiff_dir: Path):
    def tdb_c1():
        with tiledb.open(tiledb_uri, mode="r") as A:
            _ = A.query(attrs=VARS)[STEP_IDX, :, :]

    def tdb_c2():
        r0, r1, c0, c1 = bbox_a_idx(WIN_X0, WIN_Y0, WIN_X1, WIN_Y1)
        with tiledb.open(tiledb_uri, mode="r") as A:
            _ = A.query(attrs=VARS)[STEP_IDX, r0:r1, c0:c1]

    def tdb_c3():
        row, col = coord_a_pixel(PIX_X, PIX_Y)
        with tiledb.open(tiledb_uri, mode="r") as A:
            _ = A.query(attrs=["H"])[0:20, row, col]

    def tdb_c4():
        with tiledb.open(tiledb_uri, mode="r") as A:
            total = 0
            for t in range(20):
                res = A.query(attrs=["H", "QX", "QY"])[t, :, :]
                Q = np.sqrt(res["QX"]**2 + res["QY"]**2)
                total += int(((res["H"] > 0.5) | (Q > 0.5)).sum())
        return total

    def tdb_c5():
        r0, r1, c0, c1 = bbox_a_idx(WIN_X0, WIN_Y0, WIN_X1, WIN_Y1)
        with tiledb.open(tiledb_uri, mode="r") as A:
            res = A.query(attrs=["H"])[STEP_IDX, r0:r1, c0:c1]
        wet = res["H"][res["H"] > 0]
        return {} if len(wet) == 0 else {"media": float(wet.mean()), "max": float(wet.max()), "std": float(wet.std())}

    def tdb_c6():
        r0, r1, c0, c1 = bbox_a_idx(WIN_X0, WIN_Y0, WIN_X1, WIN_Y1)
        with tiledb.open(tiledb_uri, mode="r") as A:
            res = A.query(attrs=["H"])[STEP_IDX, r0:r1, c0:c1]
        grid = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
        if len(res["H"]) > 0:
            grid[res["row"] - r0, res["col"] - c0] = res["H"]
        transform = rasterio.transform.from_origin(WIN_X0, WIN_Y1, CELLSIZE, CELLSIZE)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp = f.name
        with rasterio.open(tmp, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1],
                           count=1, dtype="float32", crs="EPSG:32614",
                           transform=transform, nodata=-9999, compress="deflate") as dst:
            dst.write(grid, 1)
        os.unlink(tmp)

    return [tdb_c1, tdb_c2, tdb_c3, tdb_c4, tdb_c5, tdb_c6]


# ── Implementaciones GeoTIFF ──────────────────────────────────

def _make_tif(gtiff_dir: Path):
    CHUNK_ROWS_C4 = 2000  # ~550 MB pico (3 vars × 2000 × 23040 × 4 B)

    def tif_c1():
        step = STEPS[STEP_IDX]
        for v in VARS:
            with rasterio.open(gtiff_dir / f"{v}_{step}.tif") as src:
                _ = src.read(1)

    def tif_c2():
        step = STEPS[STEP_IDX]
        win = bbox_a_win(WIN_X0, WIN_Y0, WIN_X1, WIN_Y1)
        for v in VARS:
            with rasterio.open(gtiff_dir / f"{v}_{step}.tif") as src:
                _ = src.read(1, window=win)

    def tif_c3():
        row, col = coord_a_pixel(PIX_X, PIX_Y)
        win = rasterio.windows.Window(col, row, 1, 1)
        for step in STEPS:
            with rasterio.open(gtiff_dir / f"H_{step}.tif") as src:
                _ = src.read(1, window=win)

    def tif_c4():
        total = 0
        for step in STEPS:
            srcs = {v: rasterio.open(gtiff_dir / f"{v}_{step}.tif") for v in ["H", "QX", "QY"]}
            try:
                height = srcs["H"].height
                for row_off in range(0, height, CHUNK_ROWS_C4):
                    actual = min(CHUNK_ROWS_C4, height - row_off)
                    win = rasterio.windows.Window(0, row_off, NCOLS, actual)
                    H  = srcs["H"].read(1,  window=win).astype(np.float32)
                    QX = srcs["QX"].read(1, window=win).astype(np.float32)
                    QY = srcs["QY"].read(1, window=win).astype(np.float32)
                    Q = np.sqrt(QX**2 + QY**2)
                    total += int(((H > 0.5) | (Q > 0.5)).sum())
            finally:
                for s in srcs.values():
                    s.close()
        return total

    def tif_c5():
        step = STEPS[STEP_IDX]
        win = bbox_a_win(WIN_X0, WIN_Y0, WIN_X1, WIN_Y1)
        with rasterio.open(gtiff_dir / f"H_{step}.tif") as src:
            H = src.read(1, window=win).astype(np.float32)
        wet = H[H >= 0.01]
        return {} if len(wet) == 0 else {"media": float(wet.mean()), "max": float(wet.max()), "std": float(wet.std())}

    def tif_c6():
        step = STEPS[STEP_IDX]
        win = bbox_a_win(WIN_X0, WIN_Y0, WIN_X1, WIN_Y1)
        with rasterio.open(gtiff_dir / f"H_{step}.tif") as src:
            H = src.read(1, window=win)
            transform = src.window_transform(win)
            crs = src.crs
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp = f.name
        with rasterio.open(tmp, "w", driver="GTiff", height=H.shape[0], width=H.shape[1],
                           count=1, dtype=H.dtype, crs=crs,
                           transform=transform, nodata=src.nodata, compress="deflate") as dst:
            dst.write(H, 1)
        os.unlink(tmp)

    return [tif_c1, tif_c2, tif_c3, tif_c4, tif_c5, tif_c6]


# ── Main ──────────────────────────────────────────────────────

NOMBRES = [
    "C1 — Instante completo (4 vars)",
    "C2 — Ventana 50×50 km (4 vars)",
    "C3 — Serie temporal 1 píxel (20 pasos)",
    "C4 — Peligro personas, 20 pasos (H,QX,QY)",
    "C5 — Estadísticos zona 50×50 km",
    "C6 — Exportar ventana a GeoTIFF",
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark TileDB vs GeoTIFF")
    parser.add_argument("--dataset", required=True,
                        help="Dataset a comparar")
    args = parser.parse_args()

    tiledb_uri = f"/home/ebald/TFG/tiledb/triton_results/{resolve_dataset(args.dataset)}"
    gtiff_dir  = Path(f"/home/ebald/TFG/{args.dataset}")
    have_gtiff = gtiff_dir.exists() and any(gtiff_dir.glob("*.tif"))

    load_domain_constants(tiledb_uri)

    tdb_mb = dir_mb(tiledb_uri)
    # Tamaño GeoTIFF estimado: NROWS × NCOLS × 20 pasos × 4 vars × 4 bytes (float32)
    gtiff_mb_est = NROWS * NCOLS * len(STEPS) * len(VARS) * 4 / 1024**2

    if have_gtiff:
        gtiff_mb = sum(
            (gtiff_dir / f"{v}_{s}.tif").stat().st_size
            for v in VARS for s in STEPS
        ) / 1024**2
    else:
        gtiff_mb = gtiff_mb_est

    print(f"\n{'═'*95}")
    print(f"  BENCHMARK TileDB vs GeoTIFF — dataset: {args.dataset}  |  {REPS} reps caché caliente")
    print(f"{'═'*95}")
    if have_gtiff:
        print(f"  GeoTIFFs originales (20 pasos × 4 vars): {gtiff_mb:>8.0f} MB")
    else:
        print(f"  GeoTIFFs originales (estimado sin compresión, float32): {gtiff_mb:>8.0f} MB")
    print(f"  TileDB sparse   (ByteShuffle+ZSTD-9):    {tdb_mb:>8.0f} MB")
    print(f"  Reducción de almacenamiento: {(1 - tdb_mb/gtiff_mb)*100:.1f}%")
    print(f"\n  Ventana: ({WIN_X0:.0f},{WIN_Y0:.0f}) → ({WIN_X1:.0f},{WIN_Y1:.0f})"
          f"  [{(WIN_X1-WIN_X0)/1000:.0f}×{(WIN_Y1-WIN_Y0)/1000:.0f} km]")
    print(f"  Paso de referencia: {STEPS[STEP_IDX]} (hora {(STEP_IDX+1)*6} h)")

    fns_tdb = _make_tdb(tiledb_uri, gtiff_dir)
    resultados = []

    if have_gtiff:
        fns_tif = _make_tif(gtiff_dir)
        print()
        print(f"  {'Caso':<45}  {'Frío TDB':>8}  {'Cal TDB':>8}  {'Frío TIF':>8}  "
              f"{'Cal TIF':>8}  {'Factor':>7}  {'RAM TDB':>8}  {'RAM TIF':>8}")
        print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}")
        for nombre, fn_tdb, fn_tif in zip(NOMBRES, fns_tdb, fns_tif):
            print(f"  {nombre:<45}", end="  ", flush=True)
            tf_tdb, tc_tdb, ram_tdb = medir(fn_tdb)
            print(f"{tf_tdb:>7.3f}s  {tc_tdb:>7.3f}s", end="  ", flush=True)
            tf_tif, tc_tif, ram_tif = medir(fn_tif)
            print(f"{tf_tif:>7.3f}s  {tc_tif:>7.3f}s", end="  ", flush=True)
            factor = tc_tif / tc_tdb if tc_tdb > 0 else float("inf")
            signo = f"{factor:.1f}×" if factor >= 1 else f"1/{1/factor:.1f}×"
            print(f"{signo:>7}  {ram_tdb:>6.0f}MB  {ram_tif:>6.0f}MB")
            resultados.append((nombre, tf_tdb, tc_tdb, tf_tif, tc_tif, factor, ram_tdb, ram_tif))

        print(f"\n{'═'*95}")
        print("  Factor = tiempo_caliente_GeoTIFF / tiempo_caliente_TileDB")
        print("  Factor > 1 → TileDB más rápido  |  Factor < 1 → GeoTIFF más rápido")
        print()
        ventajas_tdb = [(n, f) for n, *_, f, _, _ in resultados if f > 1]
        ventajas_tif = [(n, f) for n, *_, f, _, _ in resultados if f < 1]
        print(f"  TileDB más rápido en {len(ventajas_tdb)}/6 casos:")
        for n, f in ventajas_tdb:
            print(f"    • {n}: {f:.1f}× más rápido")
        if ventajas_tif:
            print(f"  GeoTIFF más rápido en {len(ventajas_tif)}/6 casos:")
            for n, f in ventajas_tif:
                print(f"    • {n}: {1/f:.1f}× más rápido")
    else:
        print(f"  (GeoTIFFs eliminados tras ETL — solo tiempos TileDB)")
        print()
        print(f"  {'Caso':<45}  {'Frío TDB':>8}  {'Cal TDB':>8}  {'RAM TDB':>8}")
        print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*8}")
        for nombre, fn_tdb in zip(NOMBRES, fns_tdb):
            print(f"  {nombre:<45}", end="  ", flush=True)
            tf_tdb, tc_tdb, ram_tdb = medir(fn_tdb)
            print(f"{tf_tdb:>7.3f}s  {tc_tdb:>7.3f}s  {ram_tdb:>6.0f}MB")

    print(f"{'═'*95}\n")


if __name__ == "__main__":
    main()
