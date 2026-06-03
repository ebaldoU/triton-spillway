#!/usr/bin/env python3
"""
verify_tiledb.py — Verifica la integridad del array TileDB comparando con los GeoTIFF originales.

Muestrea N celdas mojadas al azar de un paso temporal y comprueba que los valores
H, QX, QY, MH en TileDB coinciden con los del GeoTIFF correspondiente.

Uso:
  python verify_tiledb.py --dataset datos1
  python verify_tiledb.py --dataset datos2 --step 10_00 --n 2000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
import tiledb
from config import BASE_URI, GTIFF_BASE_URI, resolve_dataset

TOLERANCE      = 1e-4
PASS_THRESHOLD = 99.0


def sample_tiledb(uri: str, step: str, n: int) -> tuple[dict, list[str], dict, np.ndarray, np.ndarray]:
    """Lee el paso indicado de TileDB y devuelve N celdas muestreadas al azar."""
    with tiledb.open(uri, mode="r") as A:
        meta       = dict(A.meta)
        time_steps = json.loads(meta["time_steps"])
        if step not in time_steps:
            raise ValueError(f"Paso '{step}' no encontrado. Disponibles: {time_steps}")
        t_idx = time_steps.index(step)
        res   = A.query(attrs=["H", "QX", "QY", "MH"])[t_idx, :, :]

    n_wet  = len(res["H"])
    n_pick = min(n, n_wet)
    rng    = np.random.default_rng(42)
    idx    = rng.choice(n_wet, n_pick, replace=False)

    sample = {v: res[v][idx] for v in ("H", "QX", "QY", "MH")}
    rows   = res["row"][idx]
    cols   = res["col"][idx]
    return meta, time_steps, sample, rows, cols


def read_gtiff_at_pixels(gtiff_path: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Lee el valor GeoTIFF para cada (row, col) agrupando por fila para minimizar I/O."""
    vals = np.zeros(len(rows), dtype=np.float32)
    row_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, (r, c) in enumerate(zip(rows, cols)):
        row_groups[int(r)].append((i, int(c)))

    with rasterio.open(gtiff_path) as src:
        for row, items in row_groups.items():
            cs = [c for _, c in items]
            c0, c1 = min(cs), max(cs)
            win      = rasterio.windows.Window(c0, row, c1 - c0 + 1, 1)
            row_data = src.read(1, window=win)[0]
            for sample_i, col in items:
                vals[sample_i] = row_data[col - c0]
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check TileDB vs GeoTIFF")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--step", default="10_00",
                        help="Paso temporal a verificar (defecto: 10_00)")
    parser.add_argument("--n", type=int, default=1000,
                        help="Celdas a muestrear (defecto: 1000)")
    args = parser.parse_args()

    uri       = f"{BASE_URI}/{resolve_dataset(args.dataset)}"
    gtiff_dir = Path(GTIFF_BASE_URI) / args.dataset

    print(f"Verificando {args.dataset}  |  paso {args.step}  |  N={args.n}")

    # Verificar fragment count (debe ser 20, uno por paso temporal)
    frags = tiledb.array_fragments(uri)
    n_frags = len(frags)
    expected = 20
    frag_ok = n_frags == expected
    print(f"Fragmentos TileDB: {n_frags}  "
          f"{'✅ OK' if frag_ok else f'❌ AVISO: esperados {expected} — array probablemente consolidado (queries lentas)'}")

    try:
        meta, time_steps, sample, rows, cols = sample_tiledb(uri, args.step, args.n)
    except ValueError as e:
        print(f"Error: {e}")
        return

    n_pick = len(rows)
    print(f"Celdas mojadas en TileDB (paso {args.step}): {meta.get('depth_threshold_m', '?')} m umbral")
    print(f"Muestra: {n_pick} celdas\n")

    print(f"  {'Var':<5}  {'Max dif':>10}  {'Media dif':>10}  {'OK (dif<1e-4)':>14}  {'GeoTIFF':>12}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*12}")

    all_ok = True
    for var in ("H", "QX", "QY", "MH"):
        gtiff_path = gtiff_dir / f"{var}_{args.step}.tif"
        if not gtiff_path.exists():
            print(f"  {var:<5}  (GeoTIFF no encontrado: {gtiff_path.name})")
            continue

        tdb_vals   = sample[var].astype(np.float64)
        gtiff_vals = read_gtiff_at_pixels(gtiff_path, rows, cols).astype(np.float64)
        diffs      = np.abs(tdb_vals - gtiff_vals)
        ok_pct     = float((diffs < TOLERANCE).mean() * 100)
        if ok_pct < PASS_THRESHOLD:
            all_ok = False
        print(f"  {var:<5}  {diffs.max():>10.6f}  {diffs.mean():>10.6f}  {ok_pct:>13.1f}%  {gtiff_path.name:>12}")

    print()
    if all_ok:
        print("VERIFICACIÓN CORRECTA: TileDB coincide con GeoTIFF en todas las variables.")
    else:
        print(f"ADVERTENCIA: se detectaron diferencias > {TOLERANCE} en alguna variable.")

    # Verificar umbral: ninguna celda almacenada debe estar por debajo del umbral
    threshold = float(meta.get("depth_threshold_m", 0.01))
    h_min_tdb = float(sample["H"].min())
    ok_thr    = h_min_tdb >= threshold
    print(f"\nUmbral H >= {threshold} m: H mín en muestra = {h_min_tdb:.6f} m"
          f"  {'OK' if ok_thr else 'FALLO: celda por debajo del umbral'}")


if __name__ == "__main__":
    main()
