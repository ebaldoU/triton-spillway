#!/usr/bin/env python3
# ============================================================
# visualize_tiledb.py
# Visualiza resultados del array TileDB sparse de Triton.
# Lee directamente de TileDB sin tocar los GeoTIFF originales.
#
# Uso:
#   python visualize_tiledb.py --dataset datos1
#   python visualize_tiledb.py --dataset datos2 01_00
#   python visualize_tiledb.py --dataset datos1 01_00 H
#   python visualize_tiledb.py --dataset datos2 01_00 MH
#   python visualize_tiledb.py --dataset datos1 01_00 full
# ============================================================
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tiledb
from config import resolve_dataset


DEFAULT_STEP  = "01_00"
DEFAULT_MODE  = "full"
ARROW_STEP    = 40      # 1 flecha cada 40 px display
ARROW_MIN_H   = 0.05    # solo flechas donde H > 5 cm
WATER_THRESH  = 1e-6
MAX_DISPLAY_PX = 4096

LABELS = {
    "H":  "Profundidad del agua (m)",
    "QX": "Caudal unitario X (m²/s)",
    "QY": "Caudal unitario Y (m²/s)",
    "MH": "Profundidad máxima acumulada (m)",
}
CMAPS = {"H": "viridis", "QX": "coolwarm", "QY": "coolwarm", "MH": "hot_r"}
TEXTS = {
    "H":  "Los colores indican la profundidad del agua en el paso actual.",
    "QX": "Los valores positivos/negativos representan la dirección del caudal en X.",
    "QY": "Los valores positivos/negativos representan la dirección del caudal en Y.",
    "MH": "Profundidad máxima registrada en cada celda desde el inicio de la simulación.",
}


# ── Lectura ───────────────────────────────────────────────────

def load_metadata(uri: str) -> dict:
    with tiledb.open(uri, mode="r") as A:
        meta = dict(A.meta)
    return {
        "width":      int(meta["width"]),
        "height":     int(meta["height"]),
        "crs":        meta.get("crs", ""),
        "transform":  json.loads(meta["transform"]),
        "bounds":     json.loads(meta["bounds"]),
        "time_steps": json.loads(meta["time_steps"]),
    }


def step_to_index(step: str, time_steps: list[str]) -> int:
    if step not in time_steps:
        raise ValueError(f"Paso '{step}' no encontrado. Disponibles: {time_steps}")
    return time_steps.index(step)


def query_timestep(uri: str, t_index: int, width: int, height: int,
                   attrs: list[str]) -> dict[str, np.ndarray]:
    """Mapea sparse → resolución de display sin construir el array denso completo."""
    step_r = max(1, height // MAX_DISPLAY_PX)
    step_c = max(1, width  // MAX_DISPLAY_PX)
    disp_h = (height + step_r - 1) // step_r
    disp_w = (width  + step_c - 1) // step_c

    with tiledb.open(uri, mode="r") as A:
        result = A.query(attrs=attrs)[t_index, :, :]

    rows = result["row"]
    cols = result["col"]
    disp_rows = rows // step_r
    disp_cols = cols // step_c

    out = {}
    for attr in attrs:
        arr = np.zeros((disp_h, disp_w), dtype=np.float32)
        if rows.size > 0:
            arr[disp_rows, disp_cols] = result[attr]
        out[attr] = arr
    return out


# ── Coordenadas ───────────────────────────────────────────────

def make_extent(bounds: list) -> list:
    return [bounds[0], bounds[2], bounds[1], bounds[3]]


CURSOR_LABELS = {"H": "Profundidad", "QX": "Caudal X", "QY": "Caudal Y", "MH": "Prof. máxima"}
CURSOR_UNITS  = {"H": "m", "QX": "m²/s", "QY": "m²/s", "MH": "m"}


def make_cursor(fig, ax, bounds: list, arrays: dict[str, np.ndarray]) -> None:
    left, bottom, right, top = bounds
    h_ds, w_ds = next(iter(arrays.values())).shape

    info = fig.text(
        0.14, 0.5, "Mueve el cursor\nsobre el mapa",
        va="center", ha="left", fontsize=12, family="monospace",
        animated=True,
        bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="#555", linewidth=1.5, alpha=0.95),
    )
    background = [None]

    def on_draw(event):
        background[0] = fig.canvas.copy_from_bbox(fig.bbox)
        fig.draw_artist(info)
        fig.canvas.blit(fig.bbox)

    def on_move(event):
        if background[0] is None or event.inaxes is not ax:
            return
        x, y = event.xdata, event.ydata
        col = int((x - left) / (right - left) * w_ds)
        row = int((top  - y) / (top - bottom) * h_ds)
        if 0 <= row < h_ds and 0 <= col < w_ds:
            lines = [f"X: {x:>12,.0f} m", f"Y: {y:>12,.0f} m", ""]
            for name, arr in arrays.items():
                lines.append(f"{CURSOR_LABELS.get(name, name)}:\n  {arr[row, col]:>8.3f} {CURSOR_UNITS.get(name, '')}")
            info.set_text("\n".join(lines))
        else:
            info.set_text("Fuera del\ndominio")
        fig.canvas.restore_region(background[0])
        fig.draw_artist(info)
        fig.canvas.blit(fig.bbox)

    fig.canvas.mpl_connect("draw_event", on_draw)
    fig.canvas.mpl_connect("motion_notify_event", on_move)


# ── Modos de visualización ────────────────────────────────────

def plot_full(uri: str, meta: dict, step: str, t_index: int) -> None:
    """H como fondo de color + flechas de flujo QX/QY."""
    t0 = time.perf_counter()
    data = query_timestep(uri, t_index, meta["width"], meta["height"], ["H", "QX", "QY"])
    t_query = time.perf_counter() - t0

    H, QX, QY = data["H"], data["QX"], data["QY"]
    water = np.where(H > WATER_THRESH, H, np.nan)
    h_ds, w_ds = H.shape
    bounds = meta["bounds"]

    xs = np.linspace(bounds[0], bounds[2], w_ds)[::ARROW_STEP]
    ys = np.linspace(bounds[3], bounds[1], h_ds)[::ARROW_STEP]
    Xs, Ys = np.meshgrid(xs, ys)
    QXs = np.where(H[::ARROW_STEP, ::ARROW_STEP] > ARROW_MIN_H,
                   QX[::ARROW_STEP, ::ARROW_STEP], np.nan)
    QYs = np.where(H[::ARROW_STEP, ::ARROW_STEP] > ARROW_MIN_H,
                   QY[::ARROW_STEP, ::ARROW_STEP], np.nan)

    t1 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.93, bottom=0.07)

    h_max = float(np.nanmax(water))
    norm  = mcolors.LogNorm(vmin=ARROW_MIN_H, vmax=h_max) if h_max > ARROW_MIN_H else None
    img   = ax.imshow(water, extent=make_extent(bounds), origin="upper", aspect="equal",
                      cmap="viridis", norm=norm)
    fig.colorbar(img, ax=ax).set_label("Profundidad (m) — escala log")
    ax.quiver(Xs, Ys, QXs, QYs, angles="xy", scale_units="xy", scale=None, width=0.002, alpha=0.6)
    ax.set_title(f"Simulación completa — paso {step}\nProfundidad (color log) + flujo QX/QY (flechas)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    make_cursor(fig, ax, bounds, {"H": H, "QX": QX, "QY": QY})
    t_render = time.perf_counter() - t1

    _print_stats(uri, meta, step, t_index, t_query, t_render, H=H, QX=QX, QY=QY)
    out = Path(f"plot_{step}_full.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Imagen guardada: {out.resolve()}")
    plt.show()


def plot_variable(uri: str, meta: dict, step: str, t_index: int, var: str) -> None:
    """Una sola variable en color."""
    t0 = time.perf_counter()
    data = query_timestep(uri, t_index, meta["width"], meta["height"], [var])
    t_query = time.perf_counter() - t0

    arr = data[var]
    t1 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.93, bottom=0.07)

    if var in ("H", "MH"):
        v_max = float(np.nanmax(arr))
        norm  = mcolors.LogNorm(vmin=ARROW_MIN_H, vmax=v_max) if v_max > ARROW_MIN_H else None
        label = LABELS[var] + " — escala log"
    else:
        norm  = None
        label = LABELS[var]

    img = ax.imshow(arr, extent=make_extent(meta["bounds"]), origin="upper", aspect="equal",
                    cmap=CMAPS[var], norm=norm)
    fig.colorbar(img, ax=ax).set_label(label)
    ax.set_title(f"{LABELS[var]}\nPaso: {step}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    make_cursor(fig, ax, meta["bounds"], {var: arr})
    t_render = time.perf_counter() - t1

    _print_stats(uri, meta, step, t_index, t_query, t_render, **{var: arr})
    out = Path(f"plot_{step}_{var}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Imagen guardada: {out.resolve()}")
    plt.show()


def _print_stats(uri, meta, step, t_index, t_query, t_render, **arrays) -> None:
    print(f"Fuente:         TileDB sparse → {uri}")
    print(f"Paso temporal:  {step} (índice {t_index})")
    print(f"CRS:            {meta['crs']}")
    print(f"Tamaño raster:  {meta['width']} x {meta['height']}")
    print(f"Tiempo consulta TileDB:  {t_query:.3f}s")
    print(f"Tiempo renderizado:      {t_render:.3f}s")
    for name, arr in arrays.items():
        valid = arr[np.isfinite(arr) & (arr != 0)]
        if valid.size > 0:
            print(f"{name:<4}  min={float(np.nanmin(arr)):.4f}  max={float(np.nanmax(arr)):.4f}  media={float(np.nanmean(valid)):.4f}")


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizador TileDB/Triton")
    parser.add_argument("--dataset", required=True,
                        help="Dataset a visualizar")
    parser.add_argument("step", nargs="?", default=DEFAULT_STEP,
                        help="Paso temporal, ej. 01_00 (defecto: 01_00)")
    parser.add_argument("mode", nargs="?", default=DEFAULT_MODE,
                        help="Modo: full | H | QX | QY | MH (defecto: full)")
    args = parser.parse_args()

    uri  = f"/home/ebald/TFG/tiledb/triton_results/{resolve_dataset(args.dataset)}"
    step = args.step
    mode = args.mode.upper()

    meta    = load_metadata(uri)
    t_index = step_to_index(step, meta["time_steps"])

    if mode == "FULL":
        plot_full(uri, meta, step, t_index)
    elif mode in ("H", "QX", "QY", "MH"):
        plot_variable(uri, meta, step, t_index, mode)
    else:
        print(f"Modo '{mode}' no reconocido. Usa: full | H | QX | QY | MH")
        sys.exit(1)


if __name__ == "__main__":
    main()
