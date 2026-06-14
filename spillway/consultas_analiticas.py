#!/usr/bin/env python3
"""
consultas_analiticas.py — Motor de consultas analíticas sobre el array TileDB de Triton.

Parámetros geoespaciales:
  cellsize   = 10 m  →  área de celda = 100 m²
  xllcorner  = 680 000 m   (EPSG:32614)
  yllcorner  = 4 330 000 m
  nrows      = 34 200
  ncols      = 23 040
  paso temp. = 6 h  (t=0 → hora 6, t=19 → hora 120)

Fórmulas de peligrosidad: Russo et al. (2013)
  Q_mod = sqrt(QX² + QY²)   [módulo caudal unitario, m²/s]
  Adultos inestables    : H > 0.50 m  ó  Q_mod > 0.50 m²/s
  Niños inestables      : H > 0.25 m  ó  Q_mod > 0.15 m²/s
  Vehículos ligeros     : H > 0.30 m
  Vehículos emergencia  : H < 0.60 m

Uso:
  python consultas_analiticas.py --dataset datos1
  python consultas_analiticas.py --dataset datos2
  TRITON_DATASET=datos2 python -c "import consultas_analiticas as ca; ..."
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tiledb
from config import BASE_URI, resolve_dataset

# ── Constantes del dominio ────────────────────────────────────
_DATASET    = os.environ.get("TRITON_DATASET", "datos1")
TILEDB_URI  = f"{BASE_URI}/{resolve_dataset(_DATASET)}"
CELLSIZE    = 10.0
CELL_AREA   = CELLSIZE**2
XLLCORNER   = 680_000.0
YLLCORNER   = 4_330_000.0
NROWS       = 34_200
NCOLS       = 23_040
STEP_H      = 6
N_STEPS     = 20
H_WET       = 0.01

# ── Umbrales Russo et al. (2013) ──────────────────────────────
HV_ADULTO    = 0.50
H_ADULTO     = 0.50
HV_NINO      = 0.15
H_NINO       = 0.25
H_VEHICULO   = 0.30
H_EMERGENCIA = 0.60

# ── Criterio Xia et al. (2014) — Inestabilidad de personas ────
# Parámetros: adulto medio español [20] y niño de 8 años [21]
XIA_ADULTO_HP  = 1.68      # estatura adulto (m)
XIA_ADULTO_MP  = 73.76     # masa adulto (kg)
XIA_NINO_HP    = 1.26      # estatura niño (m)
XIA_NINO_MP    = 25.64     # masa niño (kg)
# Coeficientes experimentales [19]
XIA_A1 = 0.633; XIA_B1 = 0.367; XIA_A2 = 1.015e-3; XIA_B2 = -4.927e-3
XIA_RHO_F      = 1000.0    # densidad del agua (kg/m³)
XIA_ALPHA_LOW  = 3.472;    XIA_BETA_LOW  = 0.188   # curva naranja (inicio zona moderada)
XIA_ALPHA_HIGH = 7.867;    XIA_BETA_HIGH = 0.462   # curva roja (inicio zona alta)

# ── Criterio Xia et al. (2022) — Inestabilidad de vehículos ───
# Parámetros para Mini Cooper [22]
XIA_VEH_HC    = 1.417      # altura del vehículo (m)
XIA_VEH_RHO_C = 1009.0     # densidad del vehículo (kg/m³)
XIA_VEH_A_PAR = 1.225;     XIA_VEH_B_PAR = -0.708   # parcialmente sumergido (H < H_c)
XIA_VEH_A_TOT = 0.932;     XIA_VEH_B_TOT =  0.121   # totalmente sumergido (H ≥ H_c)

# ── Graves daños — Real Decreto 9/2008 ────────────────────────
GD_H  = 1.0    # umbral de calado (m)
GD_V  = 1.0    # umbral de velocidad (m/s)
GD_HV = 0.5    # umbral de producto H·V = Q_mod (m²/s)


# ── Conversiones auxiliares ───────────────────────────────────

def hora_a_t(hora: int) -> int:
    """Hora de simulación (6-120) → índice de paso (0-19)."""
    if hora % STEP_H != 0:
        raise ValueError(f"Hora {hora} no es múltiplo de {STEP_H} h.")
    t = hora // STEP_H - 1
    if not (0 <= t < N_STEPS):
        raise ValueError(f"Hora {hora} fuera de rango [6, 120] en múltiplos de 6.")
    return t


def t_a_hora(t: int) -> int:
    return (t + 1) * STEP_H


def coord_a_pixel(x: float, y: float) -> tuple[int, int]:
    col = int((x - XLLCORNER) / CELLSIZE)
    row = int((YLLCORNER + NROWS * CELLSIZE - y) / CELLSIZE)
    if not (0 <= row < NROWS and 0 <= col < NCOLS):
        raise ValueError(f"Coordenada ({x}, {y}) fuera del dominio.")
    return row, col


def pixel_a_coord(row: int, col: int) -> tuple[float, float]:
    x = XLLCORNER + (col + 0.5) * CELLSIZE
    y = YLLCORNER + (NROWS - row - 0.5) * CELLSIZE
    return x, y


def bbox_a_indices(x_min, y_min, x_max, y_max) -> tuple[int, int, int, int]:
    c_min = max(0, int((x_min - XLLCORNER) / CELLSIZE))
    c_max = min(NCOLS - 1, int((x_max - XLLCORNER) / CELLSIZE))
    r_min = max(0, int((YLLCORNER + NROWS * CELLSIZE - y_max) / CELLSIZE))
    r_max = min(NROWS - 1, int((YLLCORNER + NROWS * CELLSIZE - y_min) / CELLSIZE))
    return r_min, r_max, c_min, c_max


def _query_paso(A, t: int, r_min=0, r_max=NROWS-1, c_min=0, c_max=NCOLS-1,
                attrs=("H",), cond=None) -> dict:
    return A.query(attrs=list(attrs), cond=cond)[t, r_min:r_max+1, c_min:c_max+1]


def _cond_umbral(umbral_m: float):
    """Filtro pushdown TileDB: solo si el umbral supera el del ETL (H_WET)."""
    return f"H >= {umbral_m}" if umbral_m > H_WET else None


def _bbox_args(bbox):
    if bbox is None:
        return 0, NROWS - 1, 0, NCOLS - 1
    return bbox_a_indices(*bbox)


def load_domain_constants(uri: str) -> None:
    """Lee NROWS, NCOLS, CELLSIZE, XLLCORNER, YLLCORNER desde los metadatos TileDB.
    Llamar al inicio de main() para evitar hardcodeo de constantes del dominio.
    """
    global NROWS, NCOLS, CELLSIZE, CELL_AREA, XLLCORNER, YLLCORNER
    with tiledb.open(uri, mode="r") as A:
        meta = dict(A.meta)
    tr = json.loads(meta["transform"])   # formato rasterio: (a, b, c, d, e, f)
    NCOLS      = int(meta["width"])
    NROWS      = int(meta["height"])
    CELLSIZE   = abs(float(tr[0]))
    CELL_AREA  = CELLSIZE**2
    XLLCORNER  = float(tr[2])
    YLLCORNER  = float(tr[5]) + float(tr[4]) * NROWS   # esquina inferior izquierda


# ═══════════════════════════════════════════════════════════════
# BLOQUE A — Consultas puntuales y de velocidad
# ═══════════════════════════════════════════════════════════════

def calado_en_punto(x: float, y: float, hora: int) -> float:
    """Q1 — Calado H (m) en la celda (x, y) a la hora dada. 0.0 si seca."""
    row, col = coord_a_pixel(x, y)
    t = hora_a_t(hora)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        res = A.query(attrs=["H"])[t, row, col]
    return float(res["H"][0]) if len(res["H"]) > 0 else 0.0


def serie_temporal_punto(x: float, y: float) -> dict:
    """Q2 — Serie temporal de H en el punto (x, y) a lo largo de las 120 h."""
    row, col = coord_a_pixel(x, y)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        res = A.query(attrs=["H"])[0:N_STEPS, row, col]
    h_by_t = {int(t): float(h) for t, h in zip(res["time"], res["H"])}
    return {
        "horas":  [t_a_hora(t) for t in range(N_STEPS)],
        "H":      [h_by_t.get(t, 0.0) for t in range(N_STEPS)],
    }


def velocidad_en_zona(x_min, y_min, x_max, y_max, hora: int) -> dict:
    """Q3 — Velocidad V = Q_mod/H en celdas húmedas de la zona a la hora dada."""
    t = hora_a_t(hora)
    r_min, r_max, c_min, c_max = bbox_a_indices(x_min, y_min, x_max, y_max)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        res = _query_paso(A, t, r_min, r_max, c_min, c_max, attrs=("H", "QX", "QY"))
    H = res["H"]
    # TileDB sparse solo devuelve celdas con H >= H_WET, así que la división es segura.
    V = np.sqrt(res["QX"]**2 + res["QY"]**2) / H if len(H) > 0 else np.array([], np.float32)
    return {
        "rows": res["row"], "cols": res["col"], "H": H, "V": V, "hora": hora,
        "V_max":   float(V.max())  if len(V) > 0 else 0.0,
        "V_media": float(V.mean()) if len(V) > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════
# BLOQUE B — Umbrales espaciales
# ═══════════════════════════════════════════════════════════════

def zonas_inundadas(hora: int, umbral_m: float = H_WET, bbox=None) -> dict:
    """Q4 — Celdas con H >= umbral_m en la hora dada."""
    t = hora_a_t(hora)
    r_min, r_max, c_min, c_max = _bbox_args(bbox)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        res = _query_paso(A, t, r_min, r_max, c_min, c_max, attrs=("H",),
                          cond=_cond_umbral(umbral_m))
    mask = res["H"] >= umbral_m
    n = int(mask.sum())
    return {
        "rows": res["row"][mask], "cols": res["col"][mask], "H": res["H"][mask],
        "n_celdas": n, "area_m2": n * CELL_AREA, "area_km2": n * CELL_AREA / 1e6,
        "hora": hora, "umbral_m": umbral_m,
    }



def evolucion_extension(umbral_m: float = H_WET, bbox=None) -> dict:
    """Q5 — Área inundada por hora a lo largo de las 120 h."""
    r_min, r_max, c_min, c_max = _bbox_args(bbox)
    horas, n_celdas, areas_km2 = [], [], []
    with tiledb.open(TILEDB_URI, mode="r") as A:
        for t in range(N_STEPS):
            res = A.query(attrs=["H"])[t, r_min:r_max+1, c_min:c_max+1]
            n = int((res["H"] >= umbral_m).sum())
            horas.append(t_a_hora(t))
            n_celdas.append(n)
            areas_km2.append(n * CELL_AREA / 1e6)
    return {"horas": horas, "n_celdas": n_celdas, "area_km2": areas_km2}


# ═══════════════════════════════════════════════════════════════
# BLOQUE C — Indicadores temporales por píxel
# ═══════════════════════════════════════════════════════════════

def hora_llegada_frente(bbox=None) -> dict:
    """
    Q7 — Primera hora en que H > 0.01 m en cada píxel.
    Grid acumulador uint8: memoria O(nr×nc) constante independiente del nº de celdas húmedas.
    """
    r_min, r_max, c_min, c_max = _bbox_args(bbox)
    nr, nc = r_max - r_min + 1, c_max - c_min + 1
    arrival = np.full((nr, nc), N_STEPS, dtype=np.uint8)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        for t in range(N_STEPS):
            res  = A.query(attrs=["H"])[t, r_min:r_max+1, c_min:c_max+1]
            mask = res["H"] >= H_WET
            if not mask.any():
                continue
            r = (res["row"][mask] - r_min).astype(np.intp)
            c = (res["col"][mask] - c_min).astype(np.intp)
            np.minimum.at(arrival, (r, c), np.uint8(t))
    valid = arrival < N_STEPS
    wr, wc = np.where(valid)
    return {
        "rows":         (wr + r_min).astype(np.int32),
        "cols":         (wc + c_min).astype(np.int32),
        "hora_llegada": ((arrival[valid].astype(np.int32) + 1) * STEP_H).astype(np.int32),
    }


def duracion_inundacion(umbral_m: float = H_WET, bbox=None) -> dict:
    """
    Q8 — Horas totales que cada píxel permanece con H > umbral_m.
    Grid acumulador uint8: cuenta pasos húmedos sin concatenar arrays.
    """
    r_min, r_max, c_min, c_max = _bbox_args(bbox)
    nr, nc = r_max - r_min + 1, c_max - c_min + 1
    duration = np.zeros((nr, nc), dtype=np.uint8)
    cond = _cond_umbral(umbral_m)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        for t in range(N_STEPS):
            res  = A.query(attrs=["H"], cond=cond)[t, r_min:r_max+1, c_min:c_max+1]
            mask = res["H"] >= umbral_m
            if not mask.any():
                continue
            r = (res["row"][mask] - r_min).astype(np.intp)
            c = (res["col"][mask] - c_min).astype(np.intp)
            np.add.at(duration, (r, c), 1)
    valid = duration > 0
    wr, wc = np.where(valid)
    return {
        "rows":           (wr + r_min).astype(np.int32),
        "cols":           (wc + c_min).astype(np.int32),
        "horas_inundado": (duration[valid].astype(np.int32) * STEP_H).astype(np.int32),
    }


def hora_calado_maximo(bbox=None) -> dict:
    """
    Q9 — Hora en que cada píxel alcanza su calado máximo.
    Grid acumulador: float32 para max_h + uint8 para t_peak. Sin concatenación.
    """
    r_min, r_max, c_min, c_max = _bbox_args(bbox)
    nr, nc = r_max - r_min + 1, c_max - c_min + 1
    max_h  = np.zeros((nr, nc), dtype=np.float32)
    t_peak = np.zeros((nr, nc), dtype=np.uint8)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        for t in range(N_STEPS):
            res = A.query(attrs=["H"])[t, r_min:r_max+1, c_min:c_max+1]
            h   = res["H"]
            if not len(h):
                continue
            r = (res["row"] - r_min).astype(np.intp)
            c = (res["col"] - c_min).astype(np.intp)
            better = h > max_h[r, c]
            if not better.any():
                continue
            rb, cb = r[better], c[better]
            max_h[rb, cb]  = h[better]
            t_peak[rb, cb] = np.uint8(t)
    valid = max_h >= H_WET
    wr, wc = np.where(valid)
    return {
        "rows":     (wr + r_min).astype(np.int32),
        "cols":     (wc + c_min).astype(np.int32),
        "hora_pico": ((t_peak[valid].astype(np.int32) + 1) * STEP_H).astype(np.int32),
        "MH":       max_h[valid],
    }


def ventana_vehiculos_emergencia(bbox=None) -> dict:
    """
    Q11 — Intervalo (hora_inicio, hora_fin) en que H < 0.60 m por píxel.
    Grid acumulador: dos grids uint8 para min_t y max_t. Sin concatenación.
    """
    r_min, r_max, c_min, c_max = _bbox_args(bbox)
    nr, nc = r_max - r_min + 1, c_max - c_min + 1
    min_t = np.full((nr, nc), N_STEPS, dtype=np.uint8)
    max_t = np.zeros((nr, nc), dtype=np.uint8)
    seen  = np.zeros((nr, nc), dtype=bool)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        for t in range(N_STEPS):
            res  = A.query(attrs=["H"])[t, r_min:r_max+1, c_min:c_max+1]
            mask = (res["H"] >= H_WET) & (res["H"] < H_EMERGENCIA)
            if not mask.any():
                continue
            r = (res["row"][mask] - r_min).astype(np.intp)
            c = (res["col"][mask] - c_min).astype(np.intp)
            np.minimum.at(min_t, (r, c), np.uint8(t))
            np.maximum.at(max_t, (r, c), np.uint8(t))
            seen[r, c] = True
    valid = seen
    wr, wc = np.where(valid)
    hora_inicio = ((min_t[valid].astype(np.int32) + 1) * STEP_H).astype(np.int32)
    hora_fin    = ((max_t[valid].astype(np.int32) + 1) * STEP_H).astype(np.int32)
    return {
        "rows":             (wr + r_min).astype(np.int32),
        "cols":             (wc + c_min).astype(np.int32),
        "hora_inicio":      hora_inicio,
        "hora_fin":         hora_fin,
        "horas_practicable": (hora_fin - hora_inicio + STEP_H).astype(np.int32),
    }


# ═══════════════════════════════════════════════════════════════
# BLOQUE D — Peligrosidad (Russo et al., 2013)
# ═══════════════════════════════════════════════════════════════

def _peligrosidad_base(hora: int, bbox=None, attrs=("H", "QX", "QY")) -> tuple[np.ndarray, ...]:
    t = hora_a_t(hora)
    r_min, r_max, c_min, c_max = _bbox_args(bbox)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        res = A.query(attrs=list(attrs))[t, r_min:r_max+1, c_min:c_max+1]
    Q_mod = np.sqrt(res["QX"]**2 + res["QY"]**2) if "QX" in res else None
    return res["row"], res["col"], res["H"], Q_mod


def _peligrosidad_result(rows, cols, H, Q_mod, mask, hora: int) -> dict:
    n = int(mask.sum())
    base = {"rows": rows[mask], "cols": cols[mask], "H": H[mask],
            "n_celdas": n, "area_km2": n * CELL_AREA / 1e6, "hora": hora}
    if Q_mod is not None:
        base["Q_mod"] = Q_mod[mask]
    return base


def peligrosidad_adultos(hora: int, bbox=None) -> dict:
    """Q10a — Celdas peligrosas para adultos: H > 0.50 m ó Q_mod > 0.50 m²/s."""
    rows, cols, H, Q_mod = _peligrosidad_base(hora, bbox)
    return _peligrosidad_result(rows, cols, H, Q_mod, (H > H_ADULTO) | (Q_mod > HV_ADULTO), hora)


def peligrosidad_ninos(hora: int, bbox=None) -> dict:
    """Q10b — Celdas peligrosas para niños: H > 0.25 m ó Q_mod > 0.15 m²/s."""
    rows, cols, H, Q_mod = _peligrosidad_base(hora, bbox)
    return _peligrosidad_result(rows, cols, H, Q_mod, (H > H_NINO) | (Q_mod > HV_NINO), hora)


def peligro_vehiculos_ligeros(hora: int, bbox=None) -> dict:
    """Q10c — Celdas peligrosas para vehículos ligeros: H > 0.30 m."""
    rows, cols, H, _ = _peligrosidad_base(hora, bbox, attrs=("H",))
    return _peligrosidad_result(rows, cols, H, None, H > H_VEHICULO, hora)


def transitabilidad_emergencia(hora: int, bbox=None) -> dict:
    """Q11b — Celdas practicables para vehículos de emergencia: H < 0.60 m."""
    rows, cols, H, _ = _peligrosidad_base(hora, bbox, attrs=("H",))
    return _peligrosidad_result(rows, cols, H, None, (H > 0) & (H < H_EMERGENCIA), hora)


# ═══════════════════════════════════════════════════════════════
# BLOQUE E — Estadísticos espaciales
# ═══════════════════════════════════════════════════════════════

def area_inundada_por_hora(umbral_m: float = H_WET, bbox=None) -> dict:
    """Q12 — Área inundada por hora. Alias de evolucion_extension."""
    return evolucion_extension(umbral_m=umbral_m, bbox=bbox)


def volumen_por_hora(bbox=None) -> dict:
    """Q13 — Volumen de agua (m³) por hora: sum(H) × 100 m²."""
    r_min, r_max, c_min, c_max = _bbox_args(bbox)
    horas, volumenes = [], []
    with tiledb.open(TILEDB_URI, mode="r") as A:
        for t in range(N_STEPS):
            res = A.query(attrs=["H"])[t, r_min:r_max+1, c_min:c_max+1]
            horas.append(t_a_hora(t))
            volumenes.append(float(res["H"][res["H"] >= H_WET].sum()) * CELL_AREA)
    return {"horas": horas, "volumen_m3": volumenes}


def stats_zona(x_min, y_min, x_max, y_max, hora: int) -> dict:
    """Q14 — Estadísticos de H en zona y hora: media, máx, mín, std, P25/50/75/95."""
    t = hora_a_t(hora)
    r_min, r_max, c_min, c_max = bbox_a_indices(x_min, y_min, x_max, y_max)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        res = A.query(attrs=["H"])[t, r_min:r_max+1, c_min:c_max+1]
    wet = res["H"][res["H"] >= H_WET]
    if len(wet) == 0:
        return {"hora": hora, "n_celdas": 0, "nota": "zona seca en este instante"}
    return {
        "hora": hora, "n_celdas": len(wet), "area_km2": len(wet) * CELL_AREA / 1e6,
        "H_media": float(wet.mean()), "H_max": float(wet.max()),
        "H_min": float(wet.min()),    "H_std": float(wet.std()),
        **dict(zip(["H_P25", "H_P50", "H_P75", "H_P95"],
                   [float(v) for v in np.percentile(wet, [25, 50, 75, 95])])),
    }


def tiempo_evacuacion(umbral_critico_m: float = H_ADULTO,
                      umbral_qmod: float = HV_ADULTO,
                      bbox=None) -> dict:
    """
    Q16 — Ventana de evacuación por píxel (horas).

    Para cada píxel calcula:
      t_llegada  : primera hora en que H >= H_WET (el agua llega)
      t_critico  : primera hora en que H > umbral_critico_m ó Q_mod > umbral_qmod
                   (condiciones peligrosas para el umbral seleccionado)
      ventana_h  : (t_critico - t_llegada) × 6 h
                   → horas disponibles para evacuar antes de que el calado sea peligroso
                   → 0  si el agua llega ya peligrosa
                   → NaN (representado como -1) si nunca supera el umbral crítico

    Vectorizado: groupby-min con np.minimum.at sobre clave int64 row×NCOLS+col.
    """
    r_min, r_max, c_min, c_max = _bbox_args(bbox)

    nr, nc = r_max - r_min + 1, c_max - c_min + 1
    t_llegada = np.full((nr, nc), N_STEPS, dtype=np.uint8)
    t_critico = np.full((nr, nc), N_STEPS, dtype=np.uint8)

    with tiledb.open(TILEDB_URI, mode="r") as A:
        for t in range(N_STEPS):
            res   = A.query(attrs=["H", "QX", "QY"])[t, r_min:r_max+1, c_min:c_max+1]
            H     = res["H"]
            Q_mod = np.sqrt(res["QX"]**2 + res["QY"]**2)
            r     = (res["row"] - r_min).astype(np.intp)
            c     = (res["col"] - c_min).astype(np.intp)

            mask_wet  = H >= H_WET
            mask_crit = (H > umbral_critico_m) | (Q_mod > umbral_qmod)

            if mask_wet.any():
                np.minimum.at(t_llegada, (r[mask_wet], c[mask_wet]), np.uint8(t))
            if mask_crit.any():
                np.minimum.at(t_critico, (r[mask_crit], c[mask_crit]), np.uint8(t))

    wet_mask = t_llegada < N_STEPS
    if not wet_mask.any():
        return {"rows": np.array([], np.int32), "cols": np.array([], np.int32),
                "ventana_h": np.array([], np.float32)}

    wr, wc = np.where(wet_mask)
    tl = t_llegada[wet_mask].astype(np.int32)
    tc = t_critico[wet_mask].astype(np.int32)

    # ventana = (t_critico - t_llegada) × STEP_H, mínimo 0
    # t_critico == N_STEPS → nunca llega a ser peligroso → ventana = -1 (sin límite)
    ventana_pasos = tc - tl
    ventana_h     = np.where(tc == N_STEPS,
                             -1,
                             np.maximum(0, ventana_pasos) * STEP_H).astype(np.float32)

    rows_out = (wr + r_min).astype(np.int32)
    cols_out = (wc + c_min).astype(np.int32)
    nunca_peligroso = int((tc == N_STEPS).sum())
    ya_peligroso    = int((ventana_h == 0).sum())

    return {
        "rows":             rows_out,
        "cols":             cols_out,
        "ventana_h":        ventana_h,
        "t_llegada_h":      ((tl + 1) * STEP_H).astype(np.int32),
        "t_critico_h":      np.where(tc == N_STEPS, -1,
                                     (tc + 1) * STEP_H).astype(np.int32),
        "n_pixeles":        len(rows_out),
        "nunca_peligroso":  nunca_peligroso,
        "ya_peligroso":     ya_peligroso,
        "ventana_media_h":  float(ventana_h[ventana_h >= 0].mean()) if (ventana_h >= 0).any() else 0.0,
        "ventana_min_h":    float(ventana_h[ventana_h >= 0].min())  if (ventana_h >= 0).any() else 0.0,
        "umbral_critico_m": umbral_critico_m,
        "umbral_qmod":      umbral_qmod,
    }


def xia_ucrit_personas(H: np.ndarray, tipo: str = 'adultos'):
    """Velocidades críticas de inestabilidad para personas (Xia et al., 2014).
    Devuelve (ucrit_low, ucrit_high) para la curva naranja y la roja."""
    h_p = XIA_ADULTO_HP if tipo == 'adultos' else XIA_NINO_HP
    m_p = XIA_ADULTO_MP if tipo == 'adultos' else XIA_NINO_MP
    H_s = np.maximum(H, 0.001)
    def _u(alpha, beta):
        inner = (m_p / (XIA_RHO_F * H_s**2)
                 - (XIA_A1/h_p**2 + XIA_B1/(H_s * h_p)) * (XIA_A2 * m_p + XIA_B2))
        return alpha * (H_s / h_p)**beta * np.sqrt(np.maximum(inner, 0.0))
    return _u(XIA_ALPHA_LOW, XIA_BETA_LOW), _u(XIA_ALPHA_HIGH, XIA_BETA_HIGH)


def xia_risk_personas(H: np.ndarray, Q_mod: np.ndarray, tipo: str = 'adultos') -> np.ndarray:
    """Nivel de inestabilidad para personas (Xia et al., 2014).
    0 = seguro | 1 = riesgo moderado (V ≥ curva naranja) | 2 = riesgo alto (V ≥ curva roja)."""
    V = Q_mod / np.maximum(H, 0.001)
    uc_low, uc_high = xia_ucrit_personas(H, tipo)
    risk = np.zeros(len(H), dtype=np.uint8)
    risk[V >= uc_low]  = 1
    risk[V >= uc_high] = 2
    return risk


def xia_ucrit_vehiculos(H: np.ndarray):
    """Velocidades críticas de arrastre para vehículos (Xia et al., 2022).
    Devuelve (ucrit_low=0.5×ucrit, ucrit_high=ucrit) para curva naranja y roja."""
    import math
    g = 9.81
    buoyancy = math.sqrt(
        max((XIA_VEH_RHO_C - XIA_RHO_F) / XIA_RHO_F * 2 * g * XIA_VEH_HC, 0.0))
    H_s = np.maximum(H, 0.001)
    partial = H_s < XIA_VEH_HC
    alpha = np.where(partial, XIA_VEH_A_PAR, XIA_VEH_A_TOT)
    beta  = np.where(partial, XIA_VEH_B_PAR, XIA_VEH_B_TOT)
    ucrit = alpha * (H_s / XIA_VEH_HC)**beta * buoyancy
    return ucrit * 0.5, ucrit


def xia_risk_vehiculos(H: np.ndarray, Q_mod: np.ndarray) -> np.ndarray:
    """Nivel de riesgo de arrastre para vehículos (Xia et al., 2022).
    0 = seguro | 1 = riesgo moderado | 2 = riesgo alto."""
    V = Q_mod / np.maximum(H, 0.001)
    uc_low, uc_high = xia_ucrit_vehiculos(H)
    risk = np.zeros(len(H), dtype=np.uint8)
    risk[V >= uc_low]  = 1
    risk[V >= uc_high] = 2
    return risk


def graves_danos_mask(H: np.ndarray, Q_mod: np.ndarray) -> np.ndarray:
    """Máscara de zona de graves daños según Real Decreto 9/2008.
    Criterio: H > 1 m  OR  V > 1 m/s  OR  H·V > 0,5 m²/s."""
    V = Q_mod / np.maximum(H, 0.001)
    return (H > GD_H) | (V > GD_V) | (Q_mod > GD_HV)


def russo_traffic_light_counts(H: np.ndarray) -> tuple[int, int, int]:
    """Cuenta celdas en niveles Russo: (verde H≤0.25, amarillo 0.25<H≤0.50, rojo H>0.50)."""
    verde    = int(((H >= H_WET) & (H <= H_NINO)).sum())
    amarillo = int(((H > H_NINO)  & (H <= H_ADULTO)).sum())
    rojo     = int((H > H_ADULTO).sum())
    return verde, amarillo, rojo


# Tramos propios del desglose de 5 niveles de Q17 (no son umbrales Russo)
H_CRIT    = 1.00
H_EXTREMO = 2.00


def russo_cinco_niveles(H: np.ndarray) -> tuple[int, int, int, int, int]:
    """Cuenta celdas en los 5 niveles DISJUNTOS de peligrosidad por calado (Q17):
    somera (H≤0.25), niños (0.25<H≤0.50), adultos (0.50<H≤1.00),
    crítico (1.00<H≤2.00) y extremo (H>2.00)."""
    somera  = int(((H >= H_WET)   & (H <= H_NINO)).sum())
    ninos   = int(((H > H_NINO)   & (H <= H_ADULTO)).sum())
    adultos = int(((H > H_ADULTO) & (H <= H_CRIT)).sum())
    critico = int(((H > H_CRIT)   & (H <= H_EXTREMO)).sum())
    extremo = int((H > H_EXTREMO).sum())
    return somera, ninos, adultos, critico, extremo


def area_por_nivel_peligro(hora: int, bbox=None) -> dict:
    """
    Q15 — Área (km²) por nivel de peligro según calado:
      Verde   : H_WET ≤ H ≤ 0.25 m  |  Amarillo: 0.25 < H ≤ 0.50 m  |  Rojo: H > 0.50 m
    """
    t = hora_a_t(hora)
    r_min, r_max, c_min, c_max = _bbox_args(bbox)
    with tiledb.open(TILEDB_URI, mode="r") as A:
        res = A.query(attrs=["H"])[t, r_min:r_max+1, c_min:c_max+1]
    verde, amarillo, rojo = russo_traffic_light_counts(res["H"])
    km2 = CELL_AREA / 1e6
    return {
        "hora": hora,
        "verde_km2":    verde    * km2,
        "amarillo_km2": amarillo * km2,
        "rojo_km2":     rojo     * km2,
        "total_km2":    (verde + amarillo + rojo) * km2,
    }


# ═══════════════════════════════════════════════════════════════
# Demo — Ejemplos de uso
# ═══════════════════════════════════════════════════════════════

def main():
    global TILEDB_URI
    parser = argparse.ArgumentParser(description="Motor de consultas analíticas TileDB/Triton")
    parser.add_argument("--dataset",
                        default=os.environ.get("TRITON_DATASET", "datos1"),
                        help="Dataset a consultar (env: TRITON_DATASET)")
    args = parser.parse_args()
    TILEDB_URI = f"{BASE_URI}/{resolve_dataset(args.dataset)}"
    load_domain_constants(TILEDB_URI)

    X_EJ = XLLCORNER + NCOLS * CELLSIZE / 2
    Y_EJ = YLLCORNER + NROWS * CELLSIZE / 2
    BBOX = (X_EJ - 2500, Y_EJ - 2500, X_EJ + 2500, Y_EJ + 2500)

    print("=" * 70)
    print(f"  MOTOR DE CONSULTAS ANALÍTICAS — TileDB Triton  ({args.dataset})")
    print("=" * 70)

    print("\n[Q1] Calado en punto central, hora 60:")
    print(f"  H = {calado_en_punto(X_EJ, Y_EJ, 60):.4f} m")

    X_WET = XLLCORNER + 3000
    Y_WET = YLLCORNER + 5000
    print(f"\n[Q2] Serie temporal en ({X_WET}, {Y_WET}):")
    st = serie_temporal_punto(X_WET, Y_WET)
    for hora, h in zip(st["horas"], st["H"]):
        print(f"  hora {hora:3d}h → H = {h:.4f} m{' ← seco' if h == 0 else ''}")

    print("\n[Q3] Velocidad en zona central, hora 60:")
    v = velocidad_en_zona(*BBOX, 60)
    print(f"  V_max = {v['V_max']:.3f} m/s  |  V_media = {v['V_media']:.3f} m/s")

    print("\n[Q4] Zona inundada (H > 1 cm), hora 60:")
    zi = zonas_inundadas(60)
    print(f"  {zi['n_celdas']:,} celdas  →  {zi['area_km2']:.2f} km²")

    print("\n[Q5] Evolución extensión inundación:")
    ev = evolucion_extension()
    for h, a in zip(ev["horas"], ev["area_km2"]):
        print(f"  hora {h:3d}h → {a:.2f} km²")

    print("\n[Q6] Zonas con H > 0.30 m en hora 60:")
    z30 = zonas_inundadas(60, umbral_m=0.30)
    print(f"  {z30['n_celdas']:,} celdas  →  {z30['area_km2']:.2f} km²")

    print("\n[Q12] Área inundada por hora:")
    ai = area_inundada_por_hora()
    for h, a in zip(ai["horas"], ai["area_km2"]):
        print(f"  hora {h:3d}h → {a:.2f} km²")

    print("\n[Q13] Volumen de agua por hora (zona central):")
    vol = volumen_por_hora(bbox=BBOX)
    for h, v in zip(vol["horas"], vol["volumen_m3"]):
        if v > 0:
            print(f"  hora {h:3d}h → {v/1e6:.3f} Mm³")

    print("\n[Q14] Estadísticos zona central, hora 60:")
    for k, v in stats_zona(*BBOX, 60).items():
        print(f"  {k}: {v}")

    print("\n[Q10a] Peligrosidad adultos (Russo), hora 60:")
    pa = peligrosidad_adultos(60)
    print(f"  {pa['n_celdas']:,} celdas  →  {pa['area_km2']:.2f} km²")

    print("\n[Q10b] Peligrosidad niños (Russo), hora 60:")
    pn = peligrosidad_ninos(60)
    print(f"  {pn['n_celdas']:,} celdas  →  {pn['area_km2']:.2f} km²")

    print("\n[Q10c] Peligro vehículos ligeros (H > 0.30 m), hora 60:")
    pv = peligro_vehiculos_ligeros(60)
    print(f"  {pv['n_celdas']:,} celdas  →  {pv['area_km2']:.2f} km²")

    print("\n[Q15] Área por nivel de peligro, hora 60:")
    np_ = area_por_nivel_peligro(60)
    print(f"  Verde    (H ≤ 0.25 m):  {np_['verde_km2']:.2f} km²")
    print(f"  Amarillo (0.25–0.50 m): {np_['amarillo_km2']:.2f} km²")
    print(f"  Rojo     (H > 0.50 m):  {np_['rojo_km2']:.2f} km²")

    # Q7/Q8/Q9/Q11 — recorren los 20 pasos; ahora vectorizados con numpy groupby
    BBOX_P = (X_EJ - 5000, Y_EJ - 5000, X_EJ + 5000, Y_EJ + 5000)
    print(f"\n[Q7/Q8/Q9/Q11] Zona 10×10 km centrada en el dominio (vectorizado)...")

    print("\n[Q7] Hora de llegada del frente de inundación:")
    lf = hora_llegada_frente(bbox=BBOX_P)
    if len(lf["rows"]) > 0:
        print(f"  {len(lf['rows']):,} celdas | primera: hora {lf['hora_llegada'].min()} h"
              f" | última: hora {lf['hora_llegada'].max()} h")
    else:
        print("  Ninguna celda se inunda en esta zona")

    print("\n[Q8] Duración de inundación (H > 1 cm):")
    dur = duracion_inundacion(H_WET, bbox=BBOX_P)
    if len(dur["rows"]) > 0:
        print(f"  {len(dur['rows']):,} celdas | media: {dur['horas_inundado'].mean():.1f} h"
              f" | máx: {dur['horas_inundado'].max()} h")
    else:
        print("  Ninguna celda se inunda en esta zona")

    print("\n[Q9] Hora en que se alcanza el calado máximo (MH):")
    pico = hora_calado_maximo(bbox=BBOX_P)
    if len(pico["rows"]) > 0:
        hora_pico_comun = int(np.bincount(pico["hora_pico"] // STEP_H - 1).argmax())
        print(f"  {len(pico['rows']):,} celdas | hora más frecuente: {t_a_hora(hora_pico_comun)} h"
              f" | MH máx: {pico['MH'].max():.3f} m")
    else:
        print("  Sin datos en esta zona")

    print("\n[Q11] Ventana de paso para vehículos de emergencia (H < 0.60 m):")
    vent = ventana_vehiculos_emergencia(bbox=BBOX_P)
    if len(vent["rows"]) > 0:
        print(f"  {len(vent['rows']):,} celdas | ventana media: {vent['horas_practicable'].mean():.1f} h"
              f" | mín: {vent['horas_practicable'].min()} h | máx: {vent['horas_practicable'].max()} h")
    else:
        print("  Sin celdas practicables en esta zona")

    print("\n[Q16] Tiempo de evacuación (umbral adultos: H>0.50 m ó Q_mod>0.50 m²/s):")
    evac = tiempo_evacuacion(bbox=BBOX_P)
    if evac["n_pixeles"] > 0:
        print(f"  {evac['n_pixeles']:,} píxeles inundados")
        print(f"  Nunca peligroso (ventana ilimitada): {evac['nunca_peligroso']:,}")
        print(f"  Agua llega ya peligrosa (ventana=0): {evac['ya_peligroso']:,}")
        print(f"  Ventana media de evacuación: {evac['ventana_media_h']:.1f} h")
        print(f"  Ventana mínima de evacuación: {evac['ventana_min_h']:.1f} h")
    else:
        print("  Sin datos en esta zona")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
