#!/usr/bin/env python3
"""
app.py — Interfaz web para el motor de consultas analíticas TileDB/Triton.

Arquitectura:
  - Consultas pesadas (Q7/Q8/Q9/Q11/Q16 y todas las espaciales): grid acumulador
    paso a paso → memoria constante (~10 MB) en lugar de acumular 1 300 M celdas.
  - Consultas ligeras (Q1/Q2/Q5/Q12/Q13/Q14/Q15): delegadas a consultas_analiticas.py.

Uso:
  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import threading
import time

# Carga .env local si existe (para desarrollo local sin variables de entorno del sistema)
_env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _stc
import tiledb
import folium
from streamlit_folium import st_folium
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
import consultas_analiticas as ca
from config import dataset_label

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

from config import BASE_URI
DEFAULT_MAP_RES  = 1800   # resolución máxima del grid (px lado largo)
MAP_RES          = DEFAULT_MAP_RES

PLOT_CONFIG = {
    "displayModeBar":          True,
    "displaylogo":             False,
    "modeBarButtonsToRemove":  ["lasso2d", "select2d", "autoScale2d"],
    "scrollZoom":              True,
    "doubleClick":             "reset",
    "responsive":              True,
    "toImageButtonOptions":    {"format": "png", "filename": "spillway_query", "scale": 2},
}

H_WET          = ca.H_WET
H_ADULTO       = ca.H_ADULTO
HV_ADULTO      = ca.HV_ADULTO
H_NINO         = ca.H_NINO
HV_NINO        = ca.HV_NINO
H_VEHICULO     = ca.H_VEHICULO
H_EMERGENCIA   = ca.H_EMERGENCIA
STEP_H         = ca.STEP_H

# Paletas vivas — diseñadas para que cualquier celda con dato sea visible
@st.cache_data(show_spinner=False)
def russo_colorscale(zmax: float = 2.5):
    """
    Paleta categórica para calado H basada en umbrales Russo et al. (2013).
    Cada franja es un color distinto → se ve de un vistazo dónde están las zonas críticas.
      [0.00 – 0.25 m]  → cyan claro  (agua somera, seguro)
      [0.25 – 0.50 m]  → amarillo    (peligroso para niños)
      [0.50 – 1.00 m]  → naranja     (peligroso para adultos)
      [1.00 – 2.00 m]  → rojo        (zona crítica)
      [  > 2.00 m  ]   → morado      (extremo)
    """
    thr    = [0.25, 0.50, 1.0, 2.0]
    colors = ["#4dd0e1", "#fdd835", "#fb8c00", "#d32f2f", "#6a1b9a"]
    # construir escala con saltos discretos
    scale = [[0.0, colors[0]]]
    for t, c_prev, c_next in zip(thr, colors[:-1], colors[1:]):
        r = min(t / zmax, 1.0)
        scale.append([r, c_prev])
        scale.append([r, c_next])
    scale.append([1.0, colors[-1]])
    tickvals = [0.125, 0.375, 0.75, 1.5, 2.25]
    ticktext = None  # se genera con _t() en tiempo de render
    return scale, tickvals, ticktext, zmax


@st.cache_data(show_spinner=False)
def russo_ticktext(lang: str) -> list[str]:
    labels = _LANG[lang] if lang in _LANG else _LANG["es"]
    return [
        f"≤ 0.25 m<br>{labels['russo_shallow']}",
        f"0.25–0.50<br>{labels['russo_children']}",
        f"0.50–1.00<br>{labels['russo_adults']}",
        f"1.00–2.00<br>{labels['russo_critical']}",
        f"> 2.00 m<br>{labels['russo_extreme']}",
    ]


def xia_risk_colorscale():
    """Colorscale discreta para mapas de inestabilidad Xia (valores 0/1/2).
    0=seguro(verde) | 1=riesgo moderado(naranja) | 2=riesgo alto(rojo)."""
    return [
        [0.000, "#2ecc71"],
        [0.499, "#2ecc71"],
        [0.500, "#f39c12"],
        [0.999, "#f39c12"],
        [1.000, "#e74c3c"],
    ]


CMAP_H = [   # Degradado cyan → azul, la mitad clara es la mayor parte del rango
    [0.00, "#b3e5fc"],   # cyan muy claro
    [0.12, "#81d4fa"],
    [0.28, "#4fc3f7"],   # cyan medio
    [0.48, "#29b6f6"],
    [0.68, "#039be5"],
    [0.85, "#0277bd"],
    [1.00, "#01579b"],   # azul profundo, solo en los picos
]
CMAP_H_FOCUS = [   # Más contraste para mapas de inundación extensa
    [0.00, "#eaf7ff"],
    [0.12, "#9ddcff"],
    [0.28, "#4fc3f7"],
    [0.50, "#1d9fe3"],
    [0.72, "#0b74b8"],
    [0.88, "#085089"],
    [1.00, "#08306b"],
]
CMAP_PELIGRO = [   # Rojos vivos
    [0.0, "#ffd6d6"], [0.3, "#fb8478"], [0.6, "#e83b32"],
    [0.8, "#b71c1c"], [1.0, "#7f0000"],
]
CMAP_NARANJA = [
    [0.0, "#ffe0b2"], [0.3, "#ffb74d"], [0.6, "#fb8c00"],
    [1.0, "#bf360c"],
]
CMAP_DURACION = [   # Amarillo → naranja → rojo (más tiempo = más rojo)
    [0.0, "#fff8c4"], [0.25, "#fdd835"], [0.5, "#fb8c00"],
    [0.75, "#e53935"], [1.0, "#b71c1c"],
]
CMAP_EVACUA = [   # Rojo (poco tiempo) → amarillo → verde (mucho tiempo)
    [0.0, "#c62828"], [0.5, "#fbc02d"], [1.0, "#2e7d32"],
]
CMAP_LLEGADA   = "Turbo"     # Azul → verde → amarillo → rojo (más vivo que Plasma)
CMAP_VELOCIDAD = "Viridis"

from PIL import Image as _PILImage

@st.cache_resource(show_spinner=False)
def _build_favicon():
    path = os.path.join(os.path.dirname(__file__), "Logo_Spillway.png")
    logo = _PILImage.open(path).convert("RGBA")
    fav  = _PILImage.new("RGBA", logo.size, (17, 24, 39, 255))  # #111827
    fav.paste(logo, mask=logo.split()[3])
    return fav

_FAVICON = _build_favicon()

st.set_page_config(
    page_title="Spillway · Query Engine",
    page_icon=_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── i18n ─────────────────────────────────────────────────────────────────────
_LANG = {
    "es": {
        "title_prefix": "Motor de Consultas",
        "sidebar_title": "Motor de Consultas TileDB",
        "mode_label": "Modo",
        "mode_single": "🔍 Consulta individual",
        "mode_compare": "📊 Comparar datasets",
        "dataset_a": "📂 **Dataset A**",
        "dataset_b": "📂 **Dataset B**",
        "block_label": "📁 **Bloque**",
        "query_label": "🔍 **Consulta**",
        "params_label": "### Parámetros",
        "all_hours": "🕐 **Todas las horas** (serie temporal)",
        "hour_slider": "🕐 Hora (h)",
        "map_res": "🖼️ Resolución de render (px lado largo)",
        "run_btn": "▶  Ejecutar consulta",
        "bbox_label": "🗺️ Ventana espacial (bbox)",
        "bbox_caption": "Coordenadas en **km** · EPSG:32614 · Recomendado en Q7/Q8/Q9/Q11/Q16",
        "bbox_use": "Activar bbox",
        "threshold_h": "💧 Umbral H (m)",
        "mode_entry": "Modo de entrada",
        "map_entry": "🗺️ Mapa interactivo",
        "coord_entry": "📐 Coordenadas (m)",
        "csv_entry": "📄 CSV de puntos (Q1 y Q2)",
        "click_map": "#### 📍 Haz clic en el mapa para seleccionar el punto",
        "out_of_domain": "⚠️ Punto fuera del dominio. Haz clic dentro del rectángulo azul.",
        "coord_title": "#### 📐 Introduce las coordenadas en metros (EPSG:32614)",
        "x_east": "X (m Este)", "y_north": "Y (m Norte)",
        "csv_title": "#### 📄 Carga un CSV con columnas: `Nombre,X,Y` (coordenadas en metros)",
        "csv_example": "Ejemplo: `Punto A,700000,4350000`",
        "csv_upload": "Archivo CSV",
        "active_point": "Punto activo",
        "run_config": "Configura los parámetros en el panel lateral y pulsa **▶ Ejecutar consulta**.",
        "badge_auto": "⚡ Automática", "badge_manual": "▶ Manual",
        "wet_cells": "Celdas húmedas", "flooded_cells": "Celdas inundadas",
        "area": "Área", "max_area": "Área máxima", "mean_area": "Área media",
        "peak_hour": "Hora del pico", "mean_h": "H medio",
        "max_h": "H máximo", "min_h": "H mínimo",
        "vel_max": "V máxima", "vel_mean": "V media",
        "steps_flooded": "Pasos inundados",
        "first_arrival": "Primera llegada", "last_arrival": "Última llegada",
        "mean_duration": "Duración media", "max_duration": "Duración máx",
        "cells": "Celdas", "max_volume": "Volumen máximo",
        "danger_area_max": "Área con inestabilidad máxima", "mean_intensity": "Superación media del umbral",
        "peak_intensity": "Pico de superación",
        "depth_level": "Nivel de calado",
        "q13_compare_title": "Volumen de agua por hora — comparativa",
        "cells_danger": "Celdas con inestabilidad",
        "practicable_cells": "Celdas accesibles",
        "mean_window": "Ventana media", "min_window": "Ventana mínima",
        "max_practicable": "Área accesible máxima",
        "pixels_flooded": "Píxeles inundados",
        "already_danger": "Agua llega ya peligrosa",
        "never_danger": "Nunca peligroso",
        "dry_warning": "La celda está seca en este instante.",
        "q1_result": "Resultado", "q1_depth": "Calado",
        "q1_badge_dry": "Seca", "q1_badge_flooded": "Inundada",
        "q1_badge_safe": "Segura", "q1_badge_caution": "Precaución",
        "q1_badge_danger": "Peligrosa",
        "out_domain_error": "Coordenadas fuera del dominio.",
        "summary_steps": "Pasos temporales", "summary_extent": "Extensión dominio",
        "summary_area": "Área inundada (h60)", "summary_pct": "% dominio húmedo",
        "summary_hmax": "H máximo (h60)",
        "upload_csv": "Sube un CSV para ver los resultados aquí.",
        "danger_level": "Nivel de peligro según calado",
        "safe": "🟢 Seguro", "caution": "🟡 Precaución", "danger": "🔴 Peligroso",
        "download_geotiff": "⬇️ Descargar GeoTIFF",
        "q16_crit_h": "H crítico (m)", "q16_crit_q": "Q_mod crítico (m²/s)",
        "q16_threshold": "**Umbral crítico (evacuación):**",
        "coord_caption": "Punto activo",
        "only_one_dataset": "Solo hay un dataset disponible.",
        "no_datasets": "No se encontraron datasets en",
        "error_query": "❌ Error ejecutando la consulta",
        "zone_dry": "Zona seca en este instante.",
        "processing": "Iniciando...", "completed": "Completado",
        "loading_step": "Cargando paso temporal...",
        "iterating": "Recorriendo pasos temporales...",
        "calc_arrival": "Calculando hora de llegada del frente...",
        "calc_duration": "Calculando duración de inundación...",
        "calc_hmax": "Calculando hora de calado máximo...",
        "calc_emergency": "Calculando ventana de emergencia...",
        "calc_evacuation": "Calculando ventana de evacuación...",
        "calc_stats": "Calculando estadísticos...",
        "calc_areas": "Calculando áreas por nivel...",
        "dataset_summary": "📋 Resumen del dataset",
        "depth_h": "Calado H",
        "coordinates": "Coordenadas",
        "hour": "Hora",
        "hour_axis": "Hora (h)",
        "ref_step_help": "En paso de referencia hora 60",
        "arrival_map_title": "Hora de llegada del frente (h)",
        "peak_depth_map_title": "Hora en que se alcanza el calado máximo (h)",
        "peak_hour_short": "Hora pico",
        "cap_domain": "Dominio", "cap_cells": "celdas",
        "cap_res": "Resolución", "cap_cell": "celda",
        "cap_steps": "pasos de", "cap_sim": "h simulación",
        "cap_zone": "zona",
        "summary_grid": "Resolución",
        "note_visual_floor": "Visualmente se ocultan láminas menores de {v:.3f} m para reducir el teñido global.",
        "note_cropped": "La vista se ha recortado automáticamente a la zona con agua visible.",
        "note_bbox_tip": "Si quieres inspección fina, activa una `bbox` para centrar el análisis.",
        "geotiff_unavailable": "Export GeoTIFF no disponible",
        "sidebar_map": "### Mapa",
        "all_hours_help": "Si se activa, la consulta se ejecuta sobre todos los pasos y devuelve una gráfica de evolución en lugar de un mapa.",
        "map_res_help": "Aumenta el detalle del raster al recalcular el mapa. El zoom visual del navegador no recalcula datos; una `bbox` + reejecución sí.",
        "bbox_x_min": "X mín (km)", "bbox_x_max": "X máx (km)",
        "bbox_y_min": "Y mín (km)", "bbox_y_max": "Y máx (km)",
        "csv_invalid_filename": "Nombre de archivo no permitido: extensión doble detectada.",
        "csv_too_large": "El CSV supera el límite de 200 KB.",
        "csv_empty": "CSV vacío o sin datos válidos.",
        "point_label": "Punto",
        "dry_cell": "Seco", "out_domain_label": "Fuera de dominio",
        "csv_no_valid_points": "No se encontraron puntos válidos en el dominio.",
        "q2_multi_title": "Series temporales — múltiples puntos",
        "q2_series_title": "Serie temporal",
        "q2_h_mean_wet": "H medio (húmedo)", "q2_dry": "— seco —",
        "vel_evolution": "Evolución de la velocidad V (m/s)",
        "vel_max_sim": "V máxima (toda simulación)", "vel_mean_peak": "Pico de V media",
        "vel_map_title": "Velocidad V (m/s) — hora {hora} h",
        "q5_title": "Evolución del área inundada",
        "hover_arrival": "Llegada", "hover_duration": "Duración",
        "hover_window": "Ventana", "hover_intensity": "Superación",
        "q8_map_title": "Duración de inundación (h) · H ≥ {v:.3f} m",
        "q11_map_title": "Ventana de acceso para vehículos de emergencia (h)",
        "q11b_evol_title": "Evolución del área accesible para emergencia (H < 0.60 m)",
        "q11b_map_title": "Acceso de emergencia (H < 0.60 m) — hora {hora} h",
        "q12_title": "Área inundada por hora",
        "q13_title": "Volumen de agua por hora",
        "q14_evol_title": "Evolución de los estadísticos de H en zona",
        "q14_dist_title": "Distribución de H en zona — hora {hora} h",
        "q10_evol_prefix": "Evolución del área con inestabilidad",
        "q10_map_intensity": "intensidad vs umbral",
        "q15_evol_title": "Evolución del semáforo de peligro (km² apilados)",
        "q16_map_title": "Ventana de evacuación (h) · H > {h} m ó Q_mod > {q} m²/s",
        "q16_caption": "Verde = más tiempo para evacuar · Rojo = agua llega ya peligrosa · Celdas transparentes = sin datos (secas o no alcanzan umbral).",
        "q17_map_title": "Mapa de peligrosidad (semáforo Russo) — hora {hora} h",
        "q17_caption": "🟦 Somera (≤0.25 m) · 🟨 Peligroso niños (0.25–0.50) · 🟧 Peligroso adultos (0.50–1.00) · 🟥 Crítico (1.00–2.00) · 🟪 Extremo (>2.00 m) · según umbrales Russo et al. (2013)",
        "xia_personas_title": "Inestabilidad {tipo} — Xia et al. (2014) — hora {hora} h",
        "xia_vehiculos_title": "Inestabilidad de vehículos — Xia et al. (2022) — hora {hora} h",
        "xia_safe": "🟢 Zona segura", "xia_moderate": "🟡 Riesgo moderado", "xia_high": "🔴 Riesgo alto",
        "xia_caption_personas": "Criterio de inestabilidad para {tipo}: velocidad crítica U_c,p según Xia et al. (2014). 🟢 V < curva naranja · 🟡 V entre curvas · 🔴 V ≥ curva roja.",
        "xia_caption_vehiculos": "Criterio de arrastre para Mini Cooper según Xia et al. (2022). 🟢 V < 0.5·U_c,v · 🟡 0.5·U_c,v ≤ V < U_c,v · 🔴 V ≥ U_c,v.",
        "q18_title": "Zona de graves daños (RD 9/2008) — hora {hora} h",
        "q18_area": "Área graves daños",
        "q18_caption": "Zona de graves daños según Real Decreto 9/2008 (Dominio Público Hidráulico): H > 1 m, V > 1 m/s ó H·V > 0.5 m²/s.",
        "compare_banner": "Modo comparativo activo",
        "compare_hint": "Los mapas y gráficos muestran la misma variable para los dos eventos seleccionados. Diferencias en color o magnitud indican cuál fue más severo o cuándo llegó el frente en cada caso.",
        "q5_compare_title": "Evolución del área inundada — comparativa",
        "q12_compare_title": "Área inundada por hora — comparativa",
        "max_area_a": "Área máx. (A)", "max_area_b": "Área máx. (B)", "peak_hour_a": "Pico (A)", "peak_hour_b": "Pico (B)",
        "h_max_global": "H máximo (global)", "h_mean_global": "H medio (global)",
        "median": "Mediana",
        "q15_green_max": "🟢 Verde máximo", "q15_yellow_max": "🟡 Amarillo máximo", "q15_red_max": "🔴 Rojo máximo",
        "q15_green_label": "🟢 Verde (H ≤ 0.25 m)", "q15_yellow_label": "🟡 Amarillo (0.25–0.50 m)", "q15_red_label": "🔴 Rojo (H > 0.50 m)",
        "q15_green_bar": "Verde\n(H ≤ 0.25 m)", "q15_yellow_bar": "Amarillo\n(0.25–0.50 m)", "q15_red_bar": "Rojo\n(H > 0.50 m)",
        "q17_shallow": "🟦 Somera ≤0.25", "q17_children": "🟨 Niños 0.25-0.50",
        "q17_adults": "🟧 Adultos 0.50-1.0", "q17_critical": "🟥 Crítico 1.0-2.0", "q17_extreme": "🟪 Extremo >2.0",
        "russo_shallow": "Somera", "russo_children": "Niños", "russo_adults": "Adultos",
        "russo_critical": "Crítico", "russo_extreme": "Extremo",
        "q10a_label": "Inestabilidad de adultos (Russo et al., 2013)", "q10b_label": "Inestabilidad de niños (Russo et al., 2013)",
        "q10c_label": "Riesgo de arrastre para vehículos (Russo et al., 2013)",
        "q17_adults_help": "En Q15 se agrupa como 'Rojo'; aquí se desglosa más fino con Q17.",
        "range_valid": "Rango válido",
        "guide_title": "📖 Guía rápida · Cómo usar Spillway",
        "guide_body": """\
**1. Dataset** — Selecciona el evento de inundación en el panel lateral (datos1–8, años 1980–1987).
**2. Consulta** — Elige un bloque (A–H) y la consulta que necesites (Q1–Q18) en el panel lateral.
**3. Parámetros** — Ajusta hora, umbral y/o ventana espacial (bbox) según la consulta seleccionada.
**4. Ejecutar** — Pulsa **▶ Ejecutar consulta**. Las marcadas con ⚡ Auto se lanzan solas.

#### Bloques de consultas

| Bloque | Qué calcula | Tiempo aprox. |
|--------|-------------|---------------|
| 🎯 A | Calado/velocidad en un punto o zona pequeña | < 5 s |
| 💧 B | Extensión de zonas inundadas a un umbral dado | < 5 s |
| ⏱️ C | Hora de llegada del frente, duración y hora del pico | 15–90 s |
| ⚠️ D | Inestabilidad según Russo et al. (2013): umbrales H y Q_mod | 5–30 s |
| ⚠️ E | Inestabilidad según Xia et al. (2014/2022): velocidad crítica | 5–30 s |
| 🔴 F | Zona de graves daños normativa (RD 9/2008) | 5–30 s |
| 📊 G | Estadísticos de área, volumen y calado en zona | < 10 s |
| 🚨 H | Tiempo disponible para evacuar antes de umbral crítico | 20–60 s |

> 💡 **Truco**: Para las consultas lentas (Q7, Q8, Q9, Q11, Q16) activa la **bbox** en el panel lateral para acotar el área y reducir el tiempo de cálculo ×5–×10.
> 🔍 Usa el modo **📊 Comparar datasets** para ver dos eventos de inundación en paralelo.\
""",
    },
    "en": {
        "title_prefix": "Query Engine",
        "sidebar_title": "TileDB Query Engine",
        "mode_label": "Mode",
        "mode_single": "🔍 Single query",
        "mode_compare": "📊 Compare datasets",
        "dataset_a": "📂 **Dataset A**",
        "dataset_b": "📂 **Dataset B**",
        "block_label": "📁 **Block**",
        "query_label": "🔍 **Query**",
        "params_label": "### Parameters",
        "all_hours": "🕐 **All timesteps** (time series)",
        "hour_slider": "🕐 Hour (h)",
        "map_res": "🖼️ Render resolution (px long side)",
        "run_btn": "▶  Run query",
        "bbox_label": "🗺️ Spatial window (bbox)",
        "bbox_caption": "Coordinates in **km** · EPSG:32614 · Recommended for Q7/Q8/Q9/Q11/Q16",
        "bbox_use": "Enable bbox",
        "threshold_h": "💧 H threshold (m)",
        "mode_entry": "Input mode",
        "map_entry": "🗺️ Interactive map",
        "coord_entry": "📐 Coordinates (m)",
        "csv_entry": "📄 CSV of points (Q1 & Q2)",
        "click_map": "#### 📍 Click on the map to select a point",
        "out_of_domain": "⚠️ Point outside domain. Click inside the blue rectangle.",
        "coord_title": "#### 📐 Enter coordinates in metres (EPSG:32614)",
        "x_east": "X (m Easting)", "y_north": "Y (m Northing)",
        "csv_title": "#### 📄 Upload a CSV with columns: `Name,X,Y` (coordinates in metres)",
        "csv_example": "Example: `Point A,700000,4350000`",
        "csv_upload": "CSV file",
        "active_point": "Active point",
        "run_config": "Configure parameters in the side panel and click **▶ Run query**.",
        "badge_auto": "⚡ Auto", "badge_manual": "▶ Manual",
        "wet_cells": "Wet cells", "flooded_cells": "Flooded cells",
        "area": "Area", "max_area": "Max area", "mean_area": "Mean area",
        "peak_hour": "Peak hour", "mean_h": "Mean H",
        "max_h": "Max H", "min_h": "Min H",
        "vel_max": "Max V", "vel_mean": "Mean V",
        "steps_flooded": "Flooded timesteps",
        "first_arrival": "First arrival", "last_arrival": "Last arrival",
        "mean_duration": "Mean duration", "max_duration": "Max duration",
        "cells": "Cells", "max_volume": "Max volume",
        "danger_area_max": "Max instability area", "mean_intensity": "Mean threshold exceedance",
        "peak_intensity": "Peak exceedance",
        "depth_level": "Depth level",
        "q13_compare_title": "Water volume by hour — comparison",
        "cells_danger": "Cells with instability",
        "practicable_cells": "Accessible cells (H < 0.60 m)",
        "mean_window": "Mean window", "min_window": "Min window",
        "max_practicable": "Max accessible area (emergency)",
        "pixels_flooded": "Flooded pixels",
        "already_danger": "Water arrives already dangerous",
        "never_danger": "Never dangerous",
        "dry_warning": "Cell is dry at this timestep.",
        "q1_result": "Result", "q1_depth": "Depth",
        "q1_badge_dry": "Dry", "q1_badge_flooded": "Flooded",
        "q1_badge_safe": "Safe", "q1_badge_caution": "Caution",
        "q1_badge_danger": "Dangerous",
        "out_domain_error": "Coordinates outside domain.",
        "summary_steps": "Timesteps", "summary_extent": "Domain extent",
        "summary_area": "Flooded area (h60)", "summary_pct": "% wet domain",
        "summary_hmax": "Max H (h60)",
        "upload_csv": "Upload a CSV to see results here.",
        "danger_level": "Hazard level by depth",
        "safe": "🟢 Safe", "caution": "🟡 Caution (children)", "danger": "🔴 Dangerous",
        "download_geotiff": "⬇️ Download GeoTIFF",
        "q16_crit_h": "Critical H (m)", "q16_crit_q": "Critical Q_mod (m²/s)",
        "q16_threshold": "**Critical threshold (evacuation):**",
        "coord_caption": "Active point",
        "only_one_dataset": "Only one dataset available.",
        "no_datasets": "No datasets found in",
        "error_query": "❌ Error running query",
        "zone_dry": "Dry zone at this timestep.",
        "processing": "Starting...", "completed": "Completed",
        "loading_step": "Loading timestep...",
        "iterating": "Iterating timesteps...",
        "calc_arrival": "Computing arrival time...",
        "calc_duration": "Computing flood duration...",
        "calc_hmax": "Computing peak depth time...",
        "calc_emergency": "Computing emergency window...",
        "calc_evacuation": "Computing evacuation window...",
        "calc_stats": "Computing statistics...",
        "calc_areas": "Computing areas by level...",
        "dataset_summary": "📋 Dataset summary",
        "depth_h": "Depth H",
        "coordinates": "Coordinates",
        "hour": "Hour",
        "hour_axis": "Hour (h)",
        "ref_step_help": "At reference timestep h60",
        "arrival_map_title": "Flood arrival time (h)",
        "peak_depth_map_title": "Time of maximum depth (h)",
        "peak_hour_short": "Peak time",
        "cap_domain": "Domain", "cap_cells": "cells",
        "cap_res": "Resolution", "cap_cell": "cell",
        "cap_steps": "timesteps of", "cap_sim": "h simulation",
        "cap_zone": "zone",
        "summary_grid": "Resolution",
        "note_visual_floor": "Depths below {v:.3f} m are hidden visually to reduce global tinting.",
        "note_cropped": "View automatically cropped to the visible water area.",
        "note_bbox_tip": "For fine inspection, enable a `bbox` to focus the analysis.",
        "geotiff_unavailable": "GeoTIFF export unavailable",
        "sidebar_map": "### Map",
        "all_hours_help": "When enabled, the query runs over all timesteps and returns an evolution chart instead of a map.",
        "map_res_help": "Increases raster detail when recalculating the map. Browser zoom does not recalculate data; a `bbox` + re-run does.",
        "bbox_x_min": "X min (km)", "bbox_x_max": "X max (km)",
        "bbox_y_min": "Y min (km)", "bbox_y_max": "Y max (km)",
        "csv_invalid_filename": "File name not allowed: double extension detected.",
        "csv_too_large": "CSV exceeds the 200 KB limit.",
        "csv_empty": "CSV is empty or has no valid data.",
        "point_label": "Point",
        "dry_cell": "Dry", "out_domain_label": "Outside domain",
        "csv_no_valid_points": "No valid points found in the domain.",
        "q2_multi_title": "Time series — multiple points",
        "q2_series_title": "Time series",
        "q2_h_mean_wet": "Mean H (wet)", "q2_dry": "— dry —",
        "vel_evolution": "Velocity V evolution (m/s)",
        "vel_max_sim": "Max V (entire simulation)", "vel_mean_peak": "Peak mean V",
        "vel_map_title": "Velocity V (m/s) — hour {hora} h",
        "q5_title": "Flood extent over time",
        "hover_arrival": "Arrival", "hover_duration": "Duration",
        "hover_window": "Window", "hover_intensity": "Exceedance",
        "q8_map_title": "Flood duration (h) · H ≥ {v:.3f} m",
        "q11_map_title": "Accessible window for emergency vehicles (h)",
        "q11b_evol_title": "Emergency accessible area evolution (H < 0.60 m)",
        "q11b_map_title": "Emergency access (H < 0.60 m) — hour {hora} h",
        "q12_title": "Flooded area by hour",
        "q13_title": "Water volume by hour",
        "q14_evol_title": "H statistics evolution in zone",
        "q14_dist_title": "H distribution in zone — hour {hora} h",
        "q10_evol_prefix": "Instability area evolution",
        "q10_map_intensity": "intensity vs threshold",
        "q15_evol_title": "Hazard traffic-light evolution (stacked km²)",
        "q16_map_title": "Evacuation window (h) · H > {h} m or Q_mod > {q} m²/s",
        "q16_caption": "Green = more time to evacuate · Red = water already dangerous on arrival · Transparent cells = no data (dry or below threshold).",
        "q17_map_title": "Hazard map (Russo traffic-light) — hour {hora} h",
        "q17_caption": "🟦 Shallow (≤0.25 m) · 🟨 Dangerous for children (0.25–0.50) · 🟧 Dangerous for adults (0.50–1.00) · 🟥 Critical (1.00–2.00) · 🟪 Extreme (>2.00 m) · Russo et al. (2013) thresholds",
        "xia_personas_title": "Instability ({tipo}) — Xia et al. (2014) — hour {hora} h",
        "xia_vehiculos_title": "Vehicle instability — Xia et al. (2022) — hour {hora} h",
        "xia_safe": "🟢 Safe zone", "xia_moderate": "🟡 Moderate risk", "xia_high": "🔴 High risk",
        "xia_caption_personas": "Instability criterion for {tipo}: critical velocity U_c,p from Xia et al. (2014). 🟢 V < orange curve · 🟡 V between curves · 🔴 V ≥ red curve.",
        "xia_caption_vehiculos": "Vehicle sweep criterion for Mini Cooper from Xia et al. (2022). 🟢 V < 0.5·U_c,v · 🟡 0.5·U_c,v ≤ V < U_c,v · 🔴 V ≥ U_c,v.",
        "q18_title": "Severe damage zone (RD 9/2008) — hour {hora} h",
        "q18_area": "Severe damage area",
        "q18_caption": "Severe damage zone according to Real Decreto 9/2008 (Spanish Public Water Domain Regulation): H > 1 m, V > 1 m/s or H·V > 0.5 m²/s.",
        "compare_banner": "Comparison mode active",
        "compare_hint": "Maps and charts show the same variable for both selected flood events. Differences in colour or magnitude indicate which event was more severe or when the flood front arrived in each case.",
        "q5_compare_title": "Flooded area evolution — comparison",
        "q12_compare_title": "Flooded area by hour — comparison",
        "max_area_a": "Max area (A)", "max_area_b": "Max area (B)", "peak_hour_a": "Peak (A)", "peak_hour_b": "Peak (B)",
        "h_max_global": "Max H (global)", "h_mean_global": "Mean H (global)",
        "median": "Median",
        "q15_green_max": "🟢 Max green", "q15_yellow_max": "🟡 Max yellow", "q15_red_max": "🔴 Max red",
        "q15_green_label": "🟢 Green (H ≤ 0.25 m)", "q15_yellow_label": "🟡 Yellow (0.25–0.50 m)", "q15_red_label": "🔴 Red (H > 0.50 m)",
        "q15_green_bar": "Green\n(H ≤ 0.25 m)", "q15_yellow_bar": "Yellow\n(0.25–0.50 m)", "q15_red_bar": "Red\n(H > 0.50 m)",
        "q17_shallow": "🟦 Shallow ≤0.25", "q17_children": "🟨 Children 0.25-0.50",
        "q17_adults": "🟧 Adults 0.50-1.0", "q17_critical": "🟥 Critical 1.0-2.0", "q17_extreme": "🟪 Extreme >2.0",
        "russo_shallow": "Shallow", "russo_children": "Children", "russo_adults": "Adults",
        "russo_critical": "Critical", "russo_extreme": "Extreme",
        "q10a_label": "Adult instability (Russo et al., 2013)", "q10b_label": "Child instability (Russo et al., 2013)",
        "q10c_label": "Vehicle sweep risk (Russo et al., 2013)",
        "q17_adults_help": "In Q15 grouped as 'Red'; here broken down more finely with Q17.",
        "range_valid": "Valid range",
        "guide_title": "📖 Quick guide · How to use Spillway",
        "guide_body": """\
**1. Dataset** — Select the flood event in the side panel.
**2. Query** — Choose a block (A–H) and the query you need (Q1–Q18) in the side panel.
**3. Parameters** — Set the hour, threshold and/or spatial window (bbox) for the selected query.
**4. Run** — Click **▶ Run query**. Queries marked ⚡ Auto run automatically.

#### Query blocks

| Block | What it computes | Approx. time |
|-------|-----------------|--------------|
| 🎯 A | Depth/velocity at a point or small zone | < 5 s |
| 💧 B | Extent of flooded areas at a given threshold | < 5 s |
| ⏱️ C | Arrival time of flood front, duration, and peak hour | 15–90 s |
| ⚠️ D | Instability by Russo et al. (2013): H and Q_mod thresholds | 5–30 s |
| ⚠️ E | Instability by Xia et al. (2014/2022): critical velocity | 5–30 s |
| 🔴 F | Severe damage zone: RD 9/2008 normative criterion | 5–30 s |
| 📊 G | Area, volume, and depth statistics for a zone | < 10 s |
| 🚨 H | Time available to evacuate before critical threshold | 20–60 s |

> 💡 **Tip**: For slow queries (Q7, Q8, Q9, Q11, Q16) enable the **bbox** in the side panel to narrow the area and reduce computation time ×5–×10.
> 🔍 Use the **📊 Compare datasets** mode to overlay two flood events and identify which was more severe, where flooding extended further, or how the front arrived at different times.\
""",
    },
}

def _t(key: str) -> str:
    lang = st.session_state.get("lang", "es")
    return _LANG[lang].get(key, _LANG["es"].get(key, key))


# ── Logo (base64) ────────────────────────────────────────────────────────────
import base64 as _b64

@st.cache_resource(show_spinner=False)
def _logo_uri() -> str:
    path = os.path.join(os.path.dirname(__file__), "Logo_Spillway.png")
    with open(path, "rb") as f:
        return f"data:image/png;base64,{_b64.b64encode(f.read()).decode()}"

_LOGO_URI = _logo_uri()


# ── Login ────────────────────────────────────────────────────────────────────
_APP_USER      = os.environ.get("SPILLWAY_USER", "triton")
_APP_PASS_HASH = os.environ.get("SPILLWAY_PASS_HASH", "")
if not _APP_PASS_HASH:
    st.error("⚠️ SPILLWAY_PASS_HASH not configured. Set the environment variable.")
    st.stop()

def _login_screen() -> bool:
    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,600&family=Fira+Code:wght@400;500&display=swap');
    .stApp { background: #111827 !important; }
    #MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
    header[data-testid="stHeader"] { visibility: hidden !important; background: #111827 !important; }
    .block-container {
        padding: 0 !important; max-width: 100% !important;
        min-height: 100vh !important;
        display: flex !important; align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stVerticalBlock"] { gap: 0 !important; width: 100% !important; }

    .ltitle {
        font-family: 'Cormorant Garamond', serif;
        font-size: 3.4rem; font-style: italic; font-weight: 600;
        color: #eee8d5; letter-spacing: 0.04em; line-height: 1;
        margin-bottom: 0.2rem; text-align: center;
    }
    .lsub {
        font-family: 'Fira Code', monospace; font-size: 0.57rem;
        color: rgba(95,179,232,0.65); letter-spacing: 0.22em;
        text-transform: uppercase; text-align: center; margin-bottom: 0;
    }
    /* Una sola caja */
    [data-testid="stForm"] {
        background: #1c2540 !important;
        border: 1px solid rgba(212,168,67,0.38) !important;
        border-radius: 4px !important;
        padding: 2rem 2.4rem 2rem !important;
        box-shadow: 0 28px 70px rgba(0,0,0,0.55) !important;
    }
    /* Inputs: contenedor completo con fondo uniforme */
    .stTextInput [data-baseweb="input"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(212,168,67,0.25) !important;
        border-radius: 2px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput [data-baseweb="input"]:focus-within {
        border-color: rgba(212,168,67,0.65) !important;
        box-shadow: 0 0 0 2px rgba(212,168,67,0.1) !important;
        background: rgba(255,255,255,0.07) !important;
    }
    /* Todos los hijos del contenedor: fondo transparente para heredar el del padre */
    .stTextInput [data-baseweb="input"] > div,
    .stTextInput [data-baseweb="input"] > div > div {
        background: transparent !important;
    }
    .stTextInput input {
        background: transparent !important;
        border: none !important;
        color: #eee8d5 !important;
        font-family: 'Fira Code', monospace !important; font-size: 0.88rem !important;
    }
    .stTextInput input::placeholder { color: rgba(238,232,213,0.18) !important; }
    .stTextInput label {
        color: rgba(212,168,67,0.7) !important;
        font-family: 'Fira Code', monospace !important;
        font-size: 0.62rem !important; letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
    }
    .stFormSubmitButton > button {
        background: #1a6fa8 !important; border: none !important;
        color: #fff !important; border-radius: 2px !important;
        font-family: 'Fira Code', monospace !important; font-size: 0.78rem !important;
        font-weight: 600 !important; letter-spacing: 0.1em !important;
        text-transform: uppercase !important; width: 100% !important;
        padding: 0.8rem !important; margin-top: 1rem !important;
        transition: background 0.2s, box-shadow 0.2s !important;
    }
    .stFormSubmitButton > button:hover {
        background: #2281c0 !important; box-shadow: 0 4px 20px rgba(26,111,168,0.4) !important;
    }
    [data-testid="stAlert"] {
        background: rgba(220,60,60,0.08) !important;
        border: 1px solid rgba(220,60,60,0.3) !important;
        border-radius: 2px !important; color: #f09090 !important;
        font-family: 'Fira Code', monospace !important; font-size: 0.78rem !important;
    }
    /* Ocultar "Press Enter to submit" */
    [data-testid="InputInstructions"] { display: none !important; }
    small[data-testid="InputInstructions"] { display: none !important; }
    .stTextInput ~ small { display: none !important; }

    /* Espacio entre campos y tras la línea separadora */
    [data-testid="stForm"] .stTextInput { margin-bottom: 1.1rem !important; }
    [data-testid="stForm"] .stTextInput:first-of-type { margin-top: 1.2rem !important; }

    /* Botón del ojo: blanco, sin caja, altura completa del input */
    button[aria-label="Show password text"],
    button[aria-label="Hide password text"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 0 10px !important;
        margin: 0 !important;
        height: 100% !important;
        align-self: stretch !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: rgba(255,255,255,0.7) !important;
        cursor: pointer !important;
        transition: color 0.15s !important;
    }
    button[aria-label="Show password text"]:hover,
    button[aria-label="Hide password text"]:hover {
        background: transparent !important;
        color: #fff !important;
    }
    button[aria-label="Show password text"] svg,
    button[aria-label="Hide password text"] svg {
        width: 16px !important;
        height: 16px !important;
        display: block !important;
    }
    </style>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 0.85, 1])
    with col:
        with st.form("login"):
            st.markdown(f"""
            <div style="padding:0.6rem 0 1.4rem;display:flex;flex-direction:column;align-items:center">
                <img src="{_LOGO_URI}"
                     style="width:340px;height:340px;object-fit:contain;margin-bottom:1.2rem"
                     alt="Spillway logo"/>
                <div style="width:100%;height:1px;background:rgba(212,168,67,0.4);border-radius:1px"></div>
            </div>""", unsafe_allow_html=True)
            user = st.text_input("Usuario / User")
            pwd  = st.text_input("Contraseña / Password", type="password")
            ok   = st.form_submit_button("Acceder", width="stretch")
        if ok:
            pwd_hash = hashlib.sha256(pwd.encode()).hexdigest()
            if user == _APP_USER and hmac.compare_digest(pwd_hash, _APP_PASS_HASH):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Credenciales incorrectas · Wrong credentials.")

    return False

if not _login_screen():
    st.stop()

if "lang" not in st.session_state:
    st.session_state["lang"] = "es"

# ── Topbar: widgets nativos Streamlit posicionados fixed via CSS :has() ───────
# El sentinel permite al CSS localizar el bloque de columnas que sigue
st.markdown('<div id="spw-tb-sentinel"></div>', unsafe_allow_html=True)
_la = st.session_state.get("lang", "es")
_tbc1, _tbc2, _tbc_gap, _tbc3, _tbc4, _ = st.columns([0.052, 0.052, 0.028, 0.075, 0.065, 0.728])
with _tbc1:
    if st.button("ES", key="_tb_es",
                 type="primary" if _la == "es" else "secondary"):
        st.session_state["lang"] = "es"
        st.rerun()
with _tbc2:
    if st.button("EN", key="_tb_en",
                 type="primary" if _la == "en" else "secondary"):
        st.session_state["lang"] = "en"
        st.rerun()
with _tbc3:
    if st.button("Logout", key="_tb_logout"):
        st.session_state["authenticated"] = False
        st.rerun()
with _tbc4:
    if st.button("Clear", key="_tb_clear"):
        st.cache_data.clear()
        st.toast("Caché limpiado", icon="✅")

# ── Estilos personalizados ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Fira+Code:wght@400;500&display=swap');

:root {
    --bg:      #111827;
    --surf:    #1c2540;
    --surf2:   #151e35;
    --gold:    #d4a843;
    --gold-d:  rgba(212,168,67,0.25);
    --sea:     #5fb3e8;
    --text:    #eee8d5;
    --muted:   #a8bcc8;
    --border:  rgba(212,168,67,0.22);
}

/* ── Chrome ───────────────────────────────────────────────── */
#MainMenu, footer { display: none !important; }
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] {
    height: 0 !important; min-height: 0 !important;
    visibility: hidden !important; background: var(--bg) !important;
}
[data-testid="stExpandSidebarButton"] { visibility: visible !important; }

/* ── Base ─────────────────────────────────────────────────── */
.stApp { background: var(--bg) !important; }
.block-container {
    padding: 0 2rem 2rem !important;
    max-width: 100% !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0.5rem !important;
}

/* Colapsar el espacio que ocupan sentinel + topbar en el flujo normal */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel),
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div {
    height: 0 !important;
    min-height: 0 !important;
    overflow: visible !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Subir contenido del sidebar */
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebarHeader"] {
    height: 0 !important; min-height: 0 !important;
    padding: 0 !important; overflow: visible !important;
    position: relative !important; z-index: 999 !important;
}
[data-testid="stLogoSpacer"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] {
    position: absolute !important;
    top: 0.6rem !important; right: 0.5rem !important;
    z-index: 1000 !important; pointer-events: all !important;
}

/* Patrón sutil de cuadrícula cartográfica en fondo */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(74,158,218,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(74,158,218,0.035) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── Sidebar toggle ───────────────────────────────────────── */
[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    background: var(--gold) !important;
    border-radius: 0 6px 6px 0 !important;
    box-shadow: 3px 0 16px rgba(201,168,76,0.3) !important;
    position: fixed !important;
    top: 0.6rem !important;
    left: 0 !important;
}

/* Ocultar atribución de Leaflet */
.leaflet-control-attribution { display: none !important; }

/* Ocultar anclas automáticas de headings */
[data-testid="stHeadingWithActionElements"] a,
h1 a, h2 a, h3 a { display: none !important; }

/* ── Tipografía ───────────────────────────────────────────── */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.2rem !important; font-weight: 600 !important; font-style: italic !important;
    color: var(--text) !important; letter-spacing: 0.04em !important;
    margin-bottom: 0.1rem !important;
}
h2 {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.72rem !important; font-weight: 500 !important;
    color: var(--gold) !important;
    text-transform: uppercase !important; letter-spacing: 0.2em !important;
    margin-top: 1.8rem !important;
}
h3 {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.78rem !important; font-weight: 400 !important;
    color: var(--muted) !important; letter-spacing: 0.1em !important;
}
/* Sólo texto real — sin tocar span/div que Streamlit usa para iconos Material */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] p {
    font-family: 'Fira Code', monospace !important;
    color: var(--text) !important;
}
[data-testid="InputInstructions"] { display: none !important; }

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1020 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 6px 0 40px rgba(0,0,0,0.4) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5 { color: #eee8d5 !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #b8c8d4 !important; }
section[data-testid="stSidebar"] label { color: #c8d8e0 !important; font-size: 0.65rem !important; letter-spacing: 0.08em !important; }
section[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(212,168,67,0.22) !important; border-radius: 2px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #eee8d5 !important; }
section[data-testid="stSidebar"] [data-baseweb="input"] input { color: #eee8d5 !important; }
section[data-testid="stSidebar"] [data-testid="stInfo"] {
    background: rgba(74,158,218,0.07) !important;
    border: 1px solid rgba(74,158,218,0.2) !important;
    border-radius: 2px !important; color: #b8c8d4 !important;
}

/* ── Métricas ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--gold) !important;
    border-radius: 2px !important;
    padding: 0.85rem 1.1rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    transition: border-top-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stMetric"]:hover {
    border-top-color: #e8c96a !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(201,168,76,0.08) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.9rem !important; font-weight: 600 !important;
    color: var(--text) !important; letter-spacing: 0.02em !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.6rem !important; font-weight: 500 !important;
    color: var(--gold) !important;
    text-transform: uppercase !important; letter-spacing: 0.18em !important;
}

/* ── Botones ──────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(201,168,76,0.4) !important;
    color: var(--gold) !important;
    border-radius: 2px !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.78rem !important; letter-spacing: 0.08em !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(201,168,76,0.08) !important;
    border-color: var(--gold) !important; color: var(--text) !important;
    box-shadow: 0 0 20px rgba(201,168,76,0.12) !important;
}
button[kind="primary"] {
    background: #1a6fa8 !important;
    border-color: #1a6fa8 !important;
    color: #ffffff !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.82rem !important; font-style: normal !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(26,111,168,0.4) !important;
}
button[kind="primary"]:hover {
    background: #2281c0 !important;
    border-color: #2281c0 !important;
    box-shadow: 0 6px 28px rgba(26,111,168,0.55) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button {
    border-color: rgba(74,158,218,0.4) !important; color: var(--sea) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(74,158,218,0.08) !important;
    border-color: var(--sea) !important; color: var(--text) !important;
}

/* ── Query card ───────────────────────────────────────────── */
.query-card {
    background: var(--surf);
    border: 1px solid var(--border); border-left: 3px solid var(--gold);
    border-radius: 2px; padding: 1rem 1.4rem; margin: 0.4rem 0 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    position: relative;
}
.query-card::before {
    content: ''; position: absolute;
    top: 6px; right: 6px; width: 10px; height: 10px;
    border-top: 1px solid rgba(201,168,76,0.3);
    border-right: 1px solid rgba(201,168,76,0.3);
}
.query-card h4 {
    font-family: 'Cormorant Garamond', serif; margin: 0 0 0.3rem;
    color: var(--text); font-size: 1.1rem; font-weight: 600; font-style: italic;
    letter-spacing: 0.02em;
}
.query-card p { margin: 0; color: var(--muted); font-size: 0.78rem; line-height: 1.6; font-family: 'Fira Code', monospace; }

/* ── Expanders ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--surf) !important;
    border: 1px solid var(--border) !important; border-radius: 2px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Fira Code', monospace !important; font-size: 0.78rem !important;
    font-weight: 500 !important; color: var(--gold) !important;
    background: var(--surf2) !important; padding: 0.75rem 1rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

/* ── Selectbox ────────────────────────────────────────────── */
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.03) !important;
    border-color: var(--border) !important; border-radius: 2px !important;
    color: var(--text) !important; font-family: 'Fira Code', monospace !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: rgba(201,168,76,0.5) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.1) !important;
}
[data-baseweb="select"] span { color: var(--text) !important; }

/* ── Slider ───────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--gold) !important;
    box-shadow: 0 0 10px rgba(201,168,76,0.35) !important;
}

/* ── Progress ─────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--sea), var(--gold)) !important;
    border-radius: 2px !important;
}

/* ── Alerts ───────────────────────────────────────────────── */
[data-testid="stInfo"] {
    background: rgba(74,158,218,0.07) !important;
    border: 1px solid rgba(74,158,218,0.2) !important; border-radius: 2px !important;
    color: var(--text) !important;
}
[data-testid="stWarning"] {
    background: rgba(255,180,0,0.07) !important;
    border: 1px solid rgba(255,180,0,0.25) !important; border-radius: 2px !important;
}
[data-testid="stError"] {
    background: rgba(220,60,60,0.07) !important;
    border: 1px solid rgba(220,60,60,0.25) !important; border-radius: 2px !important;
}
[data-testid="stSuccess"] {
    background: rgba(50,180,100,0.07) !important;
    border: 1px solid rgba(50,180,100,0.25) !important; border-radius: 2px !important;
}

/* ── Caption ──────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important; font-size: 0.72rem !important;
    font-family: 'Fira Code', monospace !important;
}

/* ── Divider ──────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1rem 0 !important; }

/* ── Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important; border-radius: 2px !important; overflow: hidden !important;
}

/* ── Sidebar logout ───────────────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button {
    border-color: rgba(220,60,60,0.25) !important; color: var(--muted) !important;
    font-size: 0.72rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(220,60,60,0.6) !important; color: #f09090 !important;
    background: rgba(220,60,60,0.07) !important; box-shadow: none !important;
}

/* ── Fullscreen ───────────────────────────────────────────── */
[data-testid="stFullScreenFrame"] > div { height: 100% !important; }
[data-testid="stFullScreenFrame"] .js-plotly-plot,
[data-testid="stFullScreenFrame"] .plot-container,
[data-testid="stFullScreenFrame"] .plotly {
    height: 100% !important; min-height: 80vh !important;
}
</style>
""", unsafe_allow_html=True)

# ── Help overlay — CSS checkbox trick (sin JS, compatible con Streamlit) ─────
st.markdown("""
<style>
/* Checkbox e inputs ocultos — motor del toggle */
#spw-chk, #spw-es-r, #spw-en-r {
    position: absolute; opacity: 0; pointer-events: none; width: 0; height: 0;
}

/* ── Topbar: posicionar el bloque de columnas fixed top-right via :has() ── */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div {
    position: fixed !important;
    top: 14px !important; right: 58px !important;
    z-index: 99997 !important;
    width: auto !important; padding: 0 !important;
    background: transparent !important;
}
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stHorizontalBlock"] {
    gap: 4px !important; flex-wrap: nowrap !important; width: auto !important;
}
/* Sin gap entre ES y EN para que formen un pill */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stColumn"]:nth-child(1),
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stColumn"]:nth-child(2) {
    padding-right: 0 !important; padding-left: 0 !important; margin: 0 !important;
}
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stColumn"] {
    min-width: 0 !important; flex: none !important; width: auto !important;
    padding: 0 !important;
}
/* Botones del topbar — base */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div button {
    height: 34px !important; min-height: 34px !important;
    padding: 0 13px !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.6rem !important; letter-spacing: 0.1em !important;
    font-weight: 500 !important; text-transform: uppercase !important;
    border: 1.5px solid rgba(212,168,67,0.3) !important;
    background: #151e35 !important; color: rgba(168,188,200,0.75) !important;
    box-shadow: 0 2px 14px rgba(0,0,0,0.45) !important;
    transition: border-color .18s, color .18s, background .18s !important;
    line-height: 1 !important; border-radius: 4px !important;
}
/* ES — lado izquierdo del pill switch */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stColumn"]:nth-child(1) button {
    border-radius: 20px 0 0 20px !important;
    border-right: none !important;
    padding: 0 11px 0 14px !important;
}
/* EN — lado derecho del pill switch */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stColumn"]:nth-child(2) button {
    border-radius: 0 20px 20px 0 !important;
    border-left: 1px solid rgba(212,168,67,0.15) !important;
    padding: 0 14px 0 11px !important;
}
/* Hover del switch (ambos lados) */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stColumn"]:nth-child(1) button:hover,
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stColumn"]:nth-child(2) button:hover {
    background: rgba(212,168,67,0.18) !important;
    border-color: rgba(212,168,67,0.55) !important;
    color: #d4a843 !important;
}
/* Lado activo del switch */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div button[kind="primary"] {
    background: rgba(212,168,67,0.88) !important;
    border-color: rgba(212,168,67,0.88) !important;
    color: #111827 !important; font-weight: 700 !important;
}
/* Logout hover rojo (col 4 tras el gap) */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stColumn"]:nth-child(4) button:hover {
    border-color: rgba(220,80,80,0.65) !important;
    color: #f09090 !important; background: rgba(220,80,80,0.08) !important;
}
/* Clear hover azul */
[data-testid="stVerticalBlock"] > div:has(#spw-tb-sentinel) + div [data-testid="stColumn"]:nth-child(5) button:hover {
    border-color: rgba(95,179,232,0.55) !important;
    color: #5fb3e8 !important; background: rgba(95,179,232,0.07) !important;
}

/* ── Botón "?" guía — posición fixed propia ── */
label#spw-help-btn {
    position: fixed; top: 14px; right: 16px; z-index: 99997;
    width: 34px; height: 34px; border-radius: 50%;
    background: #151e35; border: 1.5px solid rgba(212,168,67,0.55);
    color: #d4a843; font-family: 'Fira Code', monospace;
    font-size: 1rem; font-weight: 700; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background .2s, color .2s, transform .15s, box-shadow .2s;
    user-select: none; box-shadow: 0 2px 14px rgba(0,0,0,0.5);
}
label#spw-help-btn:hover {
    background: #d4a843; color: #111827; transform: scale(1.1);
    box-shadow: 0 4px 22px rgba(212,168,67,0.4);
}
#spw-chk:checked ~ label#spw-help-btn { background: #d4a843; color: #111827; }

/* ── Backdrop ── */
label#spw-backdrop {
    position: fixed; inset: 0; z-index: 99998;
    background: rgba(8,12,22,0.7); backdrop-filter: blur(4px);
    opacity: 0; pointer-events: none; cursor: default;
    transition: opacity .28s ease;
}
#spw-chk:checked ~ label#spw-backdrop { opacity: 1; pointer-events: all; }

/* ── Panel ── */
#spw-panel {
    position: fixed; top: 0; right: 0; z-index: 99999;
    height: 100vh; width: min(560px, 96vw);
    background: #1c2540;
    border-left: 1px solid rgba(212,168,67,0.2);
    box-shadow: -20px 0 80px rgba(0,0,0,0.65);
    transform: translateX(100%);
    transition: transform .32s cubic-bezier(.4,0,.2,1);
    display: flex; flex-direction: column; overflow: hidden;
}
#spw-chk:checked ~ #spw-panel { transform: translateX(0); }

#spw-panel-top-line {
    height: 3px; flex-shrink: 0;
    background: linear-gradient(90deg, transparent 0%, #d4a843 35%, #5fb3e8 65%, transparent 100%);
}
#spw-panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 1.5rem 0.7rem; flex-shrink: 0;
}
#spw-panel-title {
    font-family: 'Cormorant Garamond', serif; font-size: 1.3rem;
    font-weight: 600; color: #eee8d5; letter-spacing: .04em;
}
#spw-panel-title em { color: #d4a843; font-style: italic; }

#spw-close {
    background: none; border: 1px solid rgba(212,168,67,0.18);
    color: #a8bcc8; font-size: .85rem; width: 27px; height: 27px;
    border-radius: 3px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: color .15s, border-color .15s, background .15s;
}
#spw-close:hover {
    color: #eee8d5; border-color: rgba(212,168,67,0.45);
    background: rgba(212,168,67,0.08);
}

#spw-lang-bar {
    display: flex; gap: .35rem; padding: 0 1.5rem .75rem;
    flex-shrink: 0; border-bottom: 1px solid rgba(212,168,67,0.1);
}
.spw-ltab {
    background: none; border: 1px solid rgba(212,168,67,0.18);
    color: #7a9aaa; font-family: 'Fira Code', monospace;
    font-size: .6rem; letter-spacing: .12em; text-transform: uppercase;
    padding: .2rem .65rem; border-radius: 2px; cursor: pointer;
    transition: all .15s;
}
.spw-ltab:hover { color: #eee8d5; border-color: rgba(212,168,67,0.38); }
/* Tab activo via radio CSS */
#spw-es-r:checked ~ #spw-panel .spw-es-tab,
#spw-en-r:checked ~ #spw-panel .spw-en-tab {
    background: rgba(212,168,67,0.1); border-color: rgba(212,168,67,0.5);
    color: #d4a843;
}

#spw-scroll {
    flex: 1; overflow-y: auto; padding: 1.3rem 1.5rem 2rem;
    scrollbar-width: thin; scrollbar-color: rgba(212,168,67,0.2) transparent;
}
#spw-scroll::-webkit-scrollbar { width: 3px; }
#spw-scroll::-webkit-scrollbar-thumb { background: rgba(212,168,67,0.2); border-radius: 2px; }

/* Panes de idioma controlados por radio */
#spw-es-pane { display: block; }
#spw-en-pane { display: none; }
#spw-en-r:checked ~ #spw-panel #spw-es-pane { display: none; }
#spw-en-r:checked ~ #spw-panel #spw-en-pane { display: block; }

.spw-section {
    font-family: 'Fira Code', monospace; font-size: .56rem;
    letter-spacing: .2em; text-transform: uppercase;
    color: rgba(212,168,67,0.6); margin: 0 0 .65rem;
    display: flex; align-items: center; gap: .5rem;
}
.spw-section::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(212,168,67,0.12);
}

.spw-steps { display: flex; flex-direction: column; gap: .65rem; margin-bottom: 1.7rem; }
.spw-step { display: flex; align-items: flex-start; gap: .8rem; }
.spw-num {
    flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%;
    background: rgba(212,168,67,0.1); border: 1px solid rgba(212,168,67,0.38);
    color: #d4a843; font-family: 'Fira Code', monospace;
    font-size: .72rem; font-weight: 600;
    display: flex; align-items: center; justify-content: center; margin-top: 1px;
}
.spw-stxt {
    font-family: 'Fira Code', monospace; font-size: .75rem;
    color: #b8ccda; line-height: 1.55;
}
.spw-stxt b { color: #dde8ee; font-weight: 500; }
.spw-stxt code {
    background: rgba(95,179,232,0.1); color: #5fb3e8;
    padding: .05em .3em; border-radius: 2px; font-size: .72rem;
}

.spw-blocks { display: flex; flex-direction: column; gap: .28rem; margin-bottom: 1.7rem; }
.spw-brow {
    display: grid; grid-template-columns: 2rem 1fr;
    align-items: center; gap: .65rem;
    padding: .48rem .65rem; border-radius: 3px;
    background: rgba(255,255,255,0.018);
    border: 1px solid rgba(255,255,255,0.04);
    transition: background .15s, border-color .15s;
}
.spw-brow:hover {
    background: rgba(212,168,67,0.035);
    border-color: rgba(212,168,67,0.1);
}
.spw-bicon { font-size: .95rem; text-align: center; }
.spw-bdesc { font-family: 'Fira Code', monospace; font-size: .7rem; color: #b8ccda; line-height: 1.38; }
.spw-bdesc b { color: #dde8ee; font-weight: 500; display: block; margin-bottom: .08rem; }
.spw-btime { font-family: 'Fira Code', monospace; font-size: .62rem; color: rgba(95,179,232,0.65); white-space: nowrap; text-align: right; }

.spw-tip {
    background: rgba(95,179,232,0.05);
    border: 1px solid rgba(95,179,232,0.16);
    border-left: 3px solid rgba(95,179,232,0.6);
    border-radius: 3px; padding: .65rem .85rem;
    font-family: 'Fira Code', monospace; font-size: .71rem;
    color: #b8ccda; line-height: 1.58; margin-bottom: .55rem;
}
.spw-tip b { color: #5fb3e8; font-weight: 500; }
.spw-tip code {
    background: rgba(95,179,232,0.1); color: #5fb3e8;
    padding: .05em .3em; border-radius: 2px;
}
</style>

<!-- Motor CSS del panel de guía -->
<input type="checkbox" id="spw-chk">
<input type="radio" name="spw-lang" id="spw-es-r" checked>
<input type="radio" name="spw-lang" id="spw-en-r">
<label for="spw-chk" id="spw-help-btn" title="Guia &middot; Guide">?</label>
<label for="spw-chk" id="spw-backdrop"></label>

<div id="spw-panel">
  <div id="spw-panel-top-line"></div>
  <div id="spw-panel-header">
    <div id="spw-panel-title"><em>Spillway</em> &mdash; Guia de uso</div>
    <label for="spw-chk" id="spw-close">&#x2715;</label>
  </div>
  <div id="spw-lang-bar">
    <label for="spw-es-r" class="spw-ltab spw-es-tab">ES &#x1F1EA;&#x1F1F8;</label>
    <label for="spw-en-r" class="spw-ltab spw-en-tab">EN &#x1F1EC;&#x1F1E7;</label>
  </div>
  <div id="spw-scroll">
    <div id="spw-es-pane">
      <p class="spw-section">como usar</p>
      <div class="spw-steps">
        <div class="spw-step"><div class="spw-num">1</div><div class="spw-stxt"><b>Dataset</b> &mdash; Selecciona el evento de inundaci&oacute;n en el panel lateral.</div></div>
        <div class="spw-step"><div class="spw-num">2</div><div class="spw-stxt"><b>Consulta</b> &mdash; Elige un bloque (A&ndash;H) y la consulta que necesites (Q1&ndash;Q18).</div></div>
        <div class="spw-step"><div class="spw-num">3</div><div class="spw-stxt"><b>Par&aacute;metros</b> &mdash; Ajusta hora, umbral y ventana espacial (<code>bbox</code>) en el panel lateral.</div></div>
        <div class="spw-step"><div class="spw-num">4</div><div class="spw-stxt"><b>Ejecutar</b> &mdash; Pulsa <code>&#x25B6; Ejecutar consulta</code>. Las marcadas &#x26A1; Auto se lanzan solas.</div></div>
      </div>
      <p class="spw-section">bloques de consultas</p>
      <div class="spw-blocks">
        <div class="spw-brow"><div class="spw-bicon">&#x1F3AF;</div><div class="spw-bdesc"><b>A &mdash; Puntuales y velocidad</b>Calado H y velocidad en un punto o zona (Q1&ndash;Q3)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x1F4A7;</div><div class="spw-bdesc"><b>B &mdash; Umbrales espaciales</b>Extensi&oacute;n inundada a un umbral dado (Q4&ndash;Q6)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x23F1;&#xFE0F;</div><div class="spw-bdesc"><b>C &mdash; Indicadores temporales</b>Hora de llegada, duraci&oacute;n y hora del pico (Q7&ndash;Q9, Q11)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x26A0;&#xFE0F;</div><div class="spw-bdesc"><b>D &mdash; Inestabilidad Russo (2013)</b>Criterios adultos / ni&ntilde;os / veh&iacute;culos por H y Q_mod (Q10a&ndash;Q17)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x26A0;&#xFE0F;</div><div class="spw-bdesc"><b>E &mdash; Inestabilidad Xia (2014/2022)</b>Velocidad cr&iacute;tica para personas y veh&iacute;culos; 3 niveles (Q10a-Xia&ndash;Q10c-Xia)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x1F534;</div><div class="spw-bdesc"><b>F &mdash; Da&ntilde;os normativos (RD 9/2008)</b>Zona de graves da&ntilde;os: H&gt;1 m, V&gt;1 m/s &oacute; H&middot;V&gt;0,5 m&sup2;/s (Q18)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x1F4CA;</div><div class="spw-bdesc"><b>G &mdash; Estad&iacute;sticos espaciales</b>&Aacute;rea, volumen y percentiles de calado (Q12&ndash;Q15)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x1F6A8;</div><div class="spw-bdesc"><b>H &mdash; Evacuaci&oacute;n</b>Tiempo disponible antes del umbral cr&iacute;tico (Q16)</div></div>
      </div>
      <p class="spw-section">consejos</p>
      <div class="spw-tip"><b>bbox</b> &mdash; Para Q7&ndash;Q9, Q11, Q16 y las consultas Xia activa la ventana espacial en el panel lateral. Reduce el tiempo de c&aacute;lculo <b>&times;5&ndash;&times;10</b>.</div>
      <div class="spw-tip"><b>Comparar</b> &mdash; Usa el modo <code>&#x1F4CA; Comparar datasets</code> para analizar dos eventos en paralelo.</div>
    </div>
    <div id="spw-en-pane">
      <p class="spw-section">how to use</p>
      <div class="spw-steps">
        <div class="spw-step"><div class="spw-num">1</div><div class="spw-stxt"><b>Dataset</b> &mdash; Select the flood event in the side panel.</div></div>
        <div class="spw-step"><div class="spw-num">2</div><div class="spw-stxt"><b>Query</b> &mdash; Choose a block (A&ndash;H) and the query you need (Q1&ndash;Q18).</div></div>
        <div class="spw-step"><div class="spw-num">3</div><div class="spw-stxt"><b>Parameters</b> &mdash; Set the hour, threshold and spatial window (<code>bbox</code>) in the side panel.</div></div>
        <div class="spw-step"><div class="spw-num">4</div><div class="spw-stxt"><b>Run</b> &mdash; Click <code>&#x25B6; Run query</code>. Queries marked &#x26A1; Auto run automatically.</div></div>
      </div>
      <p class="spw-section">query blocks</p>
      <div class="spw-blocks">
        <div class="spw-brow"><div class="spw-bicon">&#x1F3AF;</div><div class="spw-bdesc"><b>A &mdash; Point queries &amp; velocity</b>Depth H and velocity at a point or zone (Q1&ndash;Q3)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x1F4A7;</div><div class="spw-bdesc"><b>B &mdash; Spatial thresholds</b>Flooded extent at a given depth threshold (Q4&ndash;Q6)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x23F1;&#xFE0F;</div><div class="spw-bdesc"><b>C &mdash; Temporal indicators</b>Arrival time, flood duration, peak depth hour (Q7&ndash;Q9, Q11)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x26A0;&#xFE0F;</div><div class="spw-bdesc"><b>D &mdash; Russo instability (2013)</b>H and Q_mod thresholds for adults / children / vehicles (Q10a&ndash;Q17)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x26A0;&#xFE0F;</div><div class="spw-bdesc"><b>E &mdash; Xia instability (2014/2022)</b>Critical velocity for people and vehicles; 3 risk levels (Q10a-Xia&ndash;Q10c-Xia)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x1F534;</div><div class="spw-bdesc"><b>F &mdash; Normative damage (RD 9/2008)</b>Severe damage zone: H&gt;1 m, V&gt;1 m/s or H&middot;V&gt;0.5 m&sup2;/s (Q18)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x1F4CA;</div><div class="spw-bdesc"><b>G &mdash; Spatial statistics</b>Area, volume, and depth percentiles (Q12&ndash;Q15)</div></div>
        <div class="spw-brow"><div class="spw-bicon">&#x1F6A8;</div><div class="spw-bdesc"><b>H &mdash; Evacuation</b>Time available before critical threshold (Q16)</div></div>
      </div>
      <p class="spw-section">tips</p>
      <div class="spw-tip"><b>bbox</b> &mdash; For Q7&ndash;Q9, Q11, Q16 and Xia queries enable the spatial window in the side panel. Reduces computation time <b>&times;5&ndash;&times;10</b>.</div>
      <div class="spw-tip"><b>Compare</b> &mdash; Use <code>&#x1F4CA; Compare datasets</code> mode to analyse two flood events side by side.</div>
    </div>
  </div>
</div>

""", unsafe_allow_html=True)




# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — metadatos, ventanas, grids, figuras
# ═══════════════════════════════════════════════════════════════════════════════

def _horas(meta: dict) -> list[int]:
    return [(i + 1) * STEP_H for i in range(meta["n_steps"])]


def _cell_km2(meta: dict) -> float:
    return meta["cellsize"]**2 / 1e6


@st.cache_data(ttl=3600, show_spinner=False)
def discover_datasets() -> list[str]:
    try:
        return sorted(d for d in os.listdir(BASE_URI)
                      if os.path.isdir(os.path.join(BASE_URI, d)))
    except FileNotFoundError:
        return []


_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _dataset_display(long_name: str) -> str:
    """datos1 alias + fecha extraída del nombre del directorio: 'datos1 · Jul 1980'."""
    if not long_name:
        return "—"
    alias = dataset_label(long_name)
    if st.session_state.get("lang") == "en":
        alias = alias.replace("datos", "dataset")
    parts = long_name.rsplit("_", 2)
    if len(parts) == 3 and len(parts[1]) == 8 and parts[1].isdigit():
        d = parts[1]
        month = _MONTHS[int(d[4:6]) - 1]
        return f"{alias} · {month} {d[:4]}"
    return alias


@st.cache_data(show_spinner=False)
def get_meta(dataset: str) -> dict:
    uri = f"{BASE_URI}/{dataset}"
    with tiledb.open(uri, mode="r") as A:
        meta = dict(A.meta)
        ts   = json.loads(meta["time_steps"])
    tr    = json.loads(meta["transform"])
    ncols = int(meta["width"])
    nrows = int(meta["height"])
    cs    = abs(float(tr[0]))
    xll   = float(tr[2])
    yll   = float(tr[5]) + float(tr[4]) * nrows
    return {
        "uri": uri, "time_steps": ts,
        "ncols": ncols, "nrows": nrows, "cellsize": cs,
        "xll": xll, "yll": yll,
        "x_max": xll + ncols * cs, "y_max": yll + nrows * cs,
        "n_steps": len(ts),
    }


_ca_lock = threading.Lock()

def setup_ca(meta: dict):
    """Apunta el módulo consultas_analiticas al dataset actual."""
    with _ca_lock:
        if ca.TILEDB_URI == meta["uri"]:
            return
        ca.TILEDB_URI = meta["uri"]
        ca.NCOLS      = meta["ncols"]
        ca.NROWS      = meta["nrows"]
        ca.CELLSIZE   = meta["cellsize"]
        ca.CELL_AREA  = meta["cellsize"] ** 2
        ca.XLLCORNER  = meta["xll"]
        ca.YLLCORNER  = meta["yll"]
        ca.N_STEPS    = meta["n_steps"]


# ── Conversión UTM ↔ WGS84 ────────────────────────────────────────────────────
_to_wgs84 = Transformer.from_crs("EPSG:32614", "EPSG:4326", always_xy=True)
_to_utm   = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)

def utm_to_latlon(x: float, y: float) -> tuple[float, float]:
    lon, lat = _to_wgs84.transform(x, y)
    return lat, lon

def latlon_to_utm(lat: float, lon: float) -> tuple[float, float]:
    x, y = _to_utm.transform(lon, lat)
    return x, y


def point_picker_map(meta: dict, x_sel: float, y_sel: float, height: int = 380):
    """
    Mapa Folium interactivo para seleccionar un punto dentro del dominio.
    Devuelve el objeto st_folium con el último clic.
    """
    lat_c, lon_c   = utm_to_latlon((meta["xll"] + meta["x_max"]) / 2,
                                    (meta["yll"] + meta["y_max"]) / 2)
    lat_sw, lon_sw = utm_to_latlon(meta["xll"],   meta["yll"])
    lat_ne, lon_ne = utm_to_latlon(meta["x_max"], meta["y_max"])
    lat_p,  lon_p  = utm_to_latlon(x_sel, y_sel)

    m = folium.Map(location=[lat_c, lon_c], zoom_start=8,
                   tiles="CartoDB positron", attr=" ")
    folium.Element(
        '<style>.leaflet-control-attribution{display:none!important}</style>'
    ).add_to(m.get_root().header)

    folium.Rectangle(
        bounds=[[lat_sw, lon_sw], [lat_ne, lon_ne]],
        color="#1565c0", weight=2,
        fill=True, fill_opacity=0.04,
        tooltip="Dominio de simulación",
    ).add_to(m)

    folium.Marker(
        [lat_p, lon_p],
        tooltip=(f"X = {x_sel/1000:.2f} km<br>Y = {y_sel/1000:.2f} km"),
        icon=folium.Icon(color="red", icon="circle", prefix="fa"),
    ).add_to(m)

    return st_folium(m, height=height, width="100%",
                     returned_objects=["last_clicked"], key=f"map_{meta['uri']}")


def compute_window(meta: dict, bbox):
    """Bbox en coords EPSG → índices (r_min, r_max, c_min, c_max) en TileDB."""
    nrows, ncols = meta["nrows"], meta["ncols"]
    cs           = meta["cellsize"]
    xll, yll     = meta["xll"], meta["yll"]
    if bbox is None:
        return 0, nrows - 1, 0, ncols - 1
    bx0, by0, bx1, by1 = bbox
    c_min = max(0,         int((bx0 - xll) / cs))
    c_max = min(ncols - 1, int((bx1 - xll) / cs))
    r_min = max(0,         int((yll + nrows * cs - by1) / cs))
    r_max = min(nrows - 1, int((yll + nrows * cs - by0) / cs))
    return r_min, r_max, c_min, c_max


def grid_info(meta, r_min, r_max, c_min, c_max):
    sub_nr = r_max - r_min + 1
    sub_nc = c_max - c_min + 1
    scale  = MAP_RES / max(sub_nr, sub_nc)
    gr     = max(1, int(sub_nr * scale))
    gc     = max(1, int(sub_nc * scale))

    cs    = meta["cellsize"]
    xll   = meta["xll"]
    yll   = meta["yll"]
    nrows = meta["nrows"]
    x0    = xll + c_min * cs
    x1    = xll + (c_max + 1) * cs
    y_top = yll + (nrows - r_min) * cs
    y_bot = yll + (nrows - r_max - 1) * cs
    x_ax  = np.linspace(x0, x1, gc)
    y_ax  = np.linspace(y_top, y_bot, gr)
    return scale, gr, gc, x_ax, y_ax


def cmetric(col, label: str, value: str, delta=None):
    """Métrica con estilo nativo de Streamlit; delta opcional en pequeño debajo."""
    col.metric(label, value, delta)


def show(fig, **kwargs):
    kwargs.pop("width", None)
    kwargs.setdefault("use_container_width", True)
    st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, **kwargs)


# ── Helpers de color ──────────────────────────────────────────────────────────

def _adaptive_clip(valid: np.ndarray) -> tuple[float, float]:
    """Percentiles de recorte adaptativos según la variabilidad del dato."""
    if len(valid) == 0:
        return 0.0, 1.0
    cv = valid.std() / (abs(valid.mean()) + 1e-10)
    if cv > 2:    return float(np.nanpercentile(valid, 5)),  float(np.nanpercentile(valid, 95))
    elif cv > 1:  return float(np.nanpercentile(valid, 2)),  float(np.nanpercentile(valid, 98))
    else:         return float(np.nanpercentile(valid, 1)),  float(np.nanpercentile(valid, 99))


def _band_starts(edges: list[float], colors: list[str], zmax: float):
    """Convierte bordes candidatos en bandas monotónicas dentro del rango visual."""
    if zmax <= 0:
        return [(0.0, colors[0])], 1.0
    clipped = sorted({float(v) for v in edges if 0.0 <= float(v) <= zmax})
    if not clipped:
        clipped = [0.0, zmax]
    elif clipped[-1] < zmax:
        clipped.append(zmax)
    if len(clipped) == 1:
        clipped.append(zmax if zmax > clipped[0] else clipped[0] + 1.0)
    return [(clipped[i], colors[min(i, len(colors) - 1)]) for i in range(len(clipped) - 1)], zmax


def _crop_to_valid(grid: np.ndarray, x_ax: np.ndarray, y_ax: np.ndarray, pad: int = 6):
    """Recorta la vista al bounding box de celdas visibles, con un pequeño margen."""
    valid = np.isfinite(grid)
    if not valid.any():
        return grid, x_ax, y_ax, False
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    r0 = max(0, int(rows[0]) - pad)
    r1 = min(grid.shape[0], int(rows[-1]) + pad + 1)
    c0 = max(0, int(cols[0]) - pad)
    c1 = min(grid.shape[1], int(cols[-1]) + pad + 1)
    cropped = (r0 > 0) or (r1 < grid.shape[0]) or (c0 > 0) or (c1 < grid.shape[1])
    return grid[r0:r1, c0:c1], x_ax[c0:c1], y_ax[r0:r1], cropped


def _focus_h_visual(grid: np.ndarray, x_ax: np.ndarray, y_ax: np.ndarray,
                    query_threshold: float, auto_crop: bool = False):
    """
    Oculta visualmente la lámina muy somera para que destaquen los cauces.
    Las métricas siguen usando el umbral original de consulta.
    """
    wet = grid[np.isfinite(grid) & (grid >= query_threshold)]
    if len(wet) == 0:
        return grid, x_ax, y_ax, query_threshold, False

    if query_threshold <= 0.05:
        p35 = float(np.nanpercentile(wet, 35))
        visual_floor = max(query_threshold, min(max(p35, query_threshold * 2.0), 0.05))
    else:
        visual_floor = query_threshold

    focus = np.where(np.isfinite(grid) & (grid >= visual_floor), grid, np.nan)
    if not np.isfinite(focus).any():
        focus = np.where(np.isfinite(grid) & (grid >= query_threshold), grid, np.nan)
        visual_floor = query_threshold

    cropped = False
    if auto_crop:
        focus, x_ax, y_ax, cropped = _crop_to_valid(focus, x_ax, y_ax, pad=6)

    return focus, x_ax, y_ax, float(visual_floor), cropped


def _discrete_cmap(bands: list[tuple], zmin: float, zmax: float):
    """
    Genera colorscale discreta de Plotly.
    bands: [(val_inicio, color), ...] en unidades de datos.
    El último color cubre hasta zmax.
    """
    span = zmax - zmin or 1.0
    scale, tvals, ttexts = [], [], []
    for i, (v, c) in enumerate(bands):
        v_next = bands[i + 1][0] if i + 1 < len(bands) else zmax
        r0 = max(0.0, min(1.0, (v      - zmin) / span))
        r1 = max(0.0, min(1.0, (v_next - zmin) / span))
        if r0 < r1:
            scale += [[r0, c], [r1, c]]
        tvals.append((v + v_next) / 2)
    if scale and scale[-1][0] < 1.0:
        scale.append([1.0, scale[-1][1]])
    return scale


def _log_h_setup(grid: np.ndarray, adaptive: bool = True):
    """
    Transforma grid de H a log10, devuelve (log_grid, tickvals, ticktext).
    Los ticks de la colorbar se muestran en metros originales.
    """
    log_grid = np.where(np.isfinite(grid) & (grid > 0),
                        np.log10(np.maximum(grid, H_WET)), np.nan)
    valid = log_grid[np.isfinite(log_grid)]
    if len(valid):
        v_floor = float(np.log10(H_WET))
        full_min = float(valid.min())
        full_max = float(valid.max())
        if adaptive:
            v_min, v_max = _adaptive_clip(valid)
            v_min = max(float(v_min), v_floor)
            v_max = min(float(v_max), full_max)
            # Si el rango útil queda demasiado estrecho, mejor volver al rango real.
            if (v_max - v_min) < 0.40:
                v_min, v_max = full_min, full_max
        else:
            v_min, v_max = full_min, full_max
    else:
        v_min, v_max = np.log10(H_WET), 1.0
    # Candidatos de tick en escala log
    candidates = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
    tvals, ttexts = [], []
    for c in candidates:
        lv = np.log10(c)
        if v_min <= lv <= v_max:
            tvals.append(lv)
            ttexts.append(f"{c:.2g} m")
    if not tvals and len(valid):
        raw_mid = float(np.power(10.0, (v_min + v_max) / 2.0))
        tvals = [(v_min + v_max) / 2.0]
        ttexts = [f"{raw_mid:.2g} m"]
    return log_grid, v_min, v_max, tvals, ttexts


# ── Definiciones de bandas discretas por tipo de variable ────────────────────

@st.cache_data(show_spinner=False)
def _bands_llegada(n_steps: int):
    """Q7 / Q9 — hora llegada frente o calado máximo. Pronto=rojo, tarde=verde."""
    total_h = n_steps * STEP_H
    return _band_starts(
        [0.0, 6.0, 12.0, 24.0, total_h * 0.4, total_h * 0.6, total_h * 0.8],
        ["#4a148c", "#b71c1c", "#e53935", "#fb8c00", "#fdd835", "#81c784", "#2e7d32"],
        total_h,
    )


@st.cache_data(show_spinner=False)
def _bands_duracion(n_steps: int):
    """Q8 — duración inundación. Corta=verde, permanente=rojo oscuro."""
    total_h = n_steps * STEP_H
    return _band_starts(
        [0.0, 6.0, 24.0, 48.0, 72.0, total_h * 0.8],
        ["#e8f5e9", "#81c784", "#fdd835", "#fb8c00", "#e53935", "#880e4f"],
        total_h,
    )


@st.cache_data(show_spinner=False)
def _bands_ventana(n_steps: int):
    """Q11 ventana emergencia / Q16 evacuación. Poco tiempo=rojo, mucho=verde."""
    total_h = n_steps * STEP_H
    return _band_starts(
        [0.0, 6.0, 12.0, 24.0, 48.0],
        ["#b71c1c", "#e53935", "#fb8c00", "#fdd835", "#81c784", "#2e7d32"],
        total_h,
    )


@st.cache_data(show_spinner=False)
def _bands_intensidad():
    """Q10a/b/c — intensidad de peligro (ratio vs umbral). 1=justo en el límite."""
    return _band_starts(
        [1.0, 1.5, 2.0, 3.0, 5.0],
        ["#fff9c4", "#fdd835", "#fb8c00", "#e53935", "#880e4f"],
        6.0,
    )


# ── Heatmap base ──────────────────────────────────────────────────────────────

def make_heatmap(grid, x_ax, y_ax, title, cmap, unit="",
                 zmin=None, zmax=None,
                 cbar_tickvals=None, cbar_ticktext=None,
                 hover_fmt=".3f",
                 hover_grid=None,
                 hover_label="Valor",
                 hover_unit=None,
                 clip_pct=None,
                 adaptive_clip=False,
                 zsmooth="best"):
    """
    Heatmap base. zmin/zmax ya vienen calculados externamente.
    """
    TXT = "#1a2834"
    x_km = x_ax / 1000.0
    y_km = y_ax / 1000.0
    valid = grid[np.isfinite(grid)]

    if zmin is None or zmax is None:
        if len(valid):
            if clip_pct is not None:
                lo = float(np.nanpercentile(valid, clip_pct[0]))
                hi = float(np.nanpercentile(valid, clip_pct[1]))
            elif adaptive_clip:
                lo, hi = _adaptive_clip(valid)
            else:
                lo = float(np.nanmin(valid))
                hi = float(np.nanmax(valid))
        else:
            lo, hi = 0.0, 1.0
        if zmin is None:
            zmin = lo
        if zmax is None:
            zmax = hi

    if not np.isfinite(zmin):
        zmin = 0.0
    if not np.isfinite(zmax):
        zmax = 1.0
    if zmin == zmax:
        zmax = zmin + 1.0

    hover_ref = "z"
    heatmap_kwargs = {}
    if hover_grid is not None:
        hover_ref = "customdata"
        heatmap_kwargs["customdata"] = hover_grid
    hover_suffix = f" {hover_unit}" if hover_unit else ""

    fig = go.Figure(go.Heatmap(
        z=grid, x=x_km, y=y_km,
        colorscale=cmap,
        zmin=zmin, zmax=zmax, zsmooth=zsmooth,
        colorbar=dict(
            title=dict(text=unit, side="right",
                       font=dict(size=13, color=TXT)),
            thickness=18, len=0.85,
            tickfont=dict(size=11, color=TXT),
            outlinewidth=0,
            tickvals=cbar_tickvals, ticktext=cbar_ticktext,
            tickmode=("array" if cbar_tickvals is not None else "auto"),
        ),
        hovertemplate=(
            f"X: %{{x:.2f}} km<br>Y: %{{y:.2f}} km"
            f"<br>{hover_label}: %{{{hover_ref}:{hover_fmt}}}{hover_suffix}<extra></extra>"
        ),
        **heatmap_kwargs,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a2834")),
        height=860, margin=dict(l=60, r=15, t=55, b=55),
        dragmode="pan",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(
            title=dict(text="X (km · EPSG:32614)", font=dict(size=13, color=TXT)),
            range=[float(x_km[0]), float(x_km[-1])], constrain="domain",
            tickformat=".0f", ticksuffix=" km",
            showgrid=False, zeroline=False, mirror=False,
            showline=True, linecolor="#90a4ae", linewidth=1,
            tickfont=dict(size=12, color=TXT), ticks="outside", ticklen=4,
        ),
        yaxis=dict(
            title=dict(text="Y (km · EPSG:32614)", font=dict(size=13, color=TXT)),
            range=[float(y_km[-1]), float(y_km[0])],
            scaleanchor="x", scaleratio=1,
            tickformat=".0f", ticksuffix=" km",
            showgrid=False, zeroline=False, mirror=False,
            showline=True, linecolor="#90a4ae", linewidth=1,
            tickfont=dict(size=12, color=TXT), ticks="outside", ticklen=4,
        ),
    )
    return fig


def heatmap_h(grid, x_ax, y_ax, title, cmap=CMAP_H):
    """Mapa de calado H con escala logarítmica. Ticks en m originales."""
    log_grid, v_min, v_max, tvals, ttexts = _log_h_setup(grid)
    return make_heatmap(log_grid, x_ax, y_ax, title, cmap, "H (m)",
                        zmin=v_min, zmax=v_max,
                        cbar_tickvals=tvals, cbar_ticktext=ttexts,
                        hover_fmt=".3f",
                        hover_grid=grid,
                        hover_label="H",
                        hover_unit="m",
                        zsmooth=False)


def heatmap_discrete(grid, x_ax, y_ax, title, bands, zmax, unit,
                     hover_fmt=".0f", hover_label="Valor", hover_unit=None):
    """Mapa con bandas de color discretas. bands = [(val_inicio, color), ...]"""
    zmin = bands[0][0] if bands else 0.0
    cmap = _discrete_cmap(bands, zmin, zmax)
    # ticks en el centro de cada banda
    tvals = [((bands[i][0] + (bands[i+1][0] if i+1 < len(bands) else zmax)) / 2)
             for i in range(len(bands))]
    # texto formateado según tipo
    if "h" in unit.lower():
        ttexts = [f"{int(round(bands[i][0]))}-{int(round(bands[i+1][0] if i+1<len(bands) else zmax))} h"
                  for i in range(len(bands))]
    else:
        ttexts = [f"{bands[i][0]:.1f}x-{(bands[i+1][0] if i+1<len(bands) else zmax):.1f}x"
                  for i in range(len(bands))]
    return make_heatmap(grid, x_ax, y_ax, title, cmap, unit,
                        zmin=zmin, zmax=zmax,
                        cbar_tickvals=tvals, cbar_ticktext=ttexts,
                        hover_fmt=hover_fmt,
                        hover_label=hover_label,
                        hover_unit=hover_unit,
                        zsmooth=False)


def _smart_yrange(y_all: list) -> tuple:
    """Devuelve (fill, fillcolor, yrange) para gráficas temporales.
    Si los datos no pasan por 0 y la variación relativa es >5%, hace zoom para ver la variación.
    """
    vals = [float(v) for v in y_all if v is not None and np.isfinite(float(v))]
    if not vals:
        return "tozeroy", "rgba(25,118,210,0.15)", None
    y_min, y_max = min(vals), max(vals)
    span = y_max - y_min
    tight = y_min > 0 and y_max > 0 and span / y_max > 0.05
    if tight:
        pad = span * 0.12
        return None, None, [max(0.0, y_min - pad), y_max + pad]
    return "tozeroy", "rgba(25,118,210,0.15)", None


def make_line(x, y, title, xlabel, ylabel, color="#1976d2", extras=None):
    TXT = "#1a2834"
    all_y = list(y) + ([v for ex in extras for v in ex["y"]] if extras else [])
    fill, fillcolor, yrange = _smart_yrange(all_y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color),
        fill=fill, fillcolor=fillcolor,
        name=ylabel,
    ))
    if extras:
        for ex in extras:
            fig.add_trace(go.Scatter(
                x=ex["x"], y=ex["y"], mode="lines+markers",
                line=dict(color=ex["color"], width=3),
                marker=dict(size=7),
                name=ex["name"],
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0d47a1")),
        xaxis_title=dict(text=xlabel, font=dict(size=13, color=TXT)),
        yaxis_title=dict(text=ylabel, font=dict(size=13, color=TXT)),
        height=500, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=10, t=50, b=40),
        font=dict(color=TXT),
        legend=dict(font=dict(size=12, color=TXT)),
        xaxis=dict(
            tickfont=dict(size=12, color=TXT),
            showline=True, linecolor="#90a4ae", linewidth=1,
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=12, color=TXT),
            showline=True, linecolor="#90a4ae", linewidth=1,
            showgrid=True, gridcolor="#d9e2ec", gridwidth=1,
            zeroline=False,
            **({"range": yrange} if yrange else {}),
        ),
    )
    return fig


def make_bar(cats, vals, title, ylabel, colors):
    TXT = "#1a2834"
    fig = go.Figure(go.Bar(x=cats, y=vals, marker_color=colors, text=[f"{v:.1f}" for v in vals],
                           textposition="outside"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0d47a1")),
        yaxis_title=dict(text=ylabel, font=dict(size=13, color=TXT)),
        height=460,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=10, t=50, b=40),
        font=dict(color=TXT),
        xaxis=dict(
            tickfont=dict(size=12, color=TXT),
            showline=True, linecolor="#90a4ae", linewidth=1,
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=12, color=TXT),
            showline=True, linecolor="#90a4ae", linewidth=1,
            showgrid=True, gridcolor="#d9e2ec", gridwidth=1,
            zeroline=False,
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING RUNNERS — grid acumulador, memoria constante
# ═══════════════════════════════════════════════════════════════════════════════

def _grid_for(meta, bbox):
    r_min, r_max, c_min, c_max   = compute_window(meta, bbox)
    scale, gr, gc, x_ax, y_ax    = grid_info(meta, r_min, r_max, c_min, c_max)
    return r_min, r_max, c_min, c_max, scale, gr, gc, x_ax, y_ax


def _step_coords(rows, cols, r_min, c_min, scale, gr, gc):
    r_loc = (rows - r_min).astype(np.int32)
    c_loc = (cols - c_min).astype(np.int32)
    r_s   = (r_loc * scale).astype(np.int32).clip(0, gr - 1)
    c_s   = (c_loc * scale).astype(np.int32).clip(0, gc - 1)
    return r_s, c_s


def run_single_step(meta, attrs, hora_idx, bbox=None, cond=None):
    """Carga 1 paso temporal y devuelve (res, window info)."""
    r_min, r_max, c_min, c_max, scale, gr, gc, x_ax, y_ax = _grid_for(meta, bbox)
    with tiledb.open(meta["uri"], mode="r") as A:
        res = A.query(attrs=attrs, cond=cond)[hora_idx, r_min:r_max+1, c_min:c_max+1]
    return res, (r_min, c_min, scale, gr, gc, x_ax, y_ax)


def stream_heatmap_single(meta, hora_idx, attrs, build_fn, bbox, progress=None, cond=None):
    """Consulta de 1 paso con filtro; devuelve grid downsampleado."""
    res, (r_min, c_min, scale, gr, gc, x_ax, y_ax) = run_single_step(meta, attrs, hora_idx, bbox, cond)
    n_orig = len(res["H"]) if "H" in res else len(next(iter(res.values())))
    if n_orig == 0:
        grid = np.full((gr, gc), np.nan, dtype=np.float32)
        return grid, x_ax, y_ax, {"n": 0}

    rows, cols = res["row"], res["col"]
    r_s, c_s   = _step_coords(rows, cols, r_min, c_min, scale, gr, gc)
    grid, mask, stats = build_fn(res, r_s, c_s, gr, gc)
    return grid, x_ax, y_ax, stats


def stream_scalar_multi(meta, attrs, compute_fn, bbox, progress=None, label="Procesando"):
    """
    Recorre los N pasos y llama compute_fn(res, t_idx) devolviendo un dict escalar.
    Devuelve lista de dicts, uno por paso. Memoria constante (1 paso cargado).
    """
    r_min, r_max, c_min, c_max = compute_window(meta, bbox)
    results = []
    n_steps = meta["n_steps"]
    with tiledb.open(meta["uri"], mode="r") as A:
        for t_idx in range(n_steps):
            if progress:
                progress.progress((t_idx + 1) / n_steps, text=f"{label} {t_idx+1}/{n_steps}")
            res = A.query(attrs=attrs)[t_idx, r_min:r_max+1, c_min:c_max+1]
            results.append(compute_fn(res, t_idx))
            del res
    return results


def stream_spatial_multi(meta, attrs, step_processor, bbox, progress=None, label="Procesando", cond=None):
    """
    Recorre los N pasos y llama step_processor(res, r_s, c_s, t_idx, grids) por cada paso.
    `grids` es un dict compartido que se va actualizando. Memoria constante.
    """
    r_min, r_max, c_min, c_max, scale, gr, gc, x_ax, y_ax = _grid_for(meta, bbox)
    grids = {"gr": gr, "gc": gc, "scale": scale, "cellsize": meta["cellsize"]}
    n_steps = meta["n_steps"]

    with tiledb.open(meta["uri"], mode="r") as A:
        for t_idx in range(n_steps):
            if progress:
                progress.progress((t_idx + 1) / n_steps, text=f"{label} {t_idx+1}/{n_steps}")
            res  = A.query(attrs=attrs, cond=cond)[t_idx, r_min:r_max+1, c_min:c_max+1]
            if len(res[attrs[0]]) == 0:
                del res
                continue
            rows, cols = res["row"], res["col"]
            r_s, c_s   = _step_coords(rows, cols, r_min, c_min, scale, gr, gc)
            step_processor(res, r_s, c_s, t_idx, grids)
            del res, rows, cols, r_s, c_s

    return grids, x_ax, y_ax


# ── Q3: velocidad en zona ────────────────────────────────────────────────────
def q3_velocidad(meta, hora, bbox, progress=None):
    def _build(res, r_s, c_s, gr, gc):
        H     = res["H"]
        Q_mod = np.sqrt(res["QX"]**2 + res["QY"]**2)
        V     = Q_mod / np.maximum(H, H_WET)
        grid_v = np.full((gr, gc), np.nan, dtype=np.float32)
        grid_c = np.zeros((gr, gc), dtype=np.int32)
        grid_s = np.zeros((gr, gc), dtype=np.float32)
        np.add.at(grid_c, (r_s, c_s), 1)
        np.add.at(grid_s, (r_s, c_s), V.astype(np.float32))
        mask = grid_c > 0
        grid_v[mask] = grid_s[mask] / grid_c[mask]
        return grid_v, mask, {
            "n": int(len(H)),
            "V_max":   float(V.max())  if len(V) else 0.0,
            "V_media": float(V.mean()) if len(V) else 0.0,
        }
    hora_idx = ca.hora_a_t(hora)
    return stream_heatmap_single(meta, hora_idx, ["H", "QX", "QY"], _build, bbox)


# ── Q4/Q6: zonas inundadas / sobre umbral ────────────────────────────────────
def q_umbral_h(meta, hora, umbral, bbox, cmap=None):
    def _build(res, r_s, c_s, gr, gc):
        H    = res["H"]
        mask = H >= umbral
        n    = int(mask.sum())
        grid_h = np.full((gr, gc), -np.inf, dtype=np.float32)
        if n > 0:
            np.maximum.at(grid_h, (r_s[mask], c_s[mask]), H[mask].astype(np.float32))
        grid_h[~np.isfinite(grid_h)] = np.nan
        return grid_h, mask, {
            "n":        n,
            "area_km2": n * _cell_km2(meta),
        }
    hora_idx = ca.hora_a_t(hora)
    cond = f"H >= {umbral}" if umbral > H_WET else None
    return stream_heatmap_single(meta, hora_idx, ["H"], _build, bbox, cond=cond)


# ── Q7: hora de llegada del frente ───────────────────────────────────────────
def q7_hora_llegada(meta, bbox, progress=None):
    n_steps = meta["n_steps"]

    def _proc(res, r_s, c_s, t_idx, grids):
        g = grids.setdefault("lleg", np.full((grids["gr"], grids["gc"]), n_steps, np.int16))
        mask = res["H"] >= H_WET
        if mask.any():
            np.minimum.at(g, (r_s[mask], c_s[mask]), t_idx)

    grids, x_ax, y_ax = stream_spatial_multi(meta, ["H"], _proc, bbox, progress, "Q7")
    g = grids.get("lleg", np.full((grids["gr"], grids["gc"]), n_steps, np.int16))
    hora_lleg = np.where(g < n_steps, (g + 1) * STEP_H, np.nan).astype(np.float32)
    valid     = np.isfinite(hora_lleg)
    stats = {
        "min": float(hora_lleg[valid].min()) if valid.any() else 0,
        "max": float(hora_lleg[valid].max()) if valid.any() else 0,
    }
    return hora_lleg, x_ax, y_ax, stats


# ── Q8: duración de inundación ───────────────────────────────────────────────
def q8_duracion(meta, umbral, bbox, progress=None):
    def _proc(res, r_s, c_s, t_idx, grids):
        g = grids.setdefault("cnt", np.zeros((grids["gr"], grids["gc"]), np.int16))
        mask = res["H"] >= umbral
        if mask.any():
            # Presencia por píxel en este paso: +1 por píxel inundado, no por celda
            # cruda (con submuestreo varias celdas caen en el mismo píxel del grid).
            step = np.zeros((grids["gr"], grids["gc"]), np.uint8)
            step[r_s[mask], c_s[mask]] = 1
            g += step
    cond = f"H >= {umbral}" if umbral > H_WET else None
    grids, x_ax, y_ax = stream_spatial_multi(meta, ["H"], _proc, bbox, progress, "Q8", cond=cond)
    g = grids.get("cnt", np.zeros((grids["gr"], grids["gc"]), np.int16))
    dur = np.where(g > 0, g.astype(np.float32) * STEP_H, np.nan)
    valid = g > 0
    stats = {
        "media": float(dur[valid].mean()) if valid.any() else 0,
        "max":   float(dur[valid].max())  if valid.any() else 0,
    }
    return dur, x_ax, y_ax, stats


# ── Q9: hora de calado máximo ────────────────────────────────────────────────
def q9_hora_max(meta, bbox, progress=None):
    def _proc(res, r_s, c_s, t_idx, grids):
        gr, gc = grids["gr"], grids["gc"]
        g_hmax = grids.setdefault("hmax", np.zeros((gr, gc), np.float32))
        g_tmax = grids.setdefault("tmax", np.full((gr, gc), -1, np.int16))
        # H máximo de este paso en cada celda del grid
        H_step = np.zeros((gr, gc), np.float32)
        np.maximum.at(H_step, (r_s, c_s), res["H"].astype(np.float32))
        upd    = H_step > g_hmax
        g_hmax[upd] = H_step[upd]
        g_tmax[upd] = t_idx
    grids, x_ax, y_ax = stream_spatial_multi(meta, ["H"], _proc, bbox, progress, "Q9")
    gr, gc = grids["gr"], grids["gc"]
    g_hmax = grids.get("hmax", np.zeros((gr, gc), np.float32))
    g_tmax = grids.get("tmax", np.full((gr, gc), -1, np.int16))
    hora_max = np.where(g_tmax >= 0, (g_tmax + 1) * STEP_H, np.nan).astype(np.float32)
    valid = g_tmax >= 0
    horas_validas = hora_max[valid]
    stats = {
        "n":             int(valid.sum()),
        "hora_pico_min": int(horas_validas.min()) if valid.any() else 0,
        "hora_pico_max": int(horas_validas.max()) if valid.any() else 0,
        "H_max":         float(g_hmax.max())      if valid.any() else 0,
    }
    return hora_max, x_ax, y_ax, stats


# ── Q10a/b/c: intensidad de peligro (ratio vs umbral) ────────────────────────
def q10_intensity_grid(meta, hora, tipo, bbox):
    """
    Devuelve grid con intensidad = max(H/H_thresh, Q_mod/Q_thresh).
    Valor 1.0 = justo en el umbral; 2.0 = doble del umbral; etc.
    Solo celdas que superan algún umbral (peligrosas).
    """
    hora_idx = ca.hora_a_t(hora)
    r_min, r_max, c_min, c_max, scale, gr, gc, x_ax, y_ax = _grid_for(meta, bbox)
    with tiledb.open(meta["uri"], mode="r") as A:
        res = A.query(attrs=["H", "QX", "QY"])[hora_idx, r_min:r_max+1, c_min:c_max+1]
    H     = res["H"]
    Q_mod = np.sqrt(res["QX"]**2 + res["QY"]**2)
    rows, cols = res["row"], res["col"]
    r_s, c_s   = _step_coords(rows, cols, r_min, c_min, scale, gr, gc)

    if tipo == "adultos":
        H_t, Q_t = H_ADULTO, HV_ADULTO
    elif tipo == "ninos":
        H_t, Q_t = H_NINO, HV_NINO
    else:
        H_t, Q_t = H_VEHICULO, 999.0

    intensity = np.maximum(H / H_t, Q_mod / Q_t)
    mask = intensity >= 1.0   # solo celdas peligrosas

    grid_i = np.full((gr, gc), np.nan, np.float32)
    grid_c = np.zeros((gr, gc), np.int32)
    grid_s = np.zeros((gr, gc), np.float32)
    if mask.any():
        np.add.at(grid_c, (r_s[mask], c_s[mask]), 1)
        np.add.at(grid_s, (r_s[mask], c_s[mask]), intensity[mask].astype(np.float32))
        has = grid_c > 0
        grid_i[has] = grid_s[has] / grid_c[has]

    n = int(mask.sum())
    return grid_i, x_ax, y_ax, {
        "n": n, "area_km2": n * _cell_km2(meta),
        "intensity_max": float(intensity[mask].max()) if mask.any() else 0.0,
        "intensity_med": float(intensity[mask].mean()) if mask.any() else 0.0,
    }


# ── Q10a/b/c: peligrosidad (1 paso) ──────────────────────────────────────────
def q10_peligrosidad(meta, hora, tipo, bbox):
    """tipo: 'adultos' | 'ninos' | 'vehiculos_ligeros'"""
    def _build(res, r_s, c_s, gr, gc):
        H     = res["H"]
        Q_mod = np.sqrt(res["QX"]**2 + res["QY"]**2)
        if tipo == "adultos":
            mask = (H > H_ADULTO) | (Q_mod > HV_ADULTO)
        elif tipo == "ninos":
            mask = (H > H_NINO)   | (Q_mod > HV_NINO)
        else:
            mask = H > H_VEHICULO
        n = int(mask.sum())
        grid_h = np.full((gr, gc), np.nan, dtype=np.float32)
        grid_c = np.zeros((gr, gc), dtype=np.int32)
        grid_s = np.zeros((gr, gc), dtype=np.float32)
        if n > 0:
            np.add.at(grid_c, (r_s[mask], c_s[mask]), 1)
            np.add.at(grid_s, (r_s[mask], c_s[mask]), H[mask].astype(np.float32))
            has = grid_c > 0
            grid_h[has] = grid_s[has] / grid_c[has]
        return grid_h, mask, {
            "n":        n,
            "area_km2": n * _cell_km2(meta),
        }
    hora_idx = ca.hora_a_t(hora)
    return stream_heatmap_single(meta, hora_idx, ["H", "QX", "QY"], _build, bbox)


# ── Xia et al.: inestabilidad personas/vehículos ─────────────────────────────
def q10_xia(meta, hora, tipo, bbox):
    """Mapa de inestabilidad según Xia et al. (2014/2022).
    tipo: 'adultos' | 'ninos' | 'vehiculos'
    Grid de salida: NaN=seco | 0=seguro | 1=riesgo moderado | 2=riesgo alto."""
    cell2 = _cell_km2(meta)
    def _build(res, r_s, c_s, gr, gc):
        H = res["H"]
        Q_mod = np.sqrt(res["QX"]**2 + res["QY"]**2)
        if tipo == "vehiculos":
            risk_arr = ca.xia_risk_vehiculos(H, Q_mod)
        else:
            risk_arr = ca.xia_risk_personas(H, Q_mod, tipo)
        risk_grid = np.full((gr, gc), -1, dtype=np.int8)
        np.maximum.at(risk_grid, (r_s, c_s), risk_arr.view(np.int8))
        grid = np.where(risk_grid >= 0, risk_grid.astype(np.float32), np.nan)
        has = risk_grid >= 0
        # Conteos sobre las celdas crudas (el grid está submuestreado y, además,
        # np.maximum.at lo colapsa al riesgo máximo del píxel, sesgando las áreas).
        n_seg  = int((risk_arr == 0).sum())
        n_mod  = int((risk_arr == 1).sum())
        n_high = int((risk_arr == 2).sum())
        return grid, has, {
            "n_seguro": n_seg, "n_moderado": n_mod,  "n_alto": n_high,
            "area_seguro_km2":   n_seg  * cell2,
            "area_moderado_km2": n_mod  * cell2,
            "area_alto_km2":     n_high * cell2,
        }
    hora_idx = ca.hora_a_t(hora)
    return stream_heatmap_single(meta, hora_idx, ["H", "QX", "QY"], _build, bbox)


# ── Q18: zona de graves daños (RD 9/2008) ────────────────────────────────────
def q18_graves_danos(meta, hora, bbox):
    """Zona de graves daños: H>1m OR V>1m/s OR H·V>0.5m²/s (Real Decreto 9/2008)."""
    cell2 = _cell_km2(meta)
    def _build(res, r_s, c_s, gr, gc):
        H = res["H"]
        Q_mod = np.sqrt(res["QX"]**2 + res["QY"]**2)
        mask = ca.graves_danos_mask(H, Q_mod)
        grid = np.full((gr, gc), np.nan, np.float32)
        if mask.any():
            np.maximum.at(grid, (r_s[mask], c_s[mask]), H[mask].astype(np.float32))
        return grid, mask, {"n": int(mask.sum()), "area_km2": int(mask.sum()) * cell2}
    hora_idx = ca.hora_a_t(hora)
    return stream_heatmap_single(meta, hora_idx, ["H", "QX", "QY"], _build, bbox)


# ── Q11: ventana vehículos emergencia (multi-step) ───────────────────────────
def q11_ventana_emergencia(meta, bbox, progress=None):
    n_steps = meta["n_steps"]
    def _proc(res, r_s, c_s, t_idx, grids):
        gr, gc = grids["gr"], grids["gc"]
        g_min = grids.setdefault("tmin", np.full((gr, gc), n_steps, np.int16))
        g_max = grids.setdefault("tmax", np.full((gr, gc), -1,       np.int16))
        mask = res["H"] < H_EMERGENCIA  # sparse ya filtra H >= H_WET
        if mask.any():
            np.minimum.at(g_min, (r_s[mask], c_s[mask]), t_idx)
            np.maximum.at(g_max, (r_s[mask], c_s[mask]), t_idx)
    grids, x_ax, y_ax = stream_spatial_multi(meta, ["H"], _proc, bbox, progress, "Q11")
    gr, gc = grids["gr"], grids["gc"]
    g_min = grids.get("tmin", np.full((gr, gc), n_steps, np.int16))
    g_max = grids.get("tmax", np.full((gr, gc), -1, np.int16))
    valid = g_max >= 0
    ventana = np.where(valid, (g_max - g_min + 1).astype(np.float32) * STEP_H, np.nan)
    stats = {
        "n":     int(valid.sum()),
        "media": float(ventana[valid].mean()) if valid.any() else 0,
        "min":   float(ventana[valid].min())  if valid.any() else 0,
        "max":   float(ventana[valid].max())  if valid.any() else 0,
    }
    return ventana, x_ax, y_ax, stats


# ── Q11b: transitabilidad emergencia (1 paso) ────────────────────────────────
def q11b_transitabilidad(meta, hora, bbox):
    def _build(res, r_s, c_s, gr, gc):
        H    = res["H"]
        mask = (H > 0) & (H < H_EMERGENCIA)
        n    = int(mask.sum())
        grid_h = np.full((gr, gc), np.nan, dtype=np.float32)
        grid_c = np.zeros((gr, gc), dtype=np.int32)
        grid_s = np.zeros((gr, gc), dtype=np.float32)
        if n > 0:
            np.add.at(grid_c, (r_s[mask], c_s[mask]), 1)
            np.add.at(grid_s, (r_s[mask], c_s[mask]), H[mask].astype(np.float32))
            has = grid_c > 0
            grid_h[has] = grid_s[has] / grid_c[has]
        return grid_h, mask, {
            "n":        n,
            "area_km2": n * _cell_km2(meta),
        }
    hora_idx = ca.hora_a_t(hora)
    return stream_heatmap_single(meta, hora_idx, ["H"], _build, bbox,
                                 cond=f"H < {H_EMERGENCIA}")


# ── Q15: área por nivel de peligro (1 paso, conteo) ──────────────────────────
def q15_area_peligro(meta, hora, bbox):
    hora_idx = ca.hora_a_t(hora)
    r_min, r_max, c_min, c_max = compute_window(meta, bbox)
    with tiledb.open(meta["uri"], mode="r") as A:
        res = A.query(attrs=["H"])[hora_idx, r_min:r_max+1, c_min:c_max+1]
    cell2 = _cell_km2(meta)
    v_c, a_c, r_c = ca.russo_traffic_light_counts(res["H"])
    v, a, r = v_c * cell2, a_c * cell2, r_c * cell2
    return {"verde": v, "amarillo": a, "rojo": r, "total": v + a + r}


def q17_area_niveles(meta, hora, bbox):
    """Q17 — km² en los 5 niveles disjuntos de peligrosidad, sobre H crudo de la
    ventana (no sobre el grid de pintado, que está submuestreado en bbox grandes)."""
    hora_idx = ca.hora_a_t(hora)
    r_min, r_max, c_min, c_max = compute_window(meta, bbox)
    with tiledb.open(meta["uri"], mode="r") as A:
        res = A.query(attrs=["H"])[hora_idx, r_min:r_max+1, c_min:c_max+1]
    cell2 = _cell_km2(meta)
    return tuple(c * cell2 for c in ca.russo_cinco_niveles(res["H"]))


# ── Q16: ventana de evacuación ───────────────────────────────────────────────
def q16_evacuacion(meta, umbral_h, umbral_q, bbox, progress=None):
    n_steps = meta["n_steps"]
    def _proc(res, r_s, c_s, t_idx, grids):
        gr, gc = grids["gr"], grids["gc"]
        g_lleg = grids.setdefault("lleg", np.full((gr, gc), n_steps, np.int16))
        g_crit = grids.setdefault("crit", np.full((gr, gc), n_steps, np.int16))
        H     = res["H"]
        Q_mod = np.sqrt(res["QX"]**2 + res["QY"]**2)
        m_wet = H >= H_WET
        if m_wet.any():
            np.minimum.at(g_lleg, (r_s[m_wet], c_s[m_wet]), t_idx)
        m_cri = (H > umbral_h) | (Q_mod > umbral_q)
        if m_cri.any():
            np.minimum.at(g_crit, (r_s[m_cri], c_s[m_cri]), t_idx)
    grids, x_ax, y_ax = stream_spatial_multi(meta, ["H", "QX", "QY"], _proc, bbox, progress, "Q16")
    gr, gc = grids["gr"], grids["gc"]
    g_lleg = grids.get("lleg", np.full((gr, gc), n_steps, np.int16))
    g_crit = grids.get("crit", np.full((gr, gc), n_steps, np.int16))
    wet = g_lleg < n_steps
    ventana = np.where(
        wet,
        np.where(g_crit == n_steps,
                 (n_steps - g_lleg) * STEP_H,            # nunca peligroso
                 np.maximum(0, g_crit - g_lleg) * STEP_H),
        np.nan
    ).astype(np.float32)
    nunca      = int(((g_crit == n_steps) & wet).sum())
    ya_peli    = int(((g_crit <= g_lleg)  & wet).sum())
    vv         = ventana[wet]
    stats = {
        "n":       int(wet.sum()),
        "nunca":   nunca,
        "ya_peli": ya_peli,
        "media":   float(vv.mean()) if vv.size else 0,
        "min":     float(vv.min())  if vv.size else 0,
    }
    return ventana, x_ax, y_ax, stats


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNERS TEMPORALES — "Todas las horas"
# ═══════════════════════════════════════════════════════════════════════════════

def q3_temporal(meta, bbox, progress=None):
    def _fn(res, t):
        H  = res["H"]
        Q  = np.sqrt(res["QX"]**2 + res["QY"]**2)
        V  = Q / np.maximum(H, H_WET) if len(H) else np.array([])
        return {
            "V_max":   float(V.max())  if len(V) else 0.0,
            "V_media": float(V.mean()) if len(V) else 0.0,
            "n":       int(len(H)),
        }
    out = stream_scalar_multi(meta, ["H", "QX", "QY"], _fn, bbox, progress, "Q3")
    horas = _horas(meta)
    return {
        "horas":   horas,
        "V_max":   [r["V_max"]   for r in out],
        "V_media": [r["V_media"] for r in out],
    }


def q10_peligrosidad_temporal(meta, tipo, bbox, progress=None):
    cell2 = _cell_km2(meta)
    def _fn(res, t):
        H = res["H"]; Q = np.sqrt(res["QX"]**2 + res["QY"]**2)
        if tipo == "adultos":
            mask = (H > H_ADULTO) | (Q > HV_ADULTO)
        elif tipo == "ninos":
            mask = (H > H_NINO)   | (Q > HV_NINO)
        else:
            mask = H > H_VEHICULO
        n = int(mask.sum())
        return {"area_km2": n * cell2, "n": n}
    out = stream_scalar_multi(meta, ["H", "QX", "QY"], _fn, bbox, progress, f"Q10 ({tipo})")
    horas = _horas(meta)
    return {"horas": horas, "area_km2": [r["area_km2"] for r in out]}


def q11b_temporal(meta, bbox, progress=None):
    cell2 = _cell_km2(meta)
    def _fn(res, t):
        H = res["H"]
        mask = (H > 0) & (H < H_EMERGENCIA)
        return {"area_km2": int(mask.sum()) * cell2}
    out = stream_scalar_multi(meta, ["H"], _fn, bbox, progress, "Q11b")
    horas = _horas(meta)
    return {"horas": horas, "area_km2": [r["area_km2"] for r in out]}


def q14_temporal(meta, bbox, progress=None):
    def _fn(res, t):
        H = res["H"]
        if len(H) == 0:
            return {"H_media": 0.0, "H_max": 0.0, "H_P50": 0.0, "H_P95": 0.0, "n": 0}
        return {
            "H_media": float(H.mean()),
            "H_max":   float(H.max()),
            "H_P50":   float(np.percentile(H, 50)),
            "H_P95":   float(np.percentile(H, 95)),
            "n":       int(len(H)),
        }
    out = stream_scalar_multi(meta, ["H"], _fn, bbox, progress, "Q14")
    horas = _horas(meta)
    return {
        "horas":   horas,
        "H_media": [r["H_media"] for r in out],
        "H_max":   [r["H_max"]   for r in out],
        "H_P50":   [r["H_P50"]   for r in out],
        "H_P95":   [r["H_P95"]   for r in out],
    }


def q15_temporal(meta, bbox, progress=None):
    cell2 = _cell_km2(meta)
    def _fn(res, t):
        v, a, r = ca.russo_traffic_light_counts(res["H"])
        return {"verde": v * cell2, "amarillo": a * cell2, "rojo": r * cell2}
    out = stream_scalar_multi(meta, ["H"], _fn, bbox, progress, "Q15")
    horas = _horas(meta)
    return {
        "horas":    horas,
        "verde":    [r["verde"]    for r in out],
        "amarillo": [r["amarillo"] for r in out],
        "rojo":     [r["rojo"]     for r in out],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CACHÉ de queries pesadas (bbox debe ser tuple|None, no list, para ser hashable)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=None, max_entries=24, persist="disk")
def _c_q5(dataset: str, bbox):
    m = get_meta(dataset); setup_ca(m)
    return ca.evolucion_extension(bbox=bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24, persist="disk")
def _c_q12(dataset: str, bbox):
    m = get_meta(dataset); setup_ca(m)
    return ca.area_inundada_por_hora(bbox=bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24, persist="disk")
def _c_q13(dataset: str, bbox):
    m = get_meta(dataset); setup_ca(m)
    return ca.volumen_por_hora(bbox=bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24, persist="disk")
def _c_q7(dataset: str, bbox):
    return q7_hora_llegada(get_meta(dataset), bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24, persist="disk")
def _c_q8(dataset: str, umbral: float, bbox):
    return q8_duracion(get_meta(dataset), umbral, bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24, persist="disk")
def _c_q9(dataset: str, bbox):
    return q9_hora_max(get_meta(dataset), bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24, persist="disk")
def _c_q11(dataset: str, bbox):
    return q11_ventana_emergencia(get_meta(dataset), bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24, persist="disk")
def _c_q16(dataset: str, uh: float, uq: float, bbox):
    return q16_evacuacion(get_meta(dataset), uh, uq, bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24, persist="disk")
def _c_q10i(dataset: str, hora: int, tipo: str, bbox):
    return q10_intensity_grid(get_meta(dataset), hora, tipo, bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24)
def _c_q3t(dataset: str, bbox):
    return q3_temporal(get_meta(dataset), bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24)
def _c_q10t(dataset: str, tipo: str, bbox):
    return q10_peligrosidad_temporal(get_meta(dataset), tipo, bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24)
def _c_q3(dataset: str, hora: int, bbox):
    return q3_velocidad(get_meta(dataset), hora, bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24)
def _c_q10p(dataset: str, hora: int, tipo: str, bbox):
    return q10_peligrosidad(get_meta(dataset), hora, tipo, bbox)

@st.cache_data(show_spinner=False, ttl=None, max_entries=24)
def _c_q11b(dataset: str, hora: int, bbox):
    return q11b_transitabilidad(get_meta(dataset), hora, bbox)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTAR A GEOTIFF
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=None, max_entries=24)
def grid_to_geotiff_bytes(grid: np.ndarray, x_ax: np.ndarray, y_ax: np.ndarray) -> bytes:
    import io, rasterio
    from rasterio.transform import from_bounds
    h, w = grid.shape
    transform = from_bounds(float(x_ax[0]), float(y_ax[-1]),
                            float(x_ax[-1]), float(y_ax[0]), w, h)
    data = grid.astype(np.float32)
    data[~np.isfinite(data)] = -9999.0
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="float32", crs="EPSG:32614",
                       transform=transform, nodata=-9999.0,
                       compress="deflate") as dst:
        dst.write(data, 1)
    return buf.getvalue()


def _downsample_grid(grid: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbour downsampling que preserva NaN."""
    rows = np.clip((np.arange(new_h) * grid.shape[0] / new_h).astype(int), 0, grid.shape[0] - 1)
    cols = np.clip((np.arange(new_w) * grid.shape[1] / new_w).astype(int), 0, grid.shape[1] - 1)
    return grid[np.ix_(rows, cols)]


_EXPORT_PRESETS = [400, 800, 1200, 1800, 2400, 3200]


def download_geotiff_button(grid, x_ax, y_ax, qid, dataset, hora=None, ds_label=""):
    h, w        = grid.shape
    current_max = max(h, w)
    m_per_px    = (float(x_ax[-1]) - float(x_ax[0])) / w

    def _label(res):
        scale   = res / current_max
        out_mpp = m_per_px / scale
        out_w   = max(1, int(w * scale))
        out_h   = max(1, int(h * scale))
        tag     = " ← actual" if res == current_max else ""
        return f"{out_w}×{out_h} px — {out_mpp:.1f} m/px{tag}"

    suffix = f"_h{hora}" if hora else ""
    fname  = f"{qid}_{dataset}{suffix}.tif"
    key    = f"export_res_{qid}_{dataset}_{hora}"

    default_idx = min(range(len(_EXPORT_PRESETS)),
                      key=lambda i: abs(_EXPORT_PRESETS[i] - current_max))

    c1, c2 = st.columns([1, 2])
    with c1:
        chosen = st.selectbox("📐", _EXPORT_PRESETS, index=default_idx,
                              format_func=_label, label_visibility="collapsed", key=key)
    with c2:
        if chosen > current_max:
            if st.button(_t("download_geotiff"), key=f"rerun_{key}"):
                st.session_state["map_res_slider"] = min(chosen, 3200)
                st.rerun()
        else:
            try:
                if chosen == current_max:
                    export_grid, export_x, export_y = grid, x_ax, y_ax
                else:
                    scale       = chosen / current_max
                    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                    export_grid = _downsample_grid(grid, new_h, new_w)
                    export_x    = np.linspace(x_ax[0], x_ax[-1], new_w)
                    export_y    = np.linspace(y_ax[0], y_ax[-1], new_h)
                data = grid_to_geotiff_bytes(export_grid, export_x, export_y)
                prefix = f"{ds_label} · " if ds_label else ""
                st.download_button(f"⬇ {prefix}{_t('download_geotiff')} — {_label(chosen)}",
                                   data, fname, mime="image/tiff")
            except Exception as e:
                st.caption(f"{_t('geotiff_unavailable')}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN DEL DATASET (cacheado, sin ejecutar queries completas)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=None, max_entries=24)
def dataset_summary(dataset: str) -> dict:
    """Lectura rápida del paso 10 (hora 60) como referencia."""
    meta = get_meta(dataset)
    uri  = meta["uri"]
    ref_t = min(9, meta["n_steps"] - 1)
    try:
        with tiledb.open(uri, mode="r") as A:
            res  = A.query(attrs=["H"])[ref_t, :, :]
            n    = len(res["H"])
            hmax = float(res["H"].max()) if n > 0 else 0.0
            del res
    except Exception:
        n, hmax = 0, 0.0
    cs       = meta["cellsize"]
    total_px = meta["nrows"] * meta["ncols"]
    return {
        "n_steps":    meta["n_steps"],
        "dur_h":      meta["n_steps"] * STEP_H,
        "nrows":      meta["nrows"],
        "ncols":      meta["ncols"],
        "cellsize":   cs,
        "extent_km2": total_px * cs**2 / 1e6,
        "n_wet_ref":  n,
        "area_ref_km2": n * cs**2 / 1e6,
        "pct_wet":    n / total_px * 100,
        "h_max_ref":  hmax,
    }


def show_dataset_summary(dataset: str):
    """Panel de métricas básicas del dataset seleccionado."""
    with st.expander(f"{_t('dataset_summary')} — {_dataset_display(dataset)}", expanded=False):
        summ = dataset_summary(dataset)
        cs = summ["cellsize"]
        ext_km_x = summ["ncols"] * cs / 1000
        ext_km_y = summ["nrows"] * cs / 1000
        st.caption(
            f"{_t('cap_domain')} {summ['ncols']:,} × {summ['nrows']:,} {_t('cap_cells')} "
            f"({ext_km_x:.0f} × {ext_km_y:.0f} km · {summ['extent_km2']:,.0f} km²) · "
            f"{_t('cap_res')} {cs:.0f} m/{_t('cap_cell')} · "
            f"{summ['n_steps']} {_t('cap_steps')} {STEP_H} h = {summ['dur_h']} h {_t('cap_sim')} · "
            f"EPSG:32614 (UTM {_t('cap_zone')} 14N)"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(_t("summary_steps"), f"{summ['n_steps']} × {STEP_H} h = {summ['dur_h']} h")
        c2.metric(_t("summary_extent"), f"{summ['extent_km2']:.0f} km²")
        c3.metric(_t("summary_grid"), f"{cs:.0f} m/{_t('cap_cell')}")


def make_multi_line(horas, series, title, ylabel):
    """series: lista de dicts {name, y, color}"""
    TXT = "#1a2834"
    all_y = [v for s in series for v in s["y"]]
    _, _, yrange = _smart_yrange(all_y)
    fig = go.Figure()
    for s in series:
        fig.add_trace(go.Scatter(
            x=horas, y=s["y"], mode="lines+markers",
            line=dict(color=s["color"], width=3),
            marker=dict(size=7),
            name=s["name"],
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0d47a1")),
        xaxis_title=dict(text=_t("hour_axis"), font=dict(size=13, color=TXT)),
        yaxis_title=dict(text=ylabel, font=dict(size=13, color=TXT)),
        height=520, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=10, t=50, b=40),
        font=dict(color=TXT),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=12, color=TXT),
        ),
        xaxis=dict(
            tickfont=dict(size=12, color=TXT),
            showline=True, linecolor="#90a4ae", linewidth=1,
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=12, color=TXT),
            showline=True, linecolor="#90a4ae", linewidth=1,
            showgrid=True, gridcolor="#d9e2ec", gridwidth=1,
            zeroline=False,
            **({"range": yrange} if yrange else {}),
        ),
    )
    return fig


def make_stacked_area(horas, series, title, ylabel):
    """Gráfica apilada para Q15 verde/amarillo/rojo."""
    TXT = "#1a2834"
    fig = go.Figure()
    for s in series:
        fig.add_trace(go.Scatter(
            x=horas, y=s["y"], mode="lines",
            stackgroup="one", name=s["name"],
            line=dict(width=0.5, color=s["color"]),
            fillcolor=s["color"],
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0d47a1")),
        xaxis_title=dict(text=_t("hour_axis"), font=dict(size=13, color=TXT)),
        yaxis_title=dict(text=ylabel, font=dict(size=13, color=TXT)),
        height=520, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=10, t=50, b=40),
        font=dict(color=TXT),
        legend=dict(font=dict(size=12, color=TXT)),
        xaxis=dict(
            tickfont=dict(size=12, color=TXT),
            showline=True, linecolor="#90a4ae", linewidth=1,
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=12, color=TXT),
            showline=True, linecolor="#90a4ae", linewidth=1,
            showgrid=True, gridcolor="#d9e2ec", gridwidth=1,
            zeroline=False,
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

# ── Toggle de idioma (antes del título para que _t() use el lang correcto) ────
st.sidebar.markdown(f"""
<div style="display:flex;justify-content:center;padding:0.4rem 0 1rem">
    <img src="{_LOGO_URI}"
         style="width:160px;height:160px;object-fit:contain;
                filter:drop-shadow(0 4px 20px rgba(212,168,67,0.3))" alt="Spillway"/>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown("---")

datasets = discover_datasets()
if not datasets:
    st.error(f"{_t('no_datasets')} {BASE_URI}")
    st.stop()

# ── Modo de operación ─────────────────────────────────────────────────────────
_mode_opts = [_t("mode_single"), _t("mode_compare")]
modo_app = st.sidebar.radio(_t("mode_label"), _mode_opts,
                             horizontal=True, label_visibility="collapsed")
comparar = _mode_opts.index(modo_app) == 1

dataset = st.sidebar.selectbox(_t("dataset_a"), datasets, format_func=_dataset_display)
if st.session_state.get("_prev_dataset") != dataset:
    st.session_state.pop("x_pt", None)
    st.session_state.pop("y_pt", None)
    st.session_state["_prev_dataset"] = dataset
dataset2 = None
if comparar:
    resto = [d for d in datasets if d != dataset]
    if resto:
        dataset2 = st.sidebar.selectbox(_t("dataset_b"), resto, format_func=_dataset_display)
    else:
        st.sidebar.warning(_t("only_one_dataset"))
        comparar = False

meta    = get_meta(dataset)
setup_ca(meta)

horas_disp = _horas(meta)

GRUPOS_ES = {
    "🎯 A — Puntuales y velocidad": {
        "Q1 — Calado en punto":                ("q1",  "Calado H en una coordenada e instante."),
        "Q2 — Serie temporal en punto":        ("q2",  "Evolución de H en un punto durante las 120 h."),
        "Q3 — Velocidad en zona":              ("q3",  "Velocidad V = Q_mod / H en celdas húmedas."),
    },
    "💧 B — Umbrales espaciales": {
        "Q4 — Zonas inundadas":                ("q4",  "Celdas con H ≥ umbral en un instante."),
        "Q5 — Evolución de extensión":         ("q5",  "Curva temporal del área inundada."),
        "Q6 — Zonas sobre umbral":             ("q6",  "Celdas que superan un umbral dado."),
    },
    "⏱️ C — Indicadores temporales": {
        "Q7 — Hora llegada del frente":        ("q7",  "Primera hora en que el agua alcanza cada píxel."),
        "Q8 — Duración de inundación":         ("q8",  "Horas totales que cada píxel está inundado."),
        "Q9 — Hora de calado máximo":          ("q9",  "Instante en que cada píxel alcanza su H máximo."),
        "Q11 — Ventana vehículos emergencia":  ("q11", "Horas con acceso disponible para vehículos de rescate (H < 0.60 m)."),
    },
    "⚠️ D — Peligrosidad Russo et al. (2013)": {
        "Q10a — Inestabilidad adultos (Russo)":    ("q10a", "H > 0.50 m ó Q_mod > 0.50 m²/s."),
        "Q10b — Inestabilidad niños (Russo)":      ("q10b", "H > 0.25 m ó Q_mod > 0.15 m²/s."),
        "Q10c — Inestabilidad vehículos (Russo)":  ("q10c", "H > 0.30 m (criterio simplificado de calado)."),
        "Q11b — Accesibilidad de emergencia":        ("q11b", "Celdas accesibles (H < 0.60 m) en un instante dado."),
        "Q17 — Semáforo Russo (5 niveles)":        ("q17",  "Mapa categórico por umbrales de calado: somera / niños / adultos / crítico / extremo."),
    },
    "⚠️ E — Inestabilidad Xia et al. (2014/2022)": {
        "Q10a-Xia — Inestabilidad adultos":        ("q10a_xia", "Velocidad crítica U_c,p para adulto medio (Xia et al., 2014)."),
        "Q10b-Xia — Inestabilidad niños":          ("q10b_xia", "Velocidad crítica U_c,p para niño de 8 años (Xia et al., 2014)."),
        "Q10c-Xia — Inestabilidad vehículos":      ("q10c_xia", "Velocidad crítica de arrastre Mini Cooper (Xia et al., 2022)."),
    },
    "🔴 F — Daños normativos (RD 9/2008)": {
        "Q18 — Zona de graves daños":              ("q18", "H > 1 m ó V > 1 m/s ó H·V > 0.5 m²/s."),
    },
    "📊 G — Estadísticos espaciales": {
        "Q12 — Área inundada por hora":        ("q12", "Evolución temporal del área inundada (km²)."),
        "Q13 — Volumen por hora":              ("q13", "Volumen total Σ H × 100 m² por paso temporal."),
        "Q14 — Estadísticos por zona":         ("q14", "Media, máximo, desviación típica y percentiles de calado en zona."),
        "Q15 — Semáforo de calado (Russo)":    ("q15", "Área por nivel: Verde H≤0,25 m · Amarillo 0,25–0,50 m · Rojo H>0,50 m."),
    },
    "🚨 H — Evacuación": {
        "Q16 — Tiempo de evacuación":          ("q16", "Horas disponibles para evacuar antes de alcanzar el umbral crítico de peligro."),
    },
}

GRUPOS_EN = {
    "🎯 A — Point queries & velocity": {
        "Q1 — Depth at point":                 ("q1",  "Water depth H at a given coordinate and timestep."),
        "Q2 — Time series at point":           ("q2",  "H evolution at a point over all 120 h."),
        "Q3 — Velocity in area":               ("q3",  "Velocity V = Q_mod / H over wet cells."),
    },
    "💧 B — Spatial thresholds": {
        "Q4 — Flooded zones":                  ("q4",  "Cells with H ≥ threshold at a given timestep."),
        "Q5 — Flood extent over time":         ("q5",  "Time curve of flooded area."),
        "Q6 — Zones above threshold":          ("q6",  "Cells exceeding a given depth threshold."),
    },
    "⏱️ C — Temporal indicators": {
        "Q7 — Flood front arrival time":       ("q7",  "First hour water reaches each pixel."),
        "Q8 — Flood duration":                 ("q8",  "Total hours each pixel remains flooded."),
        "Q9 — Peak depth hour":                ("q9",  "Timestep at which each pixel reaches its max H."),
        "Q11 — Emergency vehicle window":      ("q11", "Hours accessible for rescue vehicles (H < 0.60 m)."),
    },
    "⚠️ D — Russo et al. (2013) hazard": {
        "Q10a — Adult instability (Russo)":        ("q10a",     "H > 0.50 m or Q_mod > 0.50 m²/s."),
        "Q10b — Child instability (Russo)":        ("q10b",     "H > 0.25 m or Q_mod > 0.15 m²/s."),
        "Q10c — Vehicle instability (Russo)":      ("q10c",     "H > 0.30 m (simplified depth criterion)."),
        "Q11b — Emergency accessibility":          ("q11b",     "Cells accessible to emergency vehicles (H < 0.60 m) at a given timestep."),
        "Q17 — Russo traffic-light (5 levels)":    ("q17",      "Categorical depth map: shallow / children / adults / critical / extreme."),
    },
    "⚠️ E — Xia et al. (2014/2022) instability": {
        "Q10a-Xia — Adult instability":            ("q10a_xia", "Critical velocity U_c,p for an average adult (Xia et al., 2014)."),
        "Q10b-Xia — Child instability":            ("q10b_xia", "Critical velocity U_c,p for an 8-year-old child (Xia et al., 2014)."),
        "Q10c-Xia — Vehicle instability":          ("q10c_xia", "Critical sweep velocity for Mini Cooper (Xia et al., 2022)."),
    },
    "🔴 F — Normative damage (RD 9/2008)": {
        "Q18 — Severe damage zone":                ("q18",      "H > 1 m or V > 1 m/s or H·V > 0.5 m²/s."),
    },
    "📊 G — Spatial statistics": {
        "Q12 — Flooded area by hour":          ("q12", "Time curve of flooded area (km²)."),
        "Q13 — Volume by hour":                ("q13", "Total volume Σ H × 100 m² per timestep."),
        "Q14 — Zone statistics":               ("q14", "Mean, max, std and depth percentiles for a spatial zone."),
        "Q15 — Depth traffic-light (Russo)":   ("q15", "Area by level: Green H≤0.25 m · Yellow 0.25–0.50 m · Red H>0.50 m."),
    },
    "🚨 H — Evacuation": {
        "Q16 — Evacuation time window":        ("q16", "Hours available to evacuate before reaching the critical hazard threshold."),
    },
}

GRUPOS = GRUPOS_EN if st.session_state.get("lang") == "en" else GRUPOS_ES

grupo    = st.sidebar.selectbox(_t("block_label"), list(GRUPOS.keys()))
consulta = st.sidebar.selectbox(_t("query_label"), list(GRUPOS[grupo].keys()))
qid, qdesc = GRUPOS[grupo][consulta]

# ── Parámetros dinámicos ─────────────────────────────────────────────────────
xll, yll = meta["xll"], meta["yll"]
xmx, ymx = meta["x_max"], meta["y_max"]
xc, yc   = (xll + xmx) / 2, (yll + ymx) / 2

param_hora   = qid in {"q1", "q3", "q4", "q6", "q10a", "q10b", "q10c", "q11b", "q14", "q15", "q17",
                        "q10a_xia", "q10b_xia", "q10c_xia", "q18"}
param_punto  = qid in {"q1", "q2"}
param_bbox   = qid not in {"q1", "q2"}
param_umbral = qid in {"q4", "q6", "q8"}
param_q16    = qid == "q16"

# Consultas que aceptan modo temporal ("todas las horas")
TEMPORAL_SUPPORT = {"q3", "q10a", "q10b", "q10c", "q11b", "q14", "q15"}
soporta_temporal = qid in TEMPORAL_SUPPORT
MAP_QUERIES = {"q3", "q4", "q6", "q7", "q8", "q9", "q10a", "q10b", "q10c", "q11", "q11b", "q16", "q17",
               "q10a_xia", "q10b_xia", "q10c_xia", "q18"}

st.sidebar.markdown(_t("params_label"))

hora = umbral_m = x_pt = y_pt = bbox = umbral_h16 = umbral_q16 = None
modo_temporal = False

if soporta_temporal:
    modo_temporal = st.sidebar.toggle(
        _t("all_hours"),
        value=False,
        help=_t("all_hours_help"),
    )

if param_hora and not modo_temporal:
    hora = st.sidebar.select_slider(_t("hour_slider"), horas_disp,
                                    value=horas_disp[min(9, len(horas_disp)-1)])

if param_punto:
    # Las coordenadas se capturan desde el mapa en el panel principal
    # (session_state persiste el último punto clicado)
    if "x_pt" not in st.session_state:
        st.session_state["x_pt"] = float(xc)
        st.session_state["y_pt"] = float(yc)
    x_pt = st.session_state["x_pt"]
    y_pt = st.session_state["y_pt"]
    st.sidebar.info(f"📍 X = {x_pt/1000:.2f} km\nY = {y_pt/1000:.2f} km")

if param_bbox:
    with st.sidebar.expander(_t("bbox_label"), expanded=False):
        st.caption(_t("bbox_caption"))
        usar = st.checkbox(_t("bbox_use"))
        if usar:
            c1, c2 = st.columns(2)
            bx0 = c1.number_input(_t("bbox_x_min"), value=(xc-25000)/1000, step=1.0, format="%.1f") * 1000
            bx1 = c2.number_input(_t("bbox_x_max"), value=(xc+25000)/1000, step=1.0, format="%.1f") * 1000
            by0 = c1.number_input(_t("bbox_y_min"), value=(yc-25000)/1000, step=1.0, format="%.1f") * 1000
            by1 = c2.number_input(_t("bbox_y_max"), value=(yc+25000)/1000, step=1.0, format="%.1f") * 1000
            bbox = (bx0, by0, bx1, by1)
            st.caption(f"→ {(bx1-bx0)/1000:.0f} × {(by1-by0)/1000:.0f} km")

if param_umbral:
    umbral_m = st.sidebar.number_input(_t("threshold_h"), value=0.01,
                                        min_value=0.001, max_value=5.0, step=0.05,
                                        format="%.3f")

if param_q16:
    st.sidebar.markdown(_t("q16_threshold"))
    umbral_h16 = st.sidebar.number_input(_t("q16_crit_h"), value=0.50, step=0.05)
    umbral_q16 = st.sidebar.number_input(_t("q16_crit_q"), value=0.50, step=0.05)

if (qid in MAP_QUERIES) and not modo_temporal:
    st.sidebar.markdown(_t("sidebar_map"))
    MAP_RES = st.sidebar.slider(
        _t("map_res"),
        min_value=800,
        max_value=3200,
        value=DEFAULT_MAP_RES,
        step=200,
        key="map_res_slider",
        help=_t("map_res_help"),
    )

st.sidebar.markdown("---")
ejecutar = st.sidebar.button(_t("run_btn"), width="stretch", type="primary")


# ═══════════════════════════════════════════════════════════════════════════════
# CABECERA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Título y resumen del dataset ─────────────────────────────────────────────
titulo = f"{_t('title_prefix')} · **{_dataset_display(dataset)}**"
if comparar and dataset2:
    titulo += f" vs **{_dataset_display(dataset2)}**"
st.title(titulo)

if comparar and dataset2:
    col_sa, col_sb = st.columns(2)
    with col_sa:
        show_dataset_summary(dataset)
    with col_sb:
        show_dataset_summary(dataset2)
    st.info(
        f"🔄 **{_t('compare_banner')}** — "
        f"**{_dataset_display(dataset)}** · **{_dataset_display(dataset2)}**  \n"
        f"{_t('compare_hint')}"
    )
else:
    show_dataset_summary(dataset)

LIGHT_QUERIES = {"q1", "q2", "q5", "q12", "q13"}
auto_exec = (qid in LIGHT_QUERIES) and not comparar
_badge_txt   = _t("badge_auto") if auto_exec else _t("badge_manual")
_badge_color = "#2e7d32"       if auto_exec else "#1565c0"
st.markdown(
    f'<div class="query-card">'
    f'<h4>{consulta} &nbsp;<span style="font-size:0.72rem;font-weight:600;'
    f'background:{_badge_color};color:#fff;padding:2px 8px;border-radius:10px;">'
    f'{_badge_txt}</span></h4>'
    f'<p>{qdesc}</p></div>',
    unsafe_allow_html=True
)

if not ejecutar and not auto_exec:
    st.info(_t("run_config"))
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════

t_start  = time.perf_counter()
progress = st.progress(0.0, text=_t("processing"))

try:
    # ── Q1 / Q2 — selección de punto (mapa / coordenadas / CSV) ─────────────
    if qid in {"q1", "q2"}:
        _modo_pt = st.radio(
            _t("mode_entry"),
            [_t("map_entry"), _t("coord_entry"), _t("csv_entry")],
            horizontal=True, label_visibility="collapsed",
        )

        if _modo_pt == _t("map_entry"):
            st.markdown(_t("click_map"))
            map_result = point_picker_map(meta, x_pt, y_pt)
            if map_result and map_result.get("last_clicked"):
                lat_c = map_result["last_clicked"]["lat"]
                lon_c = map_result["last_clicked"]["lng"]
                x_new, y_new = latlon_to_utm(lat_c, lon_c)
                if meta["xll"] <= x_new <= meta["x_max"] and meta["yll"] <= y_new <= meta["y_max"]:
                    st.session_state["x_pt"] = x_new
                    st.session_state["y_pt"] = y_new
                    x_pt = x_new
                    y_pt = y_new
                    st.rerun()
                else:
                    st.warning(_t("out_of_domain"))

        elif _modo_pt == _t("coord_entry"):
            st.markdown(_t("coord_title"))
            _c1, _c2 = st.columns(2)
            x_man = _c1.number_input(_t("x_east"), value=float(xc), step=100.0, format="%.1f",
                                     help=f"{_t('range_valid')}: {meta['xll']:.0f} – {meta['x_max']:.0f} m")
            y_man = _c2.number_input(_t("y_north"), value=float(yc), step=100.0, format="%.1f",
                                     help=f"{_t('range_valid')}: {meta['yll']:.0f} – {meta['y_max']:.0f} m")
            if meta["xll"] <= x_man <= meta["x_max"] and meta["yll"] <= y_man <= meta["y_max"]:
                st.session_state["x_pt"] = x_man
                st.session_state["y_pt"] = y_man
                x_pt = x_man
                y_pt = y_man
            else:
                st.error(f"{_t('out_domain_error')} "
                         f"X: [{meta['xll']:.0f}, {meta['x_max']:.0f}]  "
                         f"Y: [{meta['yll']:.0f}, {meta['y_max']:.0f}]")

        elif _modo_pt == _t("csv_entry"):
            st.markdown(_t("csv_title"))
            st.caption(_t("csv_example"))
            _csv_file = st.file_uploader(_t("csv_upload"), type=["csv"], label_visibility="collapsed")
            if _csv_file is not None:
                import io, csv as _csv, pathlib as _pl
                if _pl.Path(_pl.Path(_csv_file.name).stem).suffix:
                    st.error(_t("csv_invalid_filename"))
                    st.stop()
                _raw = _csv_file.read(200_001)
                if len(_raw) > 200_000:
                    st.error(_t("csv_too_large"))
                    st.stop()
                content = _raw.decode("utf-8", errors="replace")
                reader = _csv.reader(io.StringIO(content))
                _filas = [r for r in reader if r and not r[0].strip().startswith("#")]
                _header = _filas[0] if _filas and not _filas[0][0].replace(".","").lstrip("-").isdigit() else None
                _datos  = (_filas[1:] if _header else _filas)[:200]
                if not _datos:
                    st.warning(_t("csv_empty"))
                    st.stop()

                def _parse_nombre(r, idx):
                    nom = r[0].strip()[:80] if len(r) > 2 else f"{_t('point_label')} {idx+1}"
                    return "".join(c for c in nom if c.isprintable())

                if qid == "q1":
                    _resultados = []
                    for _i, _r in enumerate(_datos):
                        try:
                            _nom = _parse_nombre(_r, _i)
                            _xr, _yr = float(_r[-2]), float(_r[-1])
                            _hv = ca.calado_en_punto(_xr, _yr, hora) if (
                                meta["xll"] <= _xr <= meta["x_max"] and meta["yll"] <= _yr <= meta["y_max"]
                            ) else None
                            _nivel = (_t("dry_cell") if _hv == 0 else
                                      _t("safe") if _hv < H_NINO else
                                      _t("caution") if _hv < H_ADULTO else _t("danger")
                                      ) if _hv is not None else _t("out_domain_label")
                            _resultados.append({"Nombre": _nom, "X (m)": _xr, "Y (m)": _yr,
                                                "H (m)": f"{_hv:.4f}" if _hv is not None else "—",
                                                "Nivel": _nivel})
                        except (ValueError, IndexError):
                            continue
                    import pandas as pd
                    st.dataframe(pd.DataFrame(_resultados), width="stretch")
                    progress.progress(1.0, text=_t("completed"))
                    st.stop()

                elif qid == "q2":
                    _palette = ["#1976d2","#e53935","#2e7d32","#f57c00","#7b1fa2",
                                "#00838f","#ad1457","#558b2f","#6d4c41","#37474f"]
                    _series, _horas_ref = [], None
                    for _i, _r in enumerate(_datos):
                        try:
                            _nom = _parse_nombre(_r, _i)
                            _xr, _yr = float(_r[-2]), float(_r[-1])
                            if not (meta["xll"] <= _xr <= meta["x_max"] and meta["yll"] <= _yr <= meta["y_max"]):
                                continue
                            _res = ca.serie_temporal_punto(_xr, _yr)
                            if _horas_ref is None:
                                _horas_ref = _res["horas"]
                            _series.append({"name": _nom, "y": _res["H"],
                                            "color": _palette[_i % len(_palette)]})
                        except (ValueError, IndexError):
                            continue
                    if _series and _horas_ref is not None:
                        progress.progress(1.0, text=_t("completed"))
                        show(make_multi_line(_horas_ref, _series,
                                             _t("q2_multi_title"), f"{_t('depth_h')} (m)"))
                        st.stop()
                    else:
                        st.warning(_t("csv_no_valid_points"))
                        st.stop()
            elif _csv_file is None:
                st.info(_t("upload_csv"))
                st.stop()

        st.caption(f"{_t('active_point')}: X = {x_pt:.0f} m · Y = {y_pt:.0f} m "
                   f"({x_pt/1000:.2f} km, {y_pt/1000:.2f} km)")
        st.divider()

    def _render_discrete_map(qid, dataset, dataset2, bbox, comparar,
                             load_fn, load_args,
                             spinner_key, render_metrics,
                             bands_fn, n_steps,
                             map_title, hover_label_key,
                             compare_key=None):
        """Renderiza Q7/Q8/Q9: carga, métricas, heatmap (single o comparativo) + descarga.
        compare_key: (stats_field, label_i18n_key, val_fmt, delta_fmt) para métrica en modo comparativo."""
        with st.spinner(_t(spinner_key)):
            grid, x_ax, y_ax, stats = load_fn(dataset, *load_args)
        if comparar and dataset2:
            grid2, x_ax2, y_ax2, stats2 = load_fn(dataset2, *load_args)
        progress.progress(1.0, text=_t("completed"))
        bands, zmax = bands_fn(n_steps)
        hover_label = _t(hover_label_key)
        if comparar and dataset2:
            fig_a = heatmap_discrete(grid,  x_ax,  y_ax,
                                     f"{map_title} — {_dataset_display(dataset)}",
                                     bands, zmax, "h",
                                     hover_fmt=".0f", hover_label=hover_label, hover_unit="h")
            fig_b = heatmap_discrete(grid2, x_ax2, y_ax2,
                                     f"{map_title} — {_dataset_display(dataset2)}",
                                     bands, zmax, "h",
                                     hover_fmt=".0f", hover_label=hover_label, hover_unit="h")
            ca1, ca2 = st.columns(2)
            if compare_key:
                field, label_key, val_fmt, delta_fmt = compare_key
                vA = stats.get(field, 0)
                vB = stats2.get(field, 0)
                cmetric(ca1, f"{_dataset_display(dataset)} — {_t(label_key)}", val_fmt.format(vA))
                cmetric(ca2, f"{_dataset_display(dataset2)} — {_t(label_key)}",
                        val_fmt.format(vB), delta_fmt.format(vB - vA))
            with ca1:
                ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
                download_geotiff_button(grid,  x_ax,  y_ax,  qid, dataset,  ds_label=_dataset_display(dataset))
            with ca2:
                ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
                download_geotiff_button(grid2, x_ax2, y_ax2, qid, dataset2, ds_label=_dataset_display(dataset2))
        else:
            render_metrics(stats)
            show(heatmap_discrete(grid, x_ax, y_ax, map_title,
                                  bands, zmax, "h",
                                  hover_fmt=".0f", hover_label=hover_label, hover_unit="h"))
            download_geotiff_button(grid, x_ax, y_ax, qid, dataset)

    if qid == "q1":
        h_val = ca.calado_en_punto(x_pt, y_pt, hora)
        progress.progress(1.0, text=_t("completed"))
        def _q1_nivel(h):
            if h == 0: return _t("dry_warning")
            return _t("safe") if h < H_NINO else _t("caution") if h < H_ADULTO else _t("danger")
        if comparar and dataset2:
            setup_ca(get_meta(dataset2))
            h_val2 = ca.calado_en_punto(x_pt, y_pt, hora)
            setup_ca(meta)
            ca1, ca2 = st.columns(2)
            cmetric(ca1, f"{_dataset_display(dataset)} — {_t('depth_h')}", f"{h_val:.4f} m")
            cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('depth_h')}",
                    f"{h_val2:.4f} m", f"{h_val2-h_val:+.4f} m")
            st.caption(f"📍 {x_pt/1000:.2f} km, {y_pt/1000:.2f} km · {hora} h")
            ca1.info(f"**{_dataset_display(dataset)}**: {_q1_nivel(h_val)}")
            ca2.info(f"**{_dataset_display(dataset2)}**: {_q1_nivel(h_val2)}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric(_t("depth_h"), f"{h_val:.4f} m")
            c2.metric(_t("coordinates"), f"{x_pt/1000:.2f} km, {y_pt/1000:.2f} km")
            c3.metric(_t("hour"), f"{hora} h")
            if h_val == 0:
                st.warning(_t("dry_warning"))
            else:
                nivel = _q1_nivel(h_val)
                st.info(f"**{_t('depth_level')}:** {nivel}")

    elif qid == "q2":
        res = ca.serie_temporal_punto(x_pt, y_pt)
        progress.progress(1.0, text=_t("completed"))
        h_arr     = np.array(res["H"])
        horas_arr = np.array(res["horas"])
        wet       = h_arr > 0
        hora_pico = int(horas_arr[np.argmax(h_arr)]) if wet.any() else 0
        pt_label  = f"({x_pt/1000:.2f} km, {y_pt/1000:.2f} km)"
        if comparar and dataset2:
            setup_ca(get_meta(dataset2))
            res2 = ca.serie_temporal_punto(x_pt, y_pt)
            setup_ca(meta)
            h_arr2 = np.array(res2["H"])
            wet2   = h_arr2 > 0
            show(make_multi_line(res["horas"],
                [{"name": _dataset_display(dataset),  "y": res["H"],  "color": "#1565c0"},
                 {"name": _dataset_display(dataset2), "y": res2["H"], "color": "#c62828"}],
                f"{_t('q2_series_title')} {pt_label}", f"{_t('depth_h')} (m)"))
            ca1, ca2 = st.columns(2)
            cmetric(ca1, f"{_dataset_display(dataset)} — H máx.", f"{h_arr.max():.3f} m")
            cmetric(ca2, f"{_dataset_display(dataset2)} — H máx.",
                    f"{h_arr2.max():.3f} m", f"{h_arr2.max()-h_arr.max():+.3f} m")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(_t("steps_flooded"), f"{int(wet.sum())} / {len(h_arr)}")
            c2.metric(_t("max_h"),         f"{h_arr.max():.3f} m")
            c3.metric(_t("peak_hour"),     f"{hora_pico} h")
            c4.metric(_t("q2_h_mean_wet"), f"{h_arr[wet].mean():.3f} m" if wet.any() else _t("q2_dry"))
            show(make_line(res["horas"], res["H"],
                          f"{_t('q2_series_title')} {pt_label}",
                          _t("hour_axis"), f"{_t('depth_h')} (m)"))

    elif qid == "q3":
        if modo_temporal:
            res = q3_temporal(meta, bbox, progress)
            progress.progress(1.0, text=_t("completed"))
            v_max = max(res["V_max"]) if res["V_max"] else 0
            v_med = max(res["V_media"]) if res["V_media"] else 0
            c1, c2 = st.columns(2)
            c1.metric(_t("vel_max_sim"), f"{v_max:.3f} m/s")
            c2.metric(_t("vel_mean_peak"), f"{v_med:.3f} m/s")
            show(make_multi_line(res["horas"],
                [{"name": "V_max",   "y": res["V_max"],   "color": "#e53935"},
                 {"name": "V_media", "y": res["V_media"], "color": "#1976d2"}],
                _t("vel_evolution"), "V (m/s)"))
        else:
            progress.progress(0.3, text=_t("loading_step"))
            grid, x_ax, y_ax, stats = _c_q3(dataset, hora, bbox)
            progress.progress(1.0, text=_t("completed"))
            t_vel = _t("vel_map_title").format(hora=hora)
            hm_vel = dict(adaptive_clip=True, hover_label="V", hover_unit="m/s")
            if comparar and dataset2:
                grid2, x_ax2, y_ax2, stats2 = _c_q3(dataset2, hora, bbox)
                fig_a = make_heatmap(grid,  x_ax,  y_ax,
                                     f"{t_vel} — {_dataset_display(dataset)}",
                                     CMAP_VELOCIDAD, "m/s", **hm_vel)
                fig_b = make_heatmap(grid2, x_ax2, y_ax2,
                                     f"{t_vel} — {_dataset_display(dataset2)}",
                                     CMAP_VELOCIDAD, "m/s", **hm_vel)
                ca1, ca2 = st.columns(2)
                cmetric(ca1, f"{_dataset_display(dataset)} — {_t('vel_max')}", f"{stats['V_max']:.3f} m/s")
                cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('vel_max')}", f"{stats2['V_max']:.3f} m/s",
                        f"{stats2['V_max']-stats['V_max']:+.3f} m/s")
                ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
                ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric(_t("wet_cells"), f"{stats['n']:,}")
                c2.metric(_t("vel_max"),   f"{stats['V_max']:.3f} m/s")
                c3.metric(_t("vel_mean"),  f"{stats['V_media']:.3f} m/s")
                show(make_heatmap(grid, x_ax, y_ax, t_vel, CMAP_VELOCIDAD, "m/s", **hm_vel))

    # ── Q4 / Q6 ──────────────────────────────────────────────────────────────
    elif qid in {"q4", "q6"}:
        progress.progress(0.3, text=_t("loading_step"))
        grid, x_ax, y_ax, stats = q_umbral_h(meta, hora, umbral_m, bbox)
        progress.progress(1.0, text=_t("completed"))
        grid_v, x_v, y_v, visual_floor, cropped = _focus_h_visual(
            grid, x_ax, y_ax, umbral_m, auto_crop=(bbox is None)
        )
        map_title_q4 = f"{_t('depth_h')} — {_t('hour')} {hora} h · ≥ {umbral_m:.3f} m · log"
        notes = []
        if visual_floor > umbral_m:
            notes.append(_t("note_visual_floor").format(v=visual_floor))
        if cropped:
            notes.append(_t("note_cropped"))
        if bbox is None:
            notes.append(_t("note_bbox_tip"))
        if comparar and dataset2:
            meta2 = get_meta(dataset2)
            setup_ca(meta2)
            grid2, x_ax2, y_ax2, stats2 = q_umbral_h(meta2, hora, umbral_m, bbox)
            setup_ca(meta)
            grid2_v, x_v2, y_v2, _, _ = _focus_h_visual(grid2, x_ax2, y_ax2, umbral_m, auto_crop=(bbox is None))
            fig_a = heatmap_h(grid_v,  x_v,  y_v,
                              f"{map_title_q4} — {_dataset_display(dataset)}",  cmap=CMAP_H_FOCUS)
            fig_b = heatmap_h(grid2_v, x_v2, y_v2,
                              f"{map_title_q4} — {_dataset_display(dataset2)}", cmap=CMAP_H_FOCUS)
            ca1, ca2 = st.columns(2)
            cmetric(ca1, f"{_dataset_display(dataset)} — {_t('area')}", f"{stats['area_km2']:.2f} km²")
            cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('area')}",
                    f"{stats2['area_km2']:.2f} km²",
                    f"{stats2['area_km2']-stats['area_km2']:+.2f} km²")
            ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid, x_ax, y_ax, "q4", dataset, hora,
                                    ds_label=_dataset_display(dataset))
            ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid2, x_ax2, y_ax2, "q4", dataset2, hora,
                                    ds_label=_dataset_display(dataset2))
        else:
            c1, c2 = st.columns(2)
            c1.metric(_t("cells"), f"{stats['n']:,}")
            c2.metric(_t("area"),  f"{stats['area_km2']:.2f} km²")
            show(heatmap_h(grid_v, x_v, y_v, map_title_q4, cmap=CMAP_H_FOCUS))
            if notes:
                st.caption(" ".join(notes))
            download_geotiff_button(grid, x_ax, y_ax, "q4", dataset, hora)

    elif qid == "q5":
        progress.progress(0.5, text=_t("iterating"))
        res = _c_q5(dataset, bbox)
        progress.progress(1.0, text=_t("completed"))
        areas = np.array(res["area_km2"])
        if comparar and dataset2:
            res2 = _c_q5(dataset2, bbox)
            areas2 = np.array(res2["area_km2"])
            c1, c2, c3, c4 = st.columns(4)
            _dA, _dB = areas.max(), areas2.max()
            _s = "+" if _dB >= _dA else ""
            c1.metric(_t("max_area_a"), f"{_dA:.1f} km²")
            c2.metric(_t("peak_hour_a"), f"{res['horas'][int(areas.argmax())]} h")
            cmetric(c3, _t("max_area_b"), f"{_dB:.1f} km²", f"{_s}{_dB-_dA:.1f} km²")
            c4.metric(_t("peak_hour_b"), f"{res2['horas'][int(areas2.argmax())]} h")
            show(make_multi_line(res["horas"],
                [{"name": _dataset_display(dataset),  "y": res["area_km2"],  "color": "#1565c0"},
                 {"name": _dataset_display(dataset2), "y": res2["area_km2"], "color": "#c62828"}],
                _t("q5_compare_title"), "km²"))
            setup_ca(meta)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric(_t("max_area"),  f"{areas.max():.1f} km²")
            c2.metric(_t("peak_hour"), f"{res['horas'][int(areas.argmax())]} h")
            c3.metric(_t("mean_area"), f"{areas.mean():.1f} km²")
            show(make_line(res["horas"], res["area_km2"],
                _t("q5_title"), _t("hour_axis"), "Area (km²)"))

    elif qid == "q7":
        def _m(stats):
            c1, c2 = st.columns(2)
            c1.metric(_t("first_arrival"), f"{int(stats['min'])} h")
            c2.metric(_t("last_arrival"),  f"{int(stats['max'])} h")
        _render_discrete_map("q7", dataset, dataset2, bbox, comparar,
            _c_q7, (bbox,), "calc_arrival", _m,
            _bands_llegada, meta["n_steps"],
            _t("arrival_map_title"), "hover_arrival",
            compare_key=("min", "first_arrival", "{:.0f} h", "{:+.0f} h"))

    elif qid == "q8":
        def _m(stats):
            c1, c2 = st.columns(2)
            c1.metric(_t("mean_duration"), f"{stats['media']:.1f} h")
            c2.metric(_t("max_duration"),  f"{stats['max']:.0f} h")
        _render_discrete_map("q8", dataset, dataset2, bbox, comparar,
            _c_q8, (umbral_m, bbox), "calc_duration", _m,
            _bands_duracion, meta["n_steps"],
            _t("q8_map_title").format(v=umbral_m), "hover_duration",
            compare_key=("media", "mean_duration", "{:.1f} h", "{:+.1f} h"))

    elif qid == "q9":
        def _m(stats):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(_t("cells"),         f"{stats['n']:,}")
            c2.metric(_t("max_h"),         f"{stats.get('H_max', 0):.3f} m")
            c3.metric(_t("first_arrival"), f"{int(stats.get('hora_pico_min', 0))} h")
            c4.metric(_t("last_arrival"),  f"{int(stats.get('hora_pico_max', 0))} h")
        _render_discrete_map("q9", dataset, dataset2, bbox, comparar,
            _c_q9, (bbox,), "calc_hmax", _m,
            _bands_llegada, meta["n_steps"],
            _t("peak_depth_map_title"), "peak_hour_short",
            compare_key=("H_max", "max_h", "{:.3f} m", "{:+.3f} m"))

    # ── Q10a / Q10b / Q10c ───────────────────────────────────────────────────
    elif qid in {"q10a", "q10b", "q10c"}:
        tipo_map = {
            "q10a": ("adultos",           "q10a_label", CMAP_PELIGRO, "#e53935"),
            "q10b": ("ninos",             "q10b_label", CMAP_NARANJA, "#fb8c00"),
            "q10c": ("vehiculos_ligeros", "q10c_label", CMAP_DURACION,"#f57c00"),
        }
        tipo, label_key, _, line_color = tipo_map[qid]
        label = _t(label_key)
        if modo_temporal:
            res = q10_peligrosidad_temporal(meta, tipo, bbox, progress)
            progress.progress(1.0, text=_t("completed"))
            areas = np.array(res["area_km2"])
            c1, c2, c3 = st.columns(3)
            c1.metric(_t("danger_area_max"), f"{areas.max():.1f} km²")
            c2.metric(_t("peak_hour"),       f"{res['horas'][int(areas.argmax())]} h")
            c3.metric(_t("mean_area"),       f"{areas.mean():.1f} km²")
            show(make_line(res["horas"], res["area_km2"],
                f"{_t('q10_evol_prefix')} — {label}", _t("hour_axis"), "Area (km²)",
                color=line_color))
        else:
            progress.progress(0.3, text=_t("loading_step"))
            grid, x_ax, y_ax, stats = _c_q10i(dataset, hora, tipo, bbox)
            progress.progress(1.0, text=_t("completed"))
            bands, zmax = _bands_intensidad()
            map_title_q10 = f"{label} — {_t('hour')} {hora} h · {_t('q10_map_intensity')}"
            if comparar and dataset2:
                cmap_q10 = CMAP_PELIGRO if tipo == "adultos" else \
                           CMAP_NARANJA if tipo == "ninos" else CMAP_DURACION
                ga, xa_a, ya_a, sa = _c_q10p(dataset,  hora, tipo, bbox)
                gb, xa_b, ya_b, sb = _c_q10p(dataset2, hora, tipo, bbox)
                fig_a = heatmap_h(ga, xa_a, ya_a,
                                  f"{label} — {_dataset_display(dataset)}", cmap=cmap_q10)
                fig_b = heatmap_h(gb, xa_b, ya_b,
                                  f"{label} — {_dataset_display(dataset2)}", cmap=cmap_q10)
                ca1, ca2 = st.columns(2)
                _da, _db = sa['area_km2'], sb['area_km2']
                cmetric(ca1, f"{_dataset_display(dataset)} — {_t('area')}", f"{_da:.1f} km²")
                ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
                download_geotiff_button(ga, xa_a, ya_a, qid, dataset,  hora,
                                        ds_label=_dataset_display(dataset))
                _sign = "+" if _db >= _da else ""
                cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('area')}",
                        f"{_db:.1f} km²", f"{_sign}{_db-_da:.1f} km²")
                ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
                download_geotiff_button(gb, xa_b, ya_b, qid, dataset2, hora,
                                        ds_label=_dataset_display(dataset2))
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(_t("cells_danger"),    f"{stats['n']:,}")
                c2.metric(_t("area"),            f"{stats['area_km2']:.2f} km²")
                c3.metric(_t("mean_intensity"),  f"{stats['intensity_med']:.2f}×")
                c4.metric(_t("peak_intensity"),  f"{stats['intensity_max']:.2f}×")
                show(heatmap_discrete(grid, x_ax, y_ax, map_title_q10,
                    bands, zmax, "x umbral",
                    hover_fmt=".2f", hover_label=_t("hover_intensity"), hover_unit="x"))

    elif qid == "q11":
        with st.spinner(_t("calc_emergency")):
            grid, x_ax, y_ax, stats = _c_q11(dataset, bbox)
        progress.progress(1.0, text=_t("completed"))
        bands, zmax = _bands_ventana(meta["n_steps"])
        hw = dict(hover_fmt=".0f", hover_label=_t("hover_window"), hover_unit="h")
        if comparar and dataset2:
            grid2, x_ax2, y_ax2, stats2 = _c_q11(dataset2, bbox)
            fig_a = heatmap_discrete(grid,  x_ax,  y_ax,
                                     f"{_t('q11_map_title')} — {_dataset_display(dataset)}",
                                     bands, zmax, "h", **hw)
            fig_b = heatmap_discrete(grid2, x_ax2, y_ax2,
                                     f"{_t('q11_map_title')} — {_dataset_display(dataset2)}",
                                     bands, zmax, "h", **hw)
            ca1, ca2 = st.columns(2)
            cmetric(ca1, f"{_dataset_display(dataset)} — {_t('mean_window')}", f"{stats['media']:.1f} h")
            cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('mean_window')}", f"{stats2['media']:.1f} h",
                    f"{stats2['media']-stats['media']:+.1f} h")
            ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid,  x_ax,  y_ax,  "q11", dataset,
                                    ds_label=_dataset_display(dataset))
            ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid2, x_ax2, y_ax2, "q11", dataset2,
                                    ds_label=_dataset_display(dataset2))
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(_t("practicable_cells"), f"{stats['n']:,}")
            c2.metric(_t("mean_window"),       f"{stats['media']:.1f} h")
            c3.metric(_t("min_window"),        f"{stats['min']:.0f} h")
            c4.metric(_t("max_practicable"),   f"{stats['max']:.0f} h")
            show(heatmap_discrete(grid, x_ax, y_ax, _t("q11_map_title"), bands, zmax, "h", **hw))
            download_geotiff_button(grid, x_ax, y_ax, "q11", dataset)

    elif qid == "q11b":
        if modo_temporal:
            res = q11b_temporal(meta, bbox, progress)
            progress.progress(1.0, text=_t("completed"))
            areas = np.array(res["area_km2"])
            c1, c2, c3 = st.columns(3)
            c1.metric(_t("max_practicable"), f"{areas.max():.1f} km²")
            c2.metric(_t("peak_hour"),       f"{res['horas'][int(areas.argmax())]} h")
            c3.metric(_t("mean_area"),       f"{areas.mean():.1f} km²")
            show(make_line(res["horas"], res["area_km2"],
                _t("q11b_evol_title"), _t("hour_axis"), "Area (km²)",
                color="#43a047"))
        else:
            progress.progress(0.3, text=_t("loading_step"))
            grid, x_ax, y_ax, stats = _c_q11b(dataset, hora, bbox)
            progress.progress(1.0, text=_t("completed"))
            t11b = _t("q11b_map_title").format(hora=hora)
            if comparar and dataset2:
                grid2, x_ax2, y_ax2, stats2 = _c_q11b(dataset2, hora, bbox)
                fig_a = heatmap_h(grid,  x_ax,  y_ax,
                                  f"{t11b} — {_dataset_display(dataset)}",  cmap="Greens")
                fig_b = heatmap_h(grid2, x_ax2, y_ax2,
                                  f"{t11b} — {_dataset_display(dataset2)}", cmap="Greens")
                ca1, ca2 = st.columns(2)
                cmetric(ca1, f"{_dataset_display(dataset)} — {_t('area')}", f"{stats['area_km2']:.2f} km²")
                cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('area')}", f"{stats2['area_km2']:.2f} km²",
                        f"{stats2['area_km2']-stats['area_km2']:+.2f} km²")
                ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
                ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
            else:
                c1, c2 = st.columns(2)
                c1.metric(_t("practicable_cells"), f"{stats['n']:,}")
                c2.metric(_t("area"),              f"{stats['area_km2']:.2f} km²")
                show(heatmap_h(grid, x_ax, y_ax, t11b, cmap="Greens"))

    elif qid == "q12":
        progress.progress(0.5, text=_t("iterating"))
        res = _c_q12(dataset, bbox)
        progress.progress(1.0, text=_t("completed"))
        areas = np.array(res["area_km2"])
        if comparar and dataset2:
            res2 = _c_q12(dataset2, bbox)
            areas2 = np.array(res2["area_km2"])
            c1, c2, c3, c4 = st.columns(4)
            _dA, _dB = areas.max(), areas2.max()
            _s = "+" if _dB >= _dA else ""
            c1.metric(_t("max_area_a"), f"{_dA:.1f} km²")
            c2.metric(_t("peak_hour_a"), f"{res['horas'][int(areas.argmax())]} h")
            cmetric(c3, _t("max_area_b"), f"{_dB:.1f} km²", f"{_s}{_dB-_dA:.1f} km²")
            c4.metric(_t("peak_hour_b"), f"{res2['horas'][int(areas2.argmax())]} h")
            show(make_multi_line(res["horas"],
                [{"name": _dataset_display(dataset),  "y": res["area_km2"],  "color": "#1565c0"},
                 {"name": _dataset_display(dataset2), "y": res2["area_km2"], "color": "#c62828"}],
                _t("q12_compare_title"), "km²"))
            setup_ca(meta)
        else:
            c1, c2 = st.columns(2)
            c1.metric(_t("max_area"),  f"{areas.max():.1f} km²")
            c2.metric(_t("peak_hour"), f"{res['horas'][int(areas.argmax())]} h")
            show(make_line(res["horas"], res["area_km2"],
                _t("q12_title"), _t("hour_axis"), "Area (km²)"))

    elif qid == "q13":
        progress.progress(0.5, text=_t("iterating"))
        res = _c_q13(dataset, bbox)
        progress.progress(1.0, text=_t("completed"))
        vols_mm3 = [v / 1e6 for v in res["volumen_m3"]]
        if comparar and dataset2:
            res2 = _c_q13(dataset2, bbox)
            vols2 = [v / 1e6 for v in res2["volumen_m3"]]
            _vA, _vB = max(vols_mm3), max(vols2)
            _s = "+" if _vB >= _vA else ""
            c1, c2 = st.columns(2)
            cmetric(c1, f"{_dataset_display(dataset)} — {_t('max_volume')}", f"{_vA:.1f} Mm³")
            cmetric(c2, f"{_dataset_display(dataset2)} — {_t('max_volume')}",
                    f"{_vB:.1f} Mm³", f"{_s}{_vB-_vA:.1f} Mm³")
            show(make_multi_line(res["horas"],
                [{"name": _dataset_display(dataset),  "y": vols_mm3, "color": "#1565c0"},
                 {"name": _dataset_display(dataset2), "y": vols2,    "color": "#c62828"}],
                _t("q13_compare_title"), "Mm³"))
        else:
            c1, c2 = st.columns(2)
            c1.metric(_t("max_volume"), f"{max(vols_mm3):.1f} Mm³")
            c2.metric(_t("peak_hour"),  f"{res['horas'][int(np.argmax(vols_mm3))]} h")
            show(make_line(res["horas"], vols_mm3,
                _t("q13_title"), _t("hour_axis"), "Volume (Mm³)", color="#00838f"))

    elif qid == "q14":
        bbox_q14 = bbox if bbox else (xc - 25000, yc - 25000, xc + 25000, yc + 25000)
        if modo_temporal:
            res = q14_temporal(meta, bbox_q14, progress)
            progress.progress(1.0, text=_t("completed"))
            c1, c2, c3 = st.columns(3)
            c1.metric(_t("h_max_global"), f"{max(res['H_max']):.3f} m")
            c2.metric(_t("peak_hour"),    f"{res['horas'][int(np.argmax(res['H_max']))]} h")
            c3.metric(_t("h_mean_global"), f"{np.mean(res['H_media']):.3f} m")
            show(make_multi_line(res["horas"],
                [{"name": _t("max_h"),  "y": res["H_max"],   "color": "#e53935"},
                 {"name": "P95",        "y": res["H_P95"],   "color": "#fb8c00"},
                 {"name": _t("median"), "y": res["H_P50"],   "color": "#43a047"},
                 {"name": _t("mean_h"), "y": res["H_media"], "color": "#1976d2"}],
                _t("q14_evol_title"), "H (m)"))
        else:
            progress.progress(0.5, text=_t("calc_stats"))
            res = ca.stats_zona(*bbox_q14, hora)
            progress.progress(1.0, text=_t("completed"))
            if res.get("n_celdas", 0) == 0:
                st.warning(_t("zone_dry"))
            elif comparar and dataset2:
                setup_ca(get_meta(dataset2))
                res2 = ca.stats_zona(*bbox_q14, hora)
                setup_ca(meta)
                _bars = ["P25", "P50", "P75", "P95", _t("mean_h"), _t("max_h")]
                _cols = ["#90caf9"]*4 + ["#ff9800", "#e53935"]
                fig_a = make_bar(_bars,
                    [res["H_P25"], res["H_P50"], res["H_P75"], res["H_P95"],
                     res["H_media"], res["H_max"]],
                    f"{_dataset_display(dataset)} — hora {hora} h", "H (m)", _cols)
                fig_b = make_bar(_bars,
                    [res2.get("H_P25",0), res2.get("H_P50",0), res2.get("H_P75",0),
                     res2.get("H_P95",0), res2.get("H_media",0), res2.get("H_max",0)],
                    f"{_dataset_display(dataset2)} — hora {hora} h", "H (m)", _cols)
                ca1, ca2 = st.columns(2)
                cmetric(ca1, f"{_dataset_display(dataset)} — {_t('mean_h')}",
                        f"{res['H_media']:.3f} m")
                cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('mean_h')}",
                        f"{res2.get('H_media',0):.3f} m",
                        f"{res2.get('H_media',0)-res['H_media']:+.3f} m")
                ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
                ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
            else:
                cols = st.columns(5)
                cols[0].metric(_t("wet_cells"), f"{res['n_celdas']:,}")
                cols[1].metric(_t("area"),      f"{res['area_km2']:.2f} km²")
                cols[2].metric(_t("mean_h"),    f"{res['H_media']:.3f} m")
                cols[3].metric(_t("max_h"),     f"{res['H_max']:.3f} m")
                cols[4].metric("σ H",           f"{res.get('H_std', 0):.3f} m")
                show(make_bar(
                    ["P25", "P50", "P75", "P95", _t("mean_h"), _t("max_h")],
                    [res["H_P25"], res["H_P50"], res["H_P75"], res["H_P95"],
                     res["H_media"], res["H_max"]],
                    _t("q14_dist_title").format(hora=hora), "H (m)",
                    ["#90caf9"]*4 + ["#ff9800", "#e53935"]))

    elif qid == "q15":
        if modo_temporal:
            res = q15_temporal(meta, bbox, progress)
            progress.progress(1.0, text=_t("completed"))
            c1, c2, c3 = st.columns(3)
            c1.metric(_t("q15_green_max"),  f"{max(res['verde']):.1f} km²")
            c2.metric(_t("q15_yellow_max"), f"{max(res['amarillo']):.1f} km²")
            c3.metric(_t("q15_red_max"),    f"{max(res['rojo']):.1f} km²")
            show(make_stacked_area(res["horas"],
                [{"name": _t("q15_green_label"),  "y": res["verde"],    "color": "#2ecc71"},
                 {"name": _t("q15_yellow_label"), "y": res["amarillo"], "color": "#f39c12"},
                 {"name": _t("q15_red_label"),    "y": res["rojo"],     "color": "#e74c3c"}],
                _t("q15_evol_title"), "Area (km²)"))
        else:
            progress.progress(0.5, text=_t("calc_areas"))
            res = q15_area_peligro(meta, hora, bbox)
            progress.progress(1.0, text=_t("completed"))
            if comparar and dataset2:
                res2 = q15_area_peligro(get_meta(dataset2), hora, bbox)
                fig_a = make_bar(
                    [_t("q15_green_bar"), _t("q15_yellow_bar"), _t("q15_red_bar")],
                    [res["verde"], res["amarillo"], res["rojo"]],
                    f"{_dataset_display(dataset)} — hora {hora} h", "Area (km²)",
                    ["#2ecc71", "#f39c12", "#e74c3c"])
                fig_b = make_bar(
                    [_t("q15_green_bar"), _t("q15_yellow_bar"), _t("q15_red_bar")],
                    [res2["verde"], res2["amarillo"], res2["rojo"]],
                    f"{_dataset_display(dataset2)} — hora {hora} h", "Area (km²)",
                    ["#2ecc71", "#f39c12", "#e74c3c"])
                ca1, ca2 = st.columns(2)
                cmetric(ca1, f"{_dataset_display(dataset)} — {_t('q15_red_label')}",
                        f"{res['rojo']:.1f} km²")
                cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('q15_red_label')}",
                        f"{res2['rojo']:.1f} km²", f"{res2['rojo']-res['rojo']:+.1f} km²")
                ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
                ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(_t("q15_green_label"),  f"{res['verde']:.1f} km²")
                c2.metric(_t("q15_yellow_label"), f"{res['amarillo']:.1f} km²")
                c3.metric(_t("q15_red_label"),    f"{res['rojo']:.1f} km²")
                c4.metric(_t("area"),             f"{res['total']:.1f} km²")
                show(make_bar(
                    [_t("q15_green_bar"), _t("q15_yellow_bar"), _t("q15_red_bar")],
                    [res["verde"], res["amarillo"], res["rojo"]],
                    f"{_t('danger_level')} — {_t('hour')} {hora} h", "Area (km²)",
                    ["#2ecc71", "#f39c12", "#e74c3c"]))

    # ── Q17 Mapa semáforo Russo ──────────────────────────────────────────────
    elif qid == "q17":
        progress.progress(0.3, text=_t("loading_step"))
        grid, x_ax, y_ax, stats = q_umbral_h(meta, hora, H_WET, bbox)
        progress.progress(1.0, text=_t("completed"))
        russo_cmap, tvals, _, zmax = russo_colorscale()
        ttext = russo_ticktext(st.session_state.get("lang", "es"))
        t17 = _t("q17_map_title").format(hora=hora)
        hm17 = dict(zmin=0.0, zmax=zmax, cbar_tickvals=tvals, cbar_ticktext=ttext,
                    hover_label="H", hover_unit="m", zsmooth=False)
        if comparar and dataset2:
            grid2, x_ax2, y_ax2, _ = q_umbral_h(get_meta(dataset2), hora, H_WET, bbox)
            fig_a = make_heatmap(grid,  x_ax,  y_ax,
                                 f"{t17} — {_dataset_display(dataset)}",  russo_cmap, "H (m)", **hm17)
            fig_b = make_heatmap(grid2, x_ax2, y_ax2,
                                 f"{t17} — {_dataset_display(dataset2)}", russo_cmap, "H (m)", **hm17)
            s1, n1, a1, cr1, e1 = q17_area_niveles(meta, hora, bbox)
            s2, n2, a2, cr2, e2 = q17_area_niveles(get_meta(dataset2), hora, bbox)
            ca1, ca2 = st.columns(2)
            for _lbl, _v1, _v2 in [
                (_t("q17_shallow"),  s1,  s2),
                (_t("q17_children"), n1,  n2),
                (_t("q17_adults"),   a1,  a2),
                (_t("q17_critical"), cr1, cr2),
                (_t("q17_extreme"),  e1,  e2),
            ]:
                cmetric(ca1, f"{_dataset_display(dataset)} — {_lbl}",
                        f"{_v1:.1f} km²")
                cmetric(ca2, f"{_dataset_display(dataset2)} — {_lbl}",
                        f"{_v2:.1f} km²",
                        f"{(_v2 - _v1):+.1f} km²")
            with ca1:
                ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
                download_geotiff_button(grid,  x_ax,  y_ax,  "q17", dataset,  hora,
                                        ds_label=_dataset_display(dataset))
            with ca2:
                ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
                download_geotiff_button(grid2, x_ax2, y_ax2, "q17", dataset2, hora,
                                        ds_label=_dataset_display(dataset2))
        else:
            somera, ninos, adultos, critico, extremo = q17_area_niveles(meta, hora, bbox)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(_t("q17_shallow"),  f"{somera:.1f} km²")
            c2.metric(_t("q17_children"), f"{ninos:.1f} km²")
            c3.metric(_t("q17_adults"),   f"{adultos:.1f} km²", help=_t("q17_adults_help"))
            c4.metric(_t("q17_critical"), f"{critico:.1f} km²")
            c5.metric(_t("q17_extreme"),  f"{extremo:.1f} km²")
            show(make_heatmap(grid, x_ax, y_ax, t17, russo_cmap, "H (m)", **hm17))
            download_geotiff_button(grid, x_ax, y_ax, "q17", dataset, hora)
        st.caption(_t("q17_caption"))

    elif qid == "q16":
        with st.spinner(_t("calc_evacuation")):
            grid, x_ax, y_ax, stats = _c_q16(dataset, umbral_h16, umbral_q16, bbox)
        progress.progress(1.0, text=_t("completed"))
        bands, zmax = _bands_ventana(meta["n_steps"])
        t16 = _t("q16_map_title").format(h=umbral_h16, q=umbral_q16)
        hw16 = dict(hover_fmt=".0f", hover_label=_t("hover_window"), hover_unit="h")
        if comparar and dataset2:
            grid2, x_ax2, y_ax2, stats2 = _c_q16(dataset2, umbral_h16, umbral_q16, bbox)
            fig_a = heatmap_discrete(grid,  x_ax,  y_ax,
                                     f"{t16} — {_dataset_display(dataset)}",
                                     bands, zmax, "h", **hw16)
            fig_b = heatmap_discrete(grid2, x_ax2, y_ax2,
                                     f"{t16} — {_dataset_display(dataset2)}",
                                     bands, zmax, "h", **hw16)
            ca1, ca2 = st.columns(2)
            cmetric(ca1, f"{_dataset_display(dataset)} — {_t('mean_window')}", f"{stats['media']:.1f} h")
            cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('mean_window')}", f"{stats2['media']:.1f} h",
                    f"{stats2['media']-stats['media']:+.1f} h")
            ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid,  x_ax,  y_ax,  "q16", dataset,
                                    ds_label=_dataset_display(dataset))
            ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid2, x_ax2, y_ax2, "q16", dataset2,
                                    ds_label=_dataset_display(dataset2))
        else:
            pct_peli = stats['ya_peli'] / stats['n'] * 100 if stats['n'] > 0 else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(_t("min_window"),     f"{stats['min']:.0f} h")
            c2.metric(_t("mean_window"),    f"{stats['media']:.1f} h")
            c3.metric(_t("already_danger"), f"{pct_peli:.1f}%")
            c4.metric(_t("never_danger"),   f"{stats['nunca']:,}")
            show(heatmap_discrete(grid, x_ax, y_ax, t16, bands, zmax, "h", **hw16))
            download_geotiff_button(grid, x_ax, y_ax, "q16", dataset)
        st.caption(_t("q16_caption"))

    # ── Q10a/b/c-Xia: inestabilidad según Xia et al. (2014/2022) ────────────
    elif qid in ("q10a_xia", "q10b_xia", "q10c_xia"):
        tipo_xia = {"q10a_xia": "adultos", "q10b_xia": "ninos", "q10c_xia": "vehiculos"}[qid]
        progress.progress(0.3, text=_t("loading_step"))
        grid, x_ax, y_ax, stats = q10_xia(meta, hora, tipo_xia, bbox)
        progress.progress(1.0, text=_t("completed"))
        cmap_xia = xia_risk_colorscale()
        hm_kwargs = dict(zmin=0.0, zmax=2.0,
                         cbar_tickvals=[0.33, 1.0, 1.67],
                         cbar_ticktext=[_t("xia_safe"), _t("xia_moderate"), _t("xia_high")],
                         hover_label="nivel", hover_unit="", zsmooth=False)
        if tipo_xia == "vehiculos":
            base_title = _t("xia_vehiculos_title").format(hora=hora)
            caption = _t("xia_caption_vehiculos")
        else:
            tipo_label = "adultos" if tipo_xia == "adultos" else "niños"
            base_title = _t("xia_personas_title").format(tipo=tipo_label, hora=hora)
            caption = _t("xia_caption_personas").format(tipo=tipo_label)
        if comparar and dataset2:
            grid2, x_ax2, y_ax2, stats2 = q10_xia(get_meta(dataset2), hora, tipo_xia, bbox)
            fig_a = make_heatmap(grid,  x_ax,  y_ax,
                                 f"{base_title} — {_dataset_display(dataset)}",
                                 cmap_xia, "riesgo", **hm_kwargs)
            fig_b = make_heatmap(grid2, x_ax2, y_ax2,
                                 f"{base_title} — {_dataset_display(dataset2)}",
                                 cmap_xia, "riesgo", **hm_kwargs)
            ca1, ca2 = st.columns(2)
            cmetric(ca1, f"{_dataset_display(dataset)} — {_t('xia_high')}",
                    f"{stats.get('area_alto_km2',0):.1f} km²")
            cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('xia_high')}",
                    f"{stats2.get('area_alto_km2',0):.1f} km²",
                    f"{stats2.get('area_alto_km2',0)-stats.get('area_alto_km2',0):+.1f} km²")
            ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid,  x_ax,  y_ax,  qid, dataset,  hora,
                                    ds_label=_dataset_display(dataset))
            ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid2, x_ax2, y_ax2, qid, dataset2, hora,
                                    ds_label=_dataset_display(dataset2))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric(_t("xia_safe"),     f"{stats.get('area_seguro_km2', 0):.1f} km²")
            c2.metric(_t("xia_moderate"), f"{stats.get('area_moderado_km2', 0):.1f} km²")
            c3.metric(_t("xia_high"),     f"{stats.get('area_alto_km2', 0):.1f} km²")
            show(make_heatmap(grid, x_ax, y_ax, base_title, cmap_xia, "riesgo", **hm_kwargs))
        st.caption(caption)
        if not comparar:
            download_geotiff_button(grid, x_ax, y_ax, qid, dataset, hora)

    # ── Q18: zona de graves daños (RD 9/2008) ───────────────────────────────
    elif qid == "q18":
        progress.progress(0.3, text=_t("loading_step"))
        grid, x_ax, y_ax, stats = q18_graves_danos(meta, hora, bbox)
        progress.progress(1.0, text=_t("completed"))
        cmap18 = [[0.0, "#ff9999"], [1.0, "#cc0000"]]
        t18 = _t("q18_title").format(hora=hora)
        if comparar and dataset2:
            grid2, x_ax2, y_ax2, stats2 = q18_graves_danos(get_meta(dataset2), hora, bbox)
            fig_a = make_heatmap(grid, x_ax, y_ax,
                f"{t18} — {_dataset_display(dataset)}", cmap18, "H (m)",
                hover_label="H", hover_unit="m", zsmooth=False)
            fig_b = make_heatmap(grid2, x_ax2, y_ax2,
                f"{t18} — {_dataset_display(dataset2)}", cmap18, "H (m)",
                hover_label="H", hover_unit="m", zsmooth=False)
            ca1, ca2 = st.columns(2)
            _aA = stats.get('area_km2', 0); _aB = stats2.get('area_km2', 0)
            _s18 = "+" if _aB >= _aA else ""
            cmetric(ca1, f"{_dataset_display(dataset)} — {_t('q18_area')}", f"{_aA:.1f} km²")
            cmetric(ca2, f"{_dataset_display(dataset2)} — {_t('q18_area')}",
                    f"{_aB:.1f} km²", f"{_s18}{_aB-_aA:.1f} km²")
            ca1.plotly_chart(fig_a, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid, x_ax, y_ax, "q18", dataset, hora,
                                    ds_label=_dataset_display(dataset))
            ca2.plotly_chart(fig_b, width="stretch", config=PLOT_CONFIG, theme=None)
            download_geotiff_button(grid2, x_ax2, y_ax2, "q18", dataset2, hora,
                                    ds_label=_dataset_display(dataset2))
        else:
            st.metric(_t("q18_area"), f"{stats.get('area_km2', 0):.1f} km²")
            show(make_heatmap(grid, x_ax, y_ax, t18, cmap18, "H (m)",
                hover_label="H", hover_unit="m", zsmooth=False))
        st.caption(_t("q18_caption"))
        if not comparar:
            download_geotiff_button(grid, x_ax, y_ax, "q18", dataset, hora)

    progress.empty()
    elapsed = time.perf_counter() - t_start
    st.sidebar.success(f"✓ {elapsed:.1f} s")

except Exception as e:
    progress.empty()
    st.error(f"{_t('error_query')}: {e}")
    if os.environ.get("SPILLWAY_DEBUG"):
        st.exception(e)
