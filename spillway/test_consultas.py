#!/usr/bin/env python3
"""
test_consultas.py — Tests del motor de consultas sobre un array TileDB sintético.

Array de prueba: 3 pasos × 10×10 celdas con valores conocidos, calculables a mano.
Incluye una celda en la última fila/columna (9,9) como regresión del off-by-one
en los slices TileDB (el extremo superior es exclusivo).

Ejecución:
  cd /home/ebald/spillway
  /home/ebald/venvs/tiledb_env/bin/python -m pytest test_consultas.py -v
"""
import shutil
import tempfile

import numpy as np
import pytest
import tiledb

import consultas_analiticas as ca

# Dominio sintético: XLL=0, YLL=0, cellsize=10 → x∈[0,100), y∈[0,100)
NR, NC, NT = 10, 10, 3

# Celdas húmedas por paso: (t, row, col, H, QX, QY)
CELLS = [
    (0, 0, 0, 0.02, 0.0, 0.0),
    (0, 9, 9, 0.50, 0.3, 0.4),   # última fila/col — regresión off-by-one; Q_mod=0.5
    (1, 0, 0, 0.10, 0.0, 0.0),
    (1, 5, 5, 1.20, 0.0, 0.0),
    (1, 9, 9, 0.60, 0.0, 0.0),
    (2, 5, 5, 0.30, 0.0, 0.0),
]


@pytest.fixture(scope="module")
def array_sintetico():
    tmp = tempfile.mkdtemp()
    uri = f"{tmp}/sintetico"
    dom = tiledb.Domain(
        tiledb.Dim("time", domain=(0, NT - 1), tile=1,  dtype=np.int32),
        tiledb.Dim("row",  domain=(0, NR - 1), tile=NR, dtype=np.int32),
        tiledb.Dim("col",  domain=(0, NC - 1), tile=NC, dtype=np.int32),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        attrs=[tiledb.Attr(a, dtype=np.float32) for a in ("H", "QX", "QY")],
        sparse=True,
    )
    tiledb.SparseArray.create(uri, schema)
    arr = np.array(CELLS, dtype=np.float32)
    with tiledb.open(uri, "w") as A:
        A[arr[:, 0].astype(np.int32), arr[:, 1].astype(np.int32),
          arr[:, 2].astype(np.int32)] = {
            "H": arr[:, 3], "QX": arr[:, 4], "QY": arr[:, 5],
        }

    # Apuntar el motor al array sintético
    ca.TILEDB_URI = uri
    ca.NROWS, ca.NCOLS, ca.N_STEPS = NR, NC, NT
    ca.CELLSIZE, ca.CELL_AREA = 10.0, 100.0
    ca.XLLCORNER, ca.YLLCORNER = 0.0, 0.0

    yield uri
    shutil.rmtree(tmp)


# ── Conversiones ──────────────────────────────────────────────

def test_hora_a_t_valida():
    assert ca.hora_a_t(6) == 0
    assert ca.hora_a_t(12) == 1


def test_hora_a_t_no_multiplo(array_sintetico):
    with pytest.raises(ValueError):
        ca.hora_a_t(7)


def test_hora_a_t_fuera_de_rango(array_sintetico):
    with pytest.raises(ValueError):
        ca.hora_a_t(24)   # N_STEPS=3 → máx hora 18


def test_coord_pixel_roundtrip(array_sintetico):
    x, y = ca.pixel_a_coord(9, 9)
    assert ca.coord_a_pixel(x, y) == (9, 9)


# ── Q1/Q2: consultas puntuales ────────────────────────────────

def test_q1_celda_borde(array_sintetico):
    x, y = ca.pixel_a_coord(9, 9)
    assert ca.calado_en_punto(x, y, 6) == pytest.approx(0.50)


def test_q1_celda_seca(array_sintetico):
    x, y = ca.pixel_a_coord(3, 3)
    assert ca.calado_en_punto(x, y, 6) == 0.0


def test_q2_serie_temporal(array_sintetico):
    x, y = ca.pixel_a_coord(9, 9)
    st = ca.serie_temporal_punto(x, y)
    assert st["horas"] == [6, 12, 18]
    assert st["H"] == pytest.approx([0.50, 0.60, 0.0])


# ── Q3: velocidad ─────────────────────────────────────────────

def test_q3_velocidad(array_sintetico):
    # En (9,9) a hora 6: Q_mod = sqrt(0.3²+0.4²) = 0.5; V = 0.5/0.5 = 1.0
    v = ca.velocidad_en_zona(0, 0, 100, 100, 6)
    assert v["V_max"] == pytest.approx(1.0)


# ── Q4/Q5: zonas inundadas (incluye regresión off-by-one) ─────

def test_q4_incluye_ultima_fila_col(array_sintetico):
    zi = ca.zonas_inundadas(6)
    assert zi["n_celdas"] == 2          # (0,0) y (9,9)
    assert zi["area_m2"] == pytest.approx(200.0)
    assert (9, 9) in set(zip(zi["rows"].tolist(), zi["cols"].tolist()))


def test_q4_umbral_alto_pushdown(array_sintetico):
    # umbral > H_WET activa el filtro QueryCondition en el motor
    zi = ca.zonas_inundadas(6, umbral_m=0.30)
    assert zi["n_celdas"] == 1          # solo (9,9) con H=0.50
    assert zi["rows"].tolist() == [9] and zi["cols"].tolist() == [9]


def test_q5_evolucion(array_sintetico):
    ev = ca.evolucion_extension()
    assert ev["horas"] == [6, 12, 18]
    assert ev["n_celdas"] == [2, 3, 1]


# ── Q7/Q8/Q9: indicadores temporales por píxel ────────────────

def test_q7_hora_llegada(array_sintetico):
    lf = ca.hora_llegada_frente()
    llegada = dict(zip(zip(lf["rows"].tolist(), lf["cols"].tolist()),
                       lf["hora_llegada"].tolist()))
    assert llegada == {(0, 0): 6, (9, 9): 6, (5, 5): 12}


def test_q8_duracion(array_sintetico):
    dur = ca.duracion_inundacion()
    horas = dict(zip(zip(dur["rows"].tolist(), dur["cols"].tolist()),
                     dur["horas_inundado"].tolist()))
    assert horas == {(0, 0): 12, (9, 9): 12, (5, 5): 12}


def test_q8_umbral_alto_pushdown(array_sintetico):
    dur = ca.duracion_inundacion(umbral_m=0.30)
    horas = dict(zip(zip(dur["rows"].tolist(), dur["cols"].tolist()),
                     dur["horas_inundado"].tolist()))
    assert horas == {(9, 9): 12, (5, 5): 12}   # (0,0) nunca llega a 0.30


def test_q9_hora_calado_maximo(array_sintetico):
    pico = ca.hora_calado_maximo()
    res = dict(zip(zip(pico["rows"].tolist(), pico["cols"].tolist()),
                   pico["hora_pico"].tolist()))
    assert res == {(0, 0): 12, (5, 5): 12, (9, 9): 12}
    mh = dict(zip(zip(pico["rows"].tolist(), pico["cols"].tolist()),
                  pico["MH"].tolist()))
    assert mh[(5, 5)] == pytest.approx(1.20)


# ── Q13/Q14/Q15: estadísticos ─────────────────────────────────

def test_q13_volumen(array_sintetico):
    vol = ca.volumen_por_hora()
    # t=1: (0.10 + 1.20 + 0.60) × 100 m² = 190 m³
    assert vol["volumen_m3"][1] == pytest.approx(190.0)


def test_q14_stats_zona(array_sintetico):
    s = ca.stats_zona(0, 0, 100, 100, 12)
    assert s["n_celdas"] == 3
    assert s["H_max"] == pytest.approx(1.20)
    assert s["H_media"] == pytest.approx((0.10 + 1.20 + 0.60) / 3)


def test_q15_semaforo(array_sintetico):
    # t=1: H = 0.10 (verde ≤0.25), 1.20 (rojo >0.5), 0.60 (rojo)
    np_ = ca.area_por_nivel_peligro(12)
    assert np_["verde_km2"] == pytest.approx(1 * 100 / 1e6)
    assert np_["amarillo_km2"] == 0.0
    assert np_["rojo_km2"] == pytest.approx(2 * 100 / 1e6)
