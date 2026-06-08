"""
Validación funcional de consultas mediante comparación interna:
resultado del motor analítico vs recálculo independiente sobre los
mismos datos TileDB con NumPy puro (sin pasar por las funciones del motor).
"""
import sys, os, json, random
sys.path.insert(0, "/home/ebald/spillway")

import numpy as np
import tiledb
import consultas_analiticas as ca
from config import BASE_URI, resolve_dataset

DATASET = "datos7"

def setup(alias):
    uri = f"{BASE_URI}/{resolve_dataset(alias)}"
    with tiledb.open(uri, mode="r") as A:
        m  = A.meta
        tr = json.loads(m["transform"])
        ca.NCOLS     = int(m["width"])
        ca.NROWS     = int(m["height"])
        ca.CELLSIZE  = float(tr[0])
        ca.CELL_AREA = ca.CELLSIZE ** 2
        ca.XLLCORNER = float(tr[2])
        ca.YLLCORNER = float(tr[5]) + float(tr[4]) * ca.NROWS
        ca.N_STEPS   = len(json.loads(m["time_steps"]))
    ca.TILEDB_URI = uri
    return uri

uri = setup(DATASET)
errores = 0
print(f"Dataset: {DATASET}  ({ca.NROWS}×{ca.NCOLS}, {ca.N_STEPS} pasos)\n")

# ─────────────────────────────────────────────────────────────
# Q4 — zonas inundadas: recuento de celdas húmedas en un paso
# Motor: ca.zonas_inundadas(hora)
# Oráculo: lectura directa TileDB con el mismo rango que usa el motor
# Nota: el motor usa slice [0:NROWS-1, 0:NCOLS-1] (excluye última fila/col),
# por eso el oráculo replica el mismo rango para comparación justa.
# ─────────────────────────────────────────────────────────────
print("=== Q4: zonas inundadas (recuento celdas húmedas) ===")
for hora in [6, 10, 16, 20]:
    t = ca.hora_a_t(hora)
    r_motor = ca.zonas_inundadas(hora)
    n_motor = r_motor["n_celdas"]
    with tiledb.open(uri, mode="r") as A:
        res = A.query(attrs=["H"])[t, 0:ca.NROWS-1, 0:ca.NCOLS-1]
    n_oracle = int((res["H"] >= ca.H_WET).sum())
    diff = abs(n_motor - n_oracle)
    ok = diff == 0
    print(f"  hora={hora}: motor={n_motor:,}  oráculo={n_oracle:,}  diff={diff}  {'OK' if ok else 'FALLO'}")
    if not ok:
        errores += 1

# ─────────────────────────────────────────────────────────────
# Q5 — evolución de extensión: recuento por los 20 pasos
# ─────────────────────────────────────────────────────────────
print("\n=== Q5: evolución de extensión (20 pasos) ===")
r_motor = ca.evolucion_extension()
diffs = []
for i, hora in enumerate(r_motor["horas"]):
    t = ca.hora_a_t(hora)
    with tiledb.open(uri, mode="r") as A:
        res = A.query(attrs=["H"])[t, 0:ca.NROWS-1, 0:ca.NCOLS-1]
    n_oracle = int((res["H"] >= ca.H_WET).sum())
    n_motor  = r_motor["n_celdas"][i]
    diff = abs(n_motor - n_oracle)
    diffs.append(diff)
max_diff = max(diffs)
ok = max_diff == 0
print(f"  20 pasos comparados. Diferencia máxima: {max_diff} celda(s)  {'OK' if ok else 'FALLO'}")
if not ok:
    errores += 1

# ─────────────────────────────────────────────────────────────
# Q13 — volumen por hora: suma de H*cellsize² por paso
# ─────────────────────────────────────────────────────────────
print("\n=== Q13: volumen por hora (20 pasos) ===")
r_motor = ca.volumen_por_hora()
diffs_vol = []
for i, hora in enumerate(r_motor["horas"]):
    t = ca.hora_a_t(hora)
    with tiledb.open(uri, mode="r") as A:
        res = A.query(attrs=["H"])[t, 0:ca.NROWS-1, 0:ca.NCOLS-1]
    H_wet = res["H"][res["H"] >= ca.H_WET]
    vol_oracle = float(H_wet.sum()) * ca.CELL_AREA / 1e6  # Mm³
    vol_motor  = r_motor["volumen_m3"][i] / 1e6            # m³ → Mm³
    err_rel = abs(vol_motor - vol_oracle) / max(vol_oracle, 1e-9) * 100
    diffs_vol.append(err_rel)
max_err = max(diffs_vol)
ok = max_err < 0.01
print(f"  20 pasos. Error relativo máximo: {max_err:.5f}%  {'OK' if ok else 'FALLO'}")
if not ok:
    errores += 1

# ─────────────────────────────────────────────────────────────
# Q1 — calado en punto: valor en celda específica
# ─────────────────────────────────────────────────────────────
print("\n=== Q1: calado en punto (50 puntos aleatorios) ===")
random.seed(42)
HORA_REF = 66   # hora=66 → t = 66//6 - 1 = 10
T_REF = ca.hora_a_t(HORA_REF)
with tiledb.open(uri, mode="r") as A:
    res_ref = A.query(attrs=["H"], coords=True)[T_REF, :, :]
wet_idx = np.where(res_ref["H"] >= ca.H_WET)[0]
sample_idx = random.sample(list(wet_idx[:5000]), 50)

discrepancias = 0
for idx in sample_idx:
    row = int(res_ref["row"][idx])
    col = int(res_ref["col"][idx])
    h_ref = float(res_ref["H"][idx])
    x = ca.XLLCORNER + (col + 0.5) * ca.CELLSIZE
    y = ca.YLLCORNER + (ca.NROWS - row - 0.5) * ca.CELLSIZE
    h_motor = ca.calado_en_punto(x, y, hora=HORA_REF)
    if abs(h_motor - h_ref) > 1e-4:
        discrepancias += 1

print(f"  50 puntos en paso 10. Discrepancias (>1e-4 m): {discrepancias}  {'OK' if discrepancias == 0 else 'FALLO'}")
if discrepancias > 0:
    errores += 1

# ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"TOTAL errores: {errores}")
print(f"{'='*50}")
