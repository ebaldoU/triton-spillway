#!/usr/bin/env python3
"""
benchmark_consultas.py — Mide el tiempo de todas las consultas analíticas
sobre el dataset especificado (por defecto datos29, el más pesado: 21 GB).

Uso:
  TRITON_DATASET=datos29 python benchmark_consultas.py
  TRITON_DATASET=datos8  python benchmark_consultas.py
"""
import os, time
os.environ.setdefault("TRITON_DATASET", "datos29")

import consultas_analiticas as ca

DATASET = os.environ["TRITON_DATASET"]
URI     = ca.TILEDB_URI

# Punto representativo en el centro del dominio
CX = ca.XLLCORNER + ca.NCOLS * ca.CELLSIZE / 2
CY = ca.YLLCORNER + ca.NROWS * ca.CELLSIZE / 2

# BBox: cuadrante central (~25% del dominio)
BX0 = ca.XLLCORNER + ca.NCOLS * ca.CELLSIZE * 0.25
BY0 = ca.YLLCORNER + ca.NROWS * ca.CELLSIZE * 0.25
BX1 = ca.XLLCORNER + ca.NCOLS * ca.CELLSIZE * 0.75
BY1 = ca.YLLCORNER + ca.NROWS * ca.CELLSIZE * 0.75
BBOX = (BX0, BY0, BX1, BY1)

results = []

def bench(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    results.append((label, elapsed))
    print(f"  {elapsed:7.1f}s  {label}")
    return elapsed

print(f"\n=== Benchmark — {DATASET} ({URI.split('/')[-1][:30]}...) ===\n")
print(f"  Dominio: {ca.NROWS}×{ca.NCOLS} px  |  BBox central: 50%×50%\n")

# ── Consultas de un único paso temporal ──────────────────────────
print("[ Consultas de un paso temporal — hora 60 (t=9) ]")
bench("Q4  zonas_inundadas            (dom.)",  ca.zonas_inundadas, hora=60)
bench("Q4  zonas_inundadas            (bbox)",  ca.zonas_inundadas, hora=60, bbox=BBOX)
bench("Q10a serie_temporal_punto",              ca.serie_temporal_punto, CX, CY)
bench("Q13 stats_zona                 (bbox)",  ca.stats_zona, BX0, BY0, BX1, BY1, hora=60)
bench("Q14 peligrosidad_adultos       (dom.)",  ca.peligrosidad_adultos, hora=60)
bench("Q14 peligrosidad_adultos       (bbox)",  ca.peligrosidad_adultos, hora=60, bbox=BBOX)
bench("Q15 area_por_nivel_peligro     (dom.)",  ca.area_por_nivel_peligro, hora=60)
bench("Q15 area_por_nivel_peligro     (bbox)",  ca.area_por_nivel_peligro, hora=60, bbox=BBOX)

# ── Consultas multi-paso (dominio completo y bbox) ───────────────
print("\n[ Consultas multi-paso — 20 pasos × dominio ]")
bench("Q5  evolucion_extension        (dom.)",  ca.evolucion_extension)
bench("Q5  evolucion_extension        (bbox)",  ca.evolucion_extension, bbox=BBOX)
bench("Q12 area_inundada_por_hora     (dom.)",  ca.area_inundada_por_hora)
bench("Q12 area_inundada_por_hora     (bbox)",  ca.area_inundada_por_hora, bbox=BBOX)

# ── Consultas de indicadores acumulados (las más pesadas) ────────
print("\n[ Consultas acumuladas por píxel — las más costosas ]")
bench("Q7  hora_llegada_frente        (dom.)",  ca.hora_llegada_frente)
bench("Q7  hora_llegada_frente        (bbox)",  ca.hora_llegada_frente, bbox=BBOX)
bench("Q8  hora_calado_maximo         (dom.)",  ca.hora_calado_maximo)
bench("Q8  hora_calado_maximo         (bbox)",  ca.hora_calado_maximo, bbox=BBOX)
bench("Q9  ventana_vehiculos_emerg.   (dom.)",  ca.ventana_vehiculos_emergencia)
bench("Q9  ventana_vehiculos_emerg.   (bbox)",  ca.ventana_vehiculos_emergencia, bbox=BBOX)
bench("Q11 duracion_inundacion        (dom.)",  ca.duracion_inundacion)
bench("Q11 duracion_inundacion        (bbox)",  ca.duracion_inundacion, bbox=BBOX)

# ── Resumen ──────────────────────────────────────────────────────
print("\n=== Resumen ===")
print(f"{'Consulta':<45} {'Tiempo (s)':>10}")
print("-" * 57)
for label, t in results:
    print(f"{label:<45} {t:>10.1f}")
total = sum(t for _, t in results)
print(f"\nTotal acumulado: {total:.1f} s  ({total/60:.1f} min)")
