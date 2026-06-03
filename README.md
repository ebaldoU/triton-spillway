# Triton Spillway — TFG

Sistema de almacenamiento y análisis eficiente de resultados de simulaciones hidráulicas mediante TileDB.

## Estructura

```
spillway/   — Código de la aplicación
  app.py                      # Aplicación web Streamlit
  consultas_analiticas.py     # Motor de 20 consultas analíticas
  geotiff_to_tiledb_sparse.py # ETL: GeoTIFF → TileDB sparse
  config.py                   # Configuración y alias de datasets
  benchmark_*.py              # Scripts de benchmark
  export_to_geotiff.py        # Exportación TileDB → GeoTIFF
  visualize_tiledb.py         # Visualizador matplotlib
  verify_tiledb.py            # Verificación de integridad

memoria/    — Memoria del TFG en LaTeX
```

## Requisitos

```bash
pip install tiledb rasterio numpy streamlit folium plotly
```

## Ejecución local

```bash
cd spillway
TRITON_BASE_URI=/ruta/a/triton_results streamlit run app.py
```

## ETL (GeoTIFF → TileDB)

```bash
cd spillway
python geotiff_to_tiledb_sparse.py --dataset datos1
```

Los datos (arrays TileDB y GeoTIFF fuente) no están incluidos en el repositorio por su tamaño (~17 GB por escenario).

## Autor

Enrique Baldovín Cotela — TFG, EINA, Universidad de Zaragoza, 2026
