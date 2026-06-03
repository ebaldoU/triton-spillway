# Spillway — Análisis de Simulaciones Hidráulicas con TileDB

Sistema de almacenamiento y análisis eficiente para resultados de simulaciones hidráulicas de alta resolución. Convierte los GeoTIFF generados por el simulador Triton (~235 GB por escenario) a arrays TileDB sparse (~17,5 GB, reducción del 92,7 %) y expone 24 consultas analíticas a través de una aplicación web interactiva.

> Trabajo Fin de Grado — EINA, Universidad de Zaragoza, 2026  
> Autor: Enrique Baldovín Cotela  
> Directores: Mario Morales Hernández · Sergio Ilarri Artigas

---

## Descripción

El simulador Triton produce 80 ficheros GeoTIFF por escenario (4 variables × 20 pasos temporales). Este proyecto:

1. **Ingesta** los GeoTIFF en un array TileDB sparse 3D (tiempo × fila × columna), almacenando solo las celdas húmedas (~8,3 % del dominio).
2. **Analiza** los datos con un motor de 24 funciones Python organizadas en 8 bloques (calado, velocidad, peligrosidad Russo/Xia, RD 9/2008, evacuación…).
3. **Visualiza** los resultados en una aplicación web Streamlit con mapas interactivos, modo comparativo multi-escenario y exportación a GeoTIFF.

---

## Estructura del repositorio

```
spillway/
  app.py                       # Aplicación web Streamlit (punto de entrada)
  consultas_analiticas.py      # Motor de 24 consultas analíticas
  config.py                    # Rutas y alias de datasets (configurable por env vars)
  geotiff_to_tiledb_sparse.py  # ETL: GeoTIFF → TileDB sparse
  verify_tiledb.py             # Verificación de integridad TileDB vs GeoTIFF
  export_to_geotiff.py         # Exportación TileDB → GeoTIFF (interoperabilidad SIG)
  visualize_tiledb.py          # Visualizador interactivo matplotlib
  query_tiledb.py              # Estadísticos por paso, área, máximos
  query_max_depth_tiledb.py    # Celda con mayor calado, top-N
  comparar_datasets.py         # Comparativa multi-escenario por línea de comandos
  benchmark_vs_geotiff.py      # Benchmark TileDB vs GeoTIFF (6 casos)
  benchmark_queries.py         # Benchmark 5 patrones de consulta

memoria/
  main.tex                     # Memoria TFG en LaTeX (compilar con pdflatex)
```

> Los datos (arrays TileDB y GeoTIFF fuente) **no están incluidos** en el repositorio por su tamaño. Ver sección [Datos](#datos).

---

## Requisitos del sistema

- Python 3.10 o superior
- 16 GB RAM recomendados (8 GB suficientes para la app web en uso normal)
- ~20 GB de disco por cada dataset procesado (17–21 GB TileDB + GeoTIFFs fuente)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/ebaldovin/triton-spillway.git
cd triton-spillway

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Configuración de rutas

El sistema usa variables de entorno para localizar los datos. No hay rutas absolutas en el código.

| Variable | Descripción | Defecto |
|---|---|---|
| `TRITON_BASE_URI` | Directorio con los arrays TileDB (`triton_results/`) | `<repo>/../triton_results` |
| `TRITON_GTIFF_DIR` | Directorio raíz con los GeoTIFF fuente (`datos1/`, `datos2/`…) | padre de `TRITON_BASE_URI` |
| `TRITON_OUTPUT_DIR` | Directorio de exportación GeoTIFF | `<TRITON_BASE_URI>/../export` |

Ejemplo de estructura de datos esperada:

```
/ruta/a/datos/
  triton_results/          ← TRITON_BASE_URI
    output_10_..._datos1/  ← array TileDB (~17 GB)
    output_10_..._datos2/
    ...
  datos1/                  ← TRITON_GTIFF_DIR/datos1 (GeoTIFF fuente)
    H_06_00.tif
    QX_06_00.tif
    ...
  export/                  ← TRITON_OUTPUT_DIR
```

---

## Uso

Lo más cómodo es exportar las variables de entorno una sola vez al inicio de la sesión:

```bash
export TRITON_BASE_URI=/ruta/a/datos/triton_results
export TRITON_GTIFF_DIR=/ruta/a/datos        # solo necesario para ETL, verify y benchmark_vs_geotiff
export TRITON_OUTPUT_DIR=/ruta/a/datos/export  # opcional, para export_to_geotiff
cd spillway
```

A partir de ahí, todos los scripts funcionan sin más configuración.

---

### Aplicación web

```bash
python -m streamlit run app.py
```

Abre `http://localhost:8501`. Detecta automáticamente todos los datasets disponibles en `TRITON_BASE_URI`.

---

### ETL: convertir GeoTIFF a TileDB

```bash
python geotiff_to_tiledb_sparse.py --dataset datos1
```

Lee los GeoTIFF de `TRITON_GTIFF_DIR/datos1/` y escribe el array TileDB en `TRITON_BASE_URI/output_10_.../`. Tarda ~10 min por dataset con un pico de ~4 GB de RAM. Al finalizar verifica la integridad automáticamente.

Para especificar la ruta de los GeoTIFF manualmente:

```bash
python geotiff_to_tiledb_sparse.py --dataset datos1 --gtiff-dir /otra/ruta/datos1
```

---

### Verificar integridad del array

```bash
python verify_tiledb.py --dataset datos1
python verify_tiledb.py --dataset datos2 --step 10_00 --n 2000
```

Compara 1 000 celdas aleatorias entre TileDB y los GeoTIFF originales (en `TRITON_GTIFF_DIR/datos1/`). Reporta discrepancias y porcentaje de coincidencia.

---

### Visualizador de escritorio

```bash
python visualize_tiledb.py --dataset datos1
python visualize_tiledb.py --dataset datos1 --paso 10_00
python visualize_tiledb.py --dataset datos1 --paso 10_00 H    # solo calado
python visualize_tiledb.py --dataset datos1 --paso 10_00 full # H + V + MH
```

Lee de `TRITON_BASE_URI`. Abre una ventana matplotlib interactiva con panel de cursor en tiempo real.

---

### Estadísticos por paso

```bash
python query_tiledb.py --dataset datos1
```

Imprime por pantalla estadísticas de H, área inundada, máximos y serie temporal para cada uno de los 20 pasos.

---

### Celda con mayor calado (top-N)

```bash
python query_max_depth_tiledb.py --dataset datos1
python query_max_depth_tiledb.py --dataset datos1 --top 10
```

Localiza las N celdas con mayor profundidad e imprime sus coordenadas UTM y valor H.

---

### Exportar a GeoTIFF

```bash
python export_to_geotiff.py --dataset datos1 --paso 10_00 --var H
python export_to_geotiff.py --dataset datos1 --paso 10_00 --var all
python export_to_geotiff.py --dataset datos1 --paso 10_00 --var H --out /tmp/salida
```

Reconstruye el ráster denso desde TileDB y lo guarda como GeoTIFF en `TRITON_OUTPUT_DIR` (o en `--out`). Cada fichero ocupa ~257 MB.

---

### Comparativa multi-escenario (línea de comandos)

```bash
python comparar_datasets.py --datasets datos1 datos2 datos3
```

Compara métricas clave (extensión máxima, hora del pico, volumen) entre varios datasets.

---

### Benchmarks

```bash
# Rendimiento de 5 patrones de consulta TileDB
python benchmark_queries.py --dataset datos1

# Comparativa TileDB vs GeoTIFF (requiere TRITON_GTIFF_DIR con los .tif fuente)
python benchmark_vs_geotiff.py --dataset datos1
```

---

## Datos

Los arrays TileDB y los GeoTIFF fuente no se distribuyen en este repositorio. Son generados por el simulador hidráulico Triton como parte del proyecto de investigación. Para reproducir el entorno completo:

1. Obtener los GeoTIFF fuente del simulador Triton.
2. Ejecutar el ETL para cada dataset: `python geotiff_to_tiledb_sparse.py --dataset datosN`.
3. Arrancar la aplicación con `TRITON_BASE_URI` apuntando al directorio `triton_results/`.

---

## Despliegue en servidor (producción)

La aplicación está desplegada como servicio systemd. Ver [DEPLOY.txt](spillway/DEPLOY.txt) para instrucciones detalladas de despliegue en VPS.

Acceso mediante túnel SSH:

```bash
ssh -L 8501:localhost:8501 ubuntu@<servidor>
# Luego abrir http://localhost:8501
```

---

## Licencia

Proyecto académico — TFG, EINA, Universidad de Zaragoza.
