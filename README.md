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
git clone https://github.com/ebaldoU/triton-spillway.git
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

| Variable | Apunta a… | Debe contener | Defecto |
|---|---|---|---|
| `TRITON_BASE_URI` | El directorio `triton_results/` | Un subdirectorio por dataset (arrays TileDB, ~17-21 GB cada uno) | `<repo>/triton_results` (hermano de `spillway/`) |
| `TRITON_GTIFF_DIR` | El directorio raíz de los GeoTIFF | Un subdirectorio por escenario, nombrado con el alias definido en `config.py` (p.ej. `datos1/`), con 80 ficheros `.tif` cada uno (4 variables × 20 pasos) | Dos niveles por encima de `TRITON_BASE_URI` |
| `TRITON_OUTPUT_DIR` | El directorio de exportación | Se crea automáticamente; recibe los GeoTIFF exportados por `export_to_geotiff.py` | `<TRITON_BASE_URI>/../export` |

Estructura de directorios esperada:

```
/ruta/a/datos/
  triton_results/              ← TRITON_BASE_URI
    output_10_..._datos1/      ← array TileDB de datos1 (~17 GB)
    output_10_..._datos2/
    ...
  <alias>/                       ← TRITON_GTIFF_DIR/<alias>
    H_06_00.tif                  (calado, paso 1)
    QX_06_00.tif                 (caudal X, paso 1)
    QY_06_00.tif                 (caudal Y, paso 1)
    MH_06_00.tif                 (calado máximo, paso 1)
    H_12_00.tif  ...             (20 pasos × 4 variables = 80 ficheros)
  export/                      ← TRITON_OUTPUT_DIR (opcional)
```

> `TRITON_GTIFF_DIR` solo se necesita durante el ETL. Una vez generados los arrays TileDB, la aplicación web solo usa `TRITON_BASE_URI`.

---

## Uso

### 1. Configurar rutas

Exportar las variables de entorno antes de ejecutar cualquier comando:

```bash
export TRITON_BASE_URI=/ruta/a/datos/triton_results
export TRITON_GTIFF_DIR=/ruta/a/datos        # solo necesario para el ETL
export TRITON_OUTPUT_DIR=/ruta/a/datos/export # opcional, para export_to_geotiff.py
```

### 2. Cargar datos (ETL)

Por cada dataset disponible, ejecutar el proceso de ingesta:

```bash
cd spillway
python geotiff_to_tiledb_sparse.py --dataset datos1
python geotiff_to_tiledb_sparse.py --dataset datos2
# ...
```

Lee los GeoTIFF de `TRITON_GTIFF_DIR/datos1/` y crea el array TileDB en `TRITON_BASE_URI/`. Tarda ~10 min por dataset con un pico de ~4 GB de RAM. Al finalizar verifica la integridad automáticamente.

### 3. Arrancar la aplicación

```bash
python -m streamlit run app.py
```

Abre `http://localhost:8501` en el navegador. La aplicación detecta automáticamente todos los datasets disponibles en `TRITON_BASE_URI`.

Cada consulta espacial incluye un botón para descargar el resultado como GeoTIFF directamente desde la interfaz.

### Exportación en bulk a GeoTIFF (opcional)

Para exportar varios pasos o variables de golpe desde línea de comandos:

```bash
# Exportar el calado (H) del paso 10_00
python export_to_geotiff.py --dataset datos1 --paso 10_00 --var H

# Exportar todas las variables de un paso
python export_to_geotiff.py --dataset datos1 --paso 10_00 --var all

# Especificar directorio de salida
python export_to_geotiff.py --dataset datos1 --paso 10_00 --var H --out /ruta/salida
```

Los ficheros se guardan en `TRITON_OUTPUT_DIR` (~257 MB por variable y paso).

---

## Datos

Los arrays TileDB y los GeoTIFF fuente no se distribuyen en este repositorio. Son generados por el simulador hidráulico Triton como parte del proyecto de investigación. Para reproducir el entorno completo:

1. Obtener los GeoTIFF fuente del simulador Triton.
2. Ejecutar el ETL para cada dataset: `python geotiff_to_tiledb_sparse.py --dataset datosN`.
3. Arrancar la aplicación con `TRITON_BASE_URI` apuntando al directorio `triton_results/`.

---

## Despliegue en servidor (producción)

La aplicación puede desplegarse como servicio `systemd` en cualquier servidor Linux. El fichero de unidad debe definir `TRITON_BASE_URI` y ejecutar `streamlit run app.py` en modo headless. Una vez el servicio está activo, se accede mediante un túnel SSH:

```bash
ssh -L 8501:localhost:8501 usuario@servidor
# Luego abrir http://localhost:8501
```

---

## Licencia

Proyecto académico — TFG, EINA, Universidad de Zaragoza.
