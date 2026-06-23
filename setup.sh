#!/usr/bin/env bash
#
# setup.sh — Instalación completa de Spillway en un solo comando.
#
# Crea el entorno virtual, instala las dependencias y descarga los datos de
# demostración (datos1 y datos2) en triton_results/, que es la ruta por defecto
# que usa la aplicación. Al terminar, la app arranca sin configurar nada.
#
# Uso:
#   ./setup.sh            # entorno + dependencias + datos de demostración
#   ./setup.sh --no-data  # solo entorno + dependencias (sin descargar datos)
#   ./setup.sh --help
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/venv"
DATA_DIR="$REPO_DIR/triton_results"
RELEASE_URL="https://github.com/ebaldoU/triton-spillway/releases/download/demo-data-v1"
DEMO_ASSETS=("demo_datos1_bbox50km.tar.gz" "demo_datos2_bbox50km.tar.gz")

WITH_DATA=1
for arg in "$@"; do
  case "$arg" in
    --no-data) WITH_DATA=0 ;;
    -h|--help)
      sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Opción desconocida: $arg (usa --help)"; exit 1 ;;
  esac
done

say()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '    \033[1;32mOK\033[0m %s\n' "$*"; }
warn() { printf '    \033[1;33m!!\033[0m %s\n' "$*"; }

# ── 1. Python ────────────────────────────────────────────────────────────────
say "Comprobando Python"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 no está instalado. Instálalo (3.10 o superior) y reintenta." >&2
  exit 1
fi
PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ok "python3 $PYVER"

# ── 2. Entorno virtual ───────────────────────────────────────────────────────
say "Entorno virtual"
if [ -d "$VENV_DIR" ]; then
  ok "ya existe ($VENV_DIR), se reutiliza"
else
  python3 -m venv "$VENV_DIR"
  ok "creado en $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 3. Dependencias ──────────────────────────────────────────────────────────
say "Instalando dependencias (requirements.txt)"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "$REPO_DIR/requirements.txt"
ok "dependencias instaladas"

# ── 4. Datos de demostración ─────────────────────────────────────────────────
if [ "$WITH_DATA" -eq 1 ]; then
  say "Datos de demostración"
  mkdir -p "$DATA_DIR"
  for asset in "${DEMO_ASSETS[@]}"; do
    marker="$DATA_DIR/.$asset.done"
    if [ -f "$marker" ]; then
      ok "$asset ya descomprimido, se omite"
      continue
    fi
    tmp="$(mktemp)"
    echo "    descargando $asset ..."
    if curl -fL# "$RELEASE_URL/$asset" -o "$tmp"; then
      tar -xzf "$tmp" -C "$DATA_DIR"
      touch "$marker"
      rm -f "$tmp"
      ok "$asset descomprimido en triton_results/"
    else
      rm -f "$tmp"
      warn "no se pudo descargar $asset desde la release demo-data-v1."
      warn "Descárgalo manualmente y descomprimelo en: $DATA_DIR"
    fi
  done
else
  say "Datos de demostración omitidos (--no-data)"
fi

# ── 5. Configuración de Streamlit ────────────────────────────────────────────
# Evita el aviso interactivo de telemetría ("Welcome to Streamlit!") la primera
# vez, para que la aplicación arranque sin pedir nada por consola.
say "Configurando Streamlit"
ST_CFG="$HOME/.streamlit"
mkdir -p "$ST_CFG"
if [ ! -f "$ST_CFG/credentials.toml" ]; then
  printf '[general]\nemail = ""\n' > "$ST_CFG/credentials.toml"
  ok "aviso de telemetría desactivado"
else
  ok "configuración de Streamlit ya existente, se respeta"
fi

# ── 6. Resumen ───────────────────────────────────────────────────────────────
say "Listo"
cat <<EOF

  Para arrancar la aplicación:

      source venv/bin/activate
      streamlit run spillway/app.py

  Abre http://localhost:8501 e inicia sesión con:

      usuario:     triton
      contraseña:  demo

EOF
