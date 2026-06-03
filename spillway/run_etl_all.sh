#!/usr/bin/env bash
# ETL secuencial de todos los datasets de un directorio.
# Salta los que ya tienen 20 fragmentos completos en TileDB.
#
# Uso:
#   export TRITON_BASE_URI=/ruta/triton_results
#   export TRITON_GTIFF_DIR=/ruta/datos
#   bash run_etl_all.sh
#
# Log en /tmp/etl_all.log

set -euo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT="$(dirname "$0")/geotiff_to_tiledb_sparse.py"
TILEDB_BASE="${TRITON_BASE_URI:?Debes exportar TRITON_BASE_URI}"
GTIFF_BASE="${TRITON_GTIFF_DIR:?Debes exportar TRITON_GTIFF_DIR}"
LOGDIR=/tmp

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/etl_all.log"; }

fragments_ok() {
  local uri="$TILEDB_BASE/$1"
  [[ -d "$uri/__fragments" ]] && [[ $(ls "$uri/__fragments" | wc -l) -ge 20 ]]
}

# Lista de aliases (datos1, datos2, ...) a procesar
DATASETS=($(python3 -c "
import sys; sys.path.insert(0, '$(dirname "$0")')
from config import DATASET_ALIASES
for k in sorted(DATASET_ALIASES): print(k)
"))

total=${#DATASETS[@]}
done_count=0; skip_count=0; fail_count=0

log "=== ETL de $total datasets ==="

for alias in "${DATASETS[@]}"; do
  dir=$(python3 -c "
import sys; sys.path.insert(0, '$(dirname "$0")')
from config import resolve_dataset
print(resolve_dataset('$alias'))
")

  if fragments_ok "$dir"; then
    log "SKIP $alias — ya tiene 20 fragmentos"
    skip_count=$((skip_count + 1))
    continue
  fi

  log "START $alias"
  logfile="$LOGDIR/etl_${alias}.log"

  if $PYTHON -u "$SCRIPT" --dataset "$alias" > "$logfile" 2>&1; then
    log "OK    $alias"
    done_count=$((done_count + 1))
  else
    log "FAIL  $alias — revisa $logfile"
    fail_count=$((fail_count + 1))
  fi
done

log "=== Fin: $done_count OK · $skip_count saltados · $fail_count fallidos ==="
