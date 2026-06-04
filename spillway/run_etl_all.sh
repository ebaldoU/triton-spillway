#!/usr/bin/env bash
# ETL secuencial de todos los datasets en D:
# Salta los que ya tienen 20 fragments completos en TileDB.
# Log general: /tmp/etl_all.log

set -euo pipefail

PYTHON=/home/ebald/venvs/tiledb_env/bin/python
SCRIPT=/home/ebald/spillway/geotiff_to_tiledb_sparse.py
TILEDB_BASE=/home/ebald/TFG/tiledb/triton_results
LOGDIR=/tmp

PREFIX="output_10_HUC1024_ACCESS-CM2_ssp585_r1i1p1f1_RegCM_Daymet"

# Todos los datasets en D: ordenados por fecha
DATASETS=($(ls /mnt/d/ | grep "^${PREFIX}" | sort))

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/etl_all.log"; }

fragments_ok() {
  local uri="$TILEDB_BASE/$1"
  [[ -d "$uri/__fragments" ]] && [[ $(ls "$uri/__fragments" | wc -l) -ge 20 ]]
}

alias_for() {
  # Extrae las fechas del nombre del directorio y busca el alias en config.py
  local dates
  dates=$(echo "$1" | grep -oP '\d{8}_\d{8}$')
  grep "$dates" /home/ebald/spillway/config.py | grep -oP '"datos\d+"' | tr -d '"' | head -1
}

total=${#DATASETS[@]}
done_count=0
skip_count=0
fail_count=0

log "=== ETL de $total datasets ==="

for dir in "${DATASETS[@]}"; do
  alias=$(alias_for "$dir")
  if [[ -z "$alias" ]]; then
    log "WARN: sin alias para $dir, usando nombre completo"
    alias="$dir"
  fi

  if fragments_ok "$dir"; then
    log "SKIP $alias — ya tiene 20 fragments"
    skip_count=$((skip_count + 1))
    continue
  fi

  log "START $alias ($dir)"
  logfile="$LOGDIR/etl_${alias}.log"

  if $PYTHON -u "$SCRIPT" --dataset "$alias" --gtiff-dir "/mnt/d/$dir/gtiff" > "$logfile" 2>&1; then
    log "OK    $alias"
    done_count=$((done_count + 1))
  else
    log "FAIL  $alias — revisa $logfile"
    fail_count=$((fail_count + 1))
  fi
done

log "=== Fin: $done_count OK · $skip_count saltados · $fail_count fallidos ==="
