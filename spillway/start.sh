#!/bin/bash
# Arranque de la aplicación web.
# Requiere TRITON_BASE_URI exportado o que config.py lo derive automáticamente.
source "$(dirname "$0")/../venv/bin/activate" 2>/dev/null || true
streamlit run "$(dirname "$0")/app.py" \
  --server.port 8501 \
  --server.headless true \
  --server.address 127.0.0.1
