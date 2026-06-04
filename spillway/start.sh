#!/bin/bash
source ~/venv_triton/bin/activate
TRITON_BASE_URI=/home/ubuntu/TFG/tiledb/triton_results \
  streamlit run ~/spillway/app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0
