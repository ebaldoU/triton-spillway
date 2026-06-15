"""Orquesta benchmark_vs_geotiff y benchmark_queries sobre N datasets y
consolida los JSON en bench_<hostname>.json."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def _run(script: str, dataset: str, out: str) -> dict:
    subprocess.run([PY, str(HERE / script), "--dataset", dataset, "--json-out", out],
                   check=True)
    return json.loads(Path(out).read_text())


def consolidate(json_paths: list[str]) -> dict:
    runs = [json.loads(Path(p).read_text()) for p in json_paths]
    maquina = runs[0]["maquina"] if runs else {"hostname": socket.gethostname()}
    return {"maquina": maquina, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser(description="Runner multi-dataset de benchmarks")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", default=None,
                    help="Fichero consolidado (defecto: bench_<hostname>.json)")
    args = ap.parse_args()

    paths = []
    with tempfile.TemporaryDirectory() as tmp:
        for ds in args.datasets:
            vg = f"{tmp}/vg_{ds}.json"; q = f"{tmp}/q_{ds}.json"
            _run("benchmark_vs_geotiff.py", ds, vg)
            _run("benchmark_queries.py", ds, q)
            paths += [vg, q]
        consolidated = consolidate(paths)

    out = args.out or f"bench_{socket.gethostname()}.json"
    Path(out).write_text(json.dumps(consolidated, indent=2))
    print(f"Consolidado en {out} ({len(consolidated['runs'])} runs)")


if __name__ == "__main__":
    main()
