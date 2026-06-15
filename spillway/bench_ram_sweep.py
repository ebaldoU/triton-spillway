"""Barrido de RAM: corre benchmark_queries.py bajo limites de memoria con
systemd-run y registra tiempo / OOM por nivel."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def classify_run(returncode: int) -> dict:
    # SIGKILL del OOM-killer del cgroup: subprocess lo devuelve como -9, y a
    # traves de un shell como 137 (128 + 9). Ambos significan agotamiento de RAM.
    if returncode == 0:
        return {"status": "ok"}
    if returncode in (137, -9):
        return {"status": "oom"}
    return {"status": "error"}


def run_level(dataset: str, mem_gb: int) -> dict:
    cmd = ["systemd-run", "--user", "--scope",
           "-p", f"MemoryMax={mem_gb}G", "-p", "MemorySwapMax=0",
           PY, str(HERE / "benchmark_queries.py"), "--dataset", dataset]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    res = classify_run(proc.returncode)
    res.update({"mem_gb": mem_gb, "elapsed_s": elapsed, "returncode": proc.returncode})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Barrido de RAM con systemd-run")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[12, 8, 6, 4, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    niveles = [run_level(args.dataset, g) for g in args.levels]
    for n in niveles:
        print(f"  {n['mem_gb']:>2} GB -> {n['status']:<6} {n['elapsed_s']:.1f}s")
    payload = {"tipo": "ram_sweep", "dataset": args.dataset,
               "hostname": socket.gethostname(), "niveles": niveles}
    out = args.out or f"ram_sweep_{socket.gethostname()}.json"
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"JSON escrito en {out}")


if __name__ == "__main__":
    main()
