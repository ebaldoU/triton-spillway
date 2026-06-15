"""Captura de metadatos de la máquina para los benchmarks (reproducibilidad)."""
from __future__ import annotations

import os
import platform
import re
import socket
import sys
from datetime import datetime, timezone


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocido"


def _ram_total_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(re.search(r"(\d+)", line).group(1))
                    return kb // 1024
    except OSError:
        pass
    return 0


def _disk_type() -> str:
    try:
        for dev in sorted(os.listdir("/sys/block")):
            rot = f"/sys/block/{dev}/queue/rotational"
            if os.path.exists(rot):
                with open(rot) as f:
                    val = f.read().strip()
                if val == "0":
                    return "SSD"
                if val == "1":
                    return "HDD"
    except OSError:
        pass
    return "desconocido"


def _tiledb_version() -> str:
    try:
        import tiledb
        return str(tiledb.version())
    except Exception:
        return "desconocido"


def machine_metadata() -> dict:
    return {
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_total_mb": _ram_total_mb(),
        "disk_type": _disk_type(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "tiledb": _tiledb_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
