"""Agrega los JSON de benchmark de varias maquinas y genera tablas LaTeX y
figuras para la memoria."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def latex_specs_row(machine: dict) -> str:
    import math
    m = machine["maquina"]
    ram_gb = math.ceil(m["ram_total_mb"] / 1024)
    return (f"{m['hostname']} & {m['cpu_cores']} & {ram_gb} & "
            f"{m['disk_type']} \\\\")


def specs_table(machines: list[dict]) -> str:
    head = (r"\begin{tabular}{lrrl}" "\n" r"\toprule" "\n"
            r"\textbf{Máquina} & \textbf{Núcleos} & \textbf{RAM (GB)} & \textbf{Disco} \\"
            "\n" r"\midrule" "\n")
    body = "\n".join(latex_specs_row(m) for m in machines)
    tail = "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"
    return head + body + tail


def ram_curve_points(sweep: dict) -> list[tuple[int, float, str]]:
    return [(n["mem_gb"], n["elapsed_s"], n["status"]) for n in sweep["niveles"]]


def plot_ram_curve(sweeps: list[dict], out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for sw in sweeps:
        pts = sorted(ram_curve_points(sw))
        xs = [p[0] for p in pts]
        ys = [p[1] if p[2] == "ok" else None for p in pts]
        ax.plot(xs, ys, marker="o", label=sw.get("hostname", "?"))
        for mem, _, status in pts:
            if status == "oom":
                ax.axvline(mem, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("RAM disponible (GB)")
    ax.set_ylabel("Tiempo de consultas (s)")
    ax.set_title("Degradación de rendimiento por RAM disponible")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera tablas/figuras de benchmark")
    ap.add_argument("--machines", nargs="+", required=True,
                    help="Ficheros bench_<hostname>.json")
    ap.add_argument("--sweeps", nargs="*", default=[],
                    help="Ficheros ram_sweep_*.json")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    machines = [json.loads(Path(p).read_text()) for p in args.machines]
    out = Path(args.out_dir)
    (out / "tabla_specs.tex").write_text(specs_table(machines))
    print(f"  -> {out/'tabla_specs.tex'}")
    if args.sweeps:
        sweeps = [json.loads(Path(p).read_text()) for p in args.sweeps]
        plot_ram_curve(sweeps, str(out / "curva_ram.png"))
        print(f"  -> {out/'curva_ram.png'}")


if __name__ == "__main__":
    main()
