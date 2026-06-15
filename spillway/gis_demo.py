"""Genera el paquete demostrativo GIS: COG de H + QML + proyecto .qgz."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiledb
from config import BASE_URI, resolve_dataset
from export_to_geotiff import export_step
from gis_style import write_qml
from gis_package import build_qgz


def main() -> None:
    ap = argparse.ArgumentParser(description="Paquete demostrativo GIS (COG+QML+qgz)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--step", default="10_00")
    ap.add_argument("--out", default="gis_demo")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    uri = f"{BASE_URI}/{resolve_dataset(args.dataset)}"
    with tiledb.open(uri, mode="r") as A:
        meta = dict(A.meta)
        steps = json.loads(meta["time_steps"])
        n_rows = int(meta["height"]); n_cols = int(meta["width"])
    idx = steps.index(args.step)

    export_step(uri, meta, idx, args.step, ["H"], 0, n_rows, 0, n_cols, out, cog=True)
    cog_name = f"H_{args.step}.tif"
    write_qml(str(out / f"H_{args.step}.qml"))   # mismo stem que el .tif -> QGIS lo aplica
    build_qgz(str(out / "triton_calado.qgz"), cog_filename=cog_name,
              layer_name=f"Calado {args.step}")
    print(f"\nPaquete GIS en: {out.resolve()}")


if __name__ == "__main__":
    main()
