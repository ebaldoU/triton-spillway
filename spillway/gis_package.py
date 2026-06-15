"""Genera un proyecto QGIS .qgz minimo que carga un COG por ruta relativa
con su estilo. El .qgz es un zip que contiene un .qgs (XML)."""
from __future__ import annotations

import zipfile
from pathlib import Path

from gis_style import depth_qml  # noqa: F401  (estilo coherente, reutilizado)


def _qgs_xml(cog_filename: str, layer_name: str) -> str:
    # Proyecto minimo con paths relativos (relativePaths=1) y una capa raster.
    return (
        '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">\n'
        '<qgis projectname="Triton - Calado" version="3.34">\n'
        '  <properties>\n'
        '    <Paths>\n'
        '      <absolute type="bool">false</absolute>\n'
        '    </Paths>\n'
        '  </properties>\n'
        '  <projectlayers>\n'
        '    <maplayer type="raster" name="' + layer_name + '">\n'
        '      <datasource relative="1">./' + cog_filename + '</datasource>\n'
        '      <layername>' + layer_name + '</layername>\n'
        '      <provider>gdal</provider>\n'
        '    </maplayer>\n'
        '  </projectlayers>\n'
        '</qgis>\n'
    )


def build_qgz(qgz_path: str, cog_filename: str, layer_name: str) -> None:
    stem = Path(qgz_path).stem
    xml = _qgs_xml(cog_filename, layer_name)
    with zipfile.ZipFile(qgz_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}.qgs", xml)
    print(f"  -> {qgz_path}")
