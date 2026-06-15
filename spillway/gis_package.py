"""Genera un proyecto QGIS .qgz minimo que carga un COG por ruta relativa
con su estilo. El .qgz es un zip que contiene un .qgs (XML)."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from gis_style import depth_qml  # noqa: F401  (estilo coherente, reutilizado)

# El COG debe ser un nombre de fichero simple (sin separadores ni ..),
# para que el datasource relativo no pueda apuntar fuera del paquete.
_COG_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.tif$")


def _qgs_xml(cog_filename: str, layer_name: str) -> str:
    # Proyecto minimo con paths relativos (relativePaths=1) y una capa raster.
    # Los valores interpolados se escapan para no romper el XML.
    name_attr = quoteattr(layer_name)        # incluye las comillas
    name_text = escape(layer_name)
    cog_text = escape(cog_filename)
    return (
        '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">\n'
        '<qgis projectname="Triton - Calado" version="3.34">\n'
        '  <properties>\n'
        '    <Paths>\n'
        '      <absolute type="bool">false</absolute>\n'
        '    </Paths>\n'
        '  </properties>\n'
        '  <projectlayers>\n'
        '    <maplayer type="raster" name=' + name_attr + '>\n'
        '      <datasource relative="1">./' + cog_text + '</datasource>\n'
        '      <layername>' + name_text + '</layername>\n'
        '      <provider>gdal</provider>\n'
        '    </maplayer>\n'
        '  </projectlayers>\n'
        '</qgis>\n'
    )


def build_qgz(qgz_path: str, cog_filename: str, layer_name: str) -> None:
    if not _COG_NAME_RE.match(cog_filename):
        raise ValueError(
            f"cog_filename invalido: {cog_filename!r} "
            "(se espera un nombre simple terminado en .tif, sin rutas)"
        )
    stem = Path(qgz_path).stem
    xml = _qgs_xml(cog_filename, layer_name)
    with zipfile.ZipFile(qgz_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}.qgs", xml)
    print(f"  -> {qgz_path}")
