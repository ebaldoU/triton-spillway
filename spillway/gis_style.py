"""Genera un estilo QML de calado para QGIS, coherente con la app
(umbrales de Russo: 0,01 / 0,25 / 0,50 / 1,00 / 2,00 m)."""
from __future__ import annotations

# (valor, color hex, etiqueta)
STOPS = [
    ("0.01", "#eaf4fb", "Húmedo (>=0,01 m)"),
    ("0.25", "#9ecae1", "Niños (0,25 m)"),
    ("0.5",  "#4292c6", "Adultos (0,50 m)"),
    ("1",    "#08519c", "Crítico (1,00 m)"),
    ("2",    "#08306b", "Extremo (>=2,00 m)"),
]


def depth_qml() -> str:
    items = "\n".join(
        f'        <item value="{v}" color="{c}" alpha="255" label="{lbl}"/>'
        for v, c, lbl in STOPS
    )
    return (
        '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">\n'
        '<qgis styleCategories="Symbology">\n'
        '  <pipe>\n'
        '    <rasterrenderer type="singlebandpseudocolor" band="1" '
        'classificationMin="0.01" classificationMax="2">\n'
        '      <rastershader>\n'
        '        <colorrampshader colorRampType="INTERPOLATED" clip="0">\n'
        f'{items}\n'
        '        </colorrampshader>\n'
        '      </rastershader>\n'
        '    </rasterrenderer>\n'
        '  </pipe>\n'
        '</qgis>\n'
    )


def write_qml(path: str) -> None:
    with open(path, "w") as f:
        f.write(depth_qml())
    print(f"  -> {path}")
