import numpy as np
import rasterio
from rasterio.transform import from_origin
import export_to_geotiff as ex

def test_write_cog_is_tiled_with_overviews(tmp_path):
    grid = np.random.rand(1024, 1024).astype("float32")
    tr = from_origin(0, 1024, 1, 1)
    out = tmp_path / "cog.tif"
    ex.write_geotiff(out, grid, tr, "EPSG:32614", cog=True)
    with rasterio.open(out) as ds:
        assert ds.profile["tiled"] is True
        assert ds.overviews(1), "el COG debe tener overviews"

import xml.etree.ElementTree as ET  # contenido autogenerado por gis_style, no entrada externa
import gis_style

def test_qml_has_depth_stops():
    qml = gis_style.depth_qml()
    root = ET.fromstring(qml)  # debe parsear
    values = [item.get("value") for item in root.iter("item")]
    for thr in ("0.01", "0.25", "0.5", "1", "2"):
        assert thr in values, f"falta la parada {thr}"

import zipfile, gis_package

def test_qgz_uses_relative_datasource(tmp_path):
    cog = tmp_path / "H_10_00.tif"; cog.write_bytes(b"\x00")
    qgz = tmp_path / "proyecto.qgz"
    gis_package.build_qgz(str(qgz), cog_filename="H_10_00.tif",
                          layer_name="Calado 10:00")
    with zipfile.ZipFile(qgz) as z:
        qgs_name = [n for n in z.namelist() if n.endswith(".qgs")][0]
        xml = z.read(qgs_name).decode()
    assert 'H_10_00.tif' in xml
    assert '/tmp' not in xml and str(tmp_path) not in xml  # nada de rutas absolutas
    assert 'relative' in xml.lower()
