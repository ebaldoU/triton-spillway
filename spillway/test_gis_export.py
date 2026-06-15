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

import defusedxml.ElementTree as ET
import gis_style

def test_qml_has_depth_stops():
    qml = gis_style.depth_qml()
    root = ET.fromstring(qml)  # debe parsear
    values = [item.get("value") for item in root.iter("item")]
    for thr in ("0.01", "0.25", "0.5", "1", "2"):
        assert thr in values, f"falta la parada {thr}"
