import bench_meta

def test_machine_metadata_has_required_keys():
    m = bench_meta.machine_metadata()
    for k in ("hostname", "cpu_model", "cpu_cores", "ram_total_mb",
              "disk_type", "os", "kernel", "python", "tiledb", "timestamp"):
        assert k in m, f"falta la clave {k}"
    assert isinstance(m["cpu_cores"], int) and m["cpu_cores"] >= 1
    assert isinstance(m["ram_total_mb"], int) and m["ram_total_mb"] > 0
    assert m["disk_type"] in ("SSD", "HDD", "desconocido")

import json, bench_runner

def test_consolidate_merges_runs(tmp_path):
    vg = tmp_path / "vg.json"; q = tmp_path / "q.json"
    vg.write_text(json.dumps({"tipo": "vs_geotiff", "dataset": "datos1",
                              "casos": [], "maquina": {"hostname": "h1"}}))
    q.write_text(json.dumps({"tipo": "queries", "dataset": "datos1",
                             "casos": [], "maquina": {"hostname": "h1"}}))
    out = bench_runner.consolidate([str(vg), str(q)])
    assert out["maquina"]["hostname"] == "h1"
    assert {r["tipo"] for r in out["runs"]} == {"vs_geotiff", "queries"}
