import bench_meta

def test_machine_metadata_has_required_keys():
    m = bench_meta.machine_metadata()
    for k in ("hostname", "cpu_model", "cpu_cores", "ram_total_mb",
              "disk_type", "os", "kernel", "python", "tiledb", "timestamp"):
        assert k in m, f"falta la clave {k}"
    assert isinstance(m["cpu_cores"], int) and m["cpu_cores"] >= 1
    assert isinstance(m["ram_total_mb"], int) and m["ram_total_mb"] > 0
    assert m["disk_type"] in ("SSD", "HDD", "desconocido")
