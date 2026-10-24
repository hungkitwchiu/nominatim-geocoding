"""Time the three situations that matter in practice:  python -m nomgeo benchmark configs/*.yaml

1. cold: no caches at all (first use, or new data);
2. after a rule change: street indexes and Nominatim answers cached, every record re-matched;
3. unchanged: every result comes from the results cache.

It empties the caches first. The times go to outputs/benchmark.json, which `report` turns
into the timing table of the documentation.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

from .cache import KVCache
from .config import load_dataset
from .pipeline import code_fingerprint, run


def _empty_caches(folder: Path) -> None:
    for name in ("results.sqlite", "nominatim.sqlite"):
        for suffix in ("", "-wal", "-shm"):
            (folder / (name + suffix)).unlink(missing_ok=True)
    for p in (folder / "areas").glob("*.pkl"):
        p.unlink()


def benchmark(configs: list[str], jobs: int) -> dict:
    datasets = [load_dataset(c) for c in configs]
    folders = sorted({Path(ds.cache_dir) for ds in datasets})
    for folder in folders:
        _empty_caches(folder)
    phases, records, dv = {}, 0, None
    for phase in ("cold", "rule_change", "unchanged"):
        if phase == "rule_change":
            for folder in folders:
                KVCache(folder / "results.sqlite").clear()
        t0 = time.time()
        per = {}
        for ds in datasets:
            s = run(ds, jobs=jobs, progress=False)
            per[ds.name] = s["seconds_total"]
            if phase == "cold":
                records += s["records"]
                dv = s["data_version"]
        phases[phase] = {"seconds": round(time.time() - t0, 1), "datasets": per}
        print(f"{phase}: {phases[phase]['seconds']} s  (" + ", ".join(f"{k} {v} s" for k, v in per.items()) + ")")
    out = {"date": date.today().isoformat(), "jobs": jobs, "cpus": os.cpu_count(),
           "code": code_fingerprint(), "data_version": dv, "records": records, "phases": phases}
    path = Path(datasets[0].output_dir) / "benchmark.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return out
