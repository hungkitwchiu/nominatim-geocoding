"""Command line: python -m nomgeo {run, explain, compare} ..."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

from .area import load_area
from .config import PROJECT_DIR, load_dataset
from .db import connect, data_version
from .parse import parse_record
from .pipeline import detect_numbering, geocode_record, read_addresses, reason_group, run
from .resolve import resolve


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _datasets(paths):
    """Dataset files only, so `configs/*.yaml` can be passed (defaults.yaml is skipped)."""
    return [p for p in paths if Path(p).name != "defaults.yaml"]


def cmd_run(args) -> None:
    for cfg in _datasets(args.config):
        ds = load_dataset(cfg)
        s = run(ds, jobs=args.jobs, use_cache=not args.no_cache, limit=args.limit,
                progress=not args.quiet)
        print(f"{ds.name}: {s['matched']}/{s['records']} matched ({s['match_rate']}%), "
              f"{s['from_cache']} from cache, {s['seconds_total']} s "
              f"(index {s['seconds_index']} s, geocode {s['seconds_geocode']} s)")
        print("   match types:", s["match_types"])
        print("   unmatched:", dict(list(s["unmatched_reasons"].items())[:8]))


def cmd_explain(args) -> None:
    ds = load_dataset(args.config)
    conn = connect(ds.database)
    dv = data_version(conn)
    index, info = load_area(conn, ds, dv)
    mode = ds.numbering if ds.numbering != "auto" else detect_numbering(
        [parse_record(a, ds) for a in read_addresses(ds)[:5000]])
    for raw in args.address:
        rec = parse_record(raw, ds)
        print(f"\n{raw}\n  parsed: kind={rec.kind} number={rec.number} streets={rec.streets} "
              f"flags={rec.flags} note={rec.note} truncated={rec.truncated}")
        for i, part in enumerate(rec.streets):
            cands = resolve(part, index, ds, truncated=rec.truncated and i == len(rec.streets) - 1,
                            wildcard_first=rec.wildcard_first and i == 0)
            print(f"  candidates for {part!r}:")
            for c in cands[:12]:
                cl = index.clusters(c.entry)
                print(f"     pen={c.penalty} {c.entry.display!r:40} flags={list(c.flags)} "
                      f"pieces={len(c.entry.segs)} separate_streets={len(cl)}")
            if not cands:
                print("     (none)")
        row = geocode_record(raw, index, ds, mode, None)
        print("  result:", json.dumps({k: v for k, v in row.items() if v not in (None, "")}))


def _read_legacy(path: Path, matched: bool) -> dict:
    out = {}
    if not path or not path.exists():
        return out
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            a = row.get("original_address") or row.get("address")
            if a is None:
                continue
            if matched and row.get("lat") and row.get("lon"):
                res = row.get("result", "")
                kind = ("false intersection" if "False intersection" in res else
                        "intersection" if "Intersection" in res else
                        "buffer/centroid" if res.startswith("PostGIS") else "nominatim")
                out[a] = (float(row["lat"]), float(row["lon"]), kind)
            elif not matched:
                out[a] = None
    return out


def cmd_compare(args) -> None:
    for cfg in _datasets(args.config):
        ds = load_dataset(cfg)
        out_dir = Path(ds.output_dir) / ds.name
        new = list(csv.DictReader(open(out_dir / "results.csv", newline="", encoding="utf-8")))
        old_m = _read_legacy(PROJECT_DIR / ds.legacy.get("matches", ""), True)
        old_u = _read_legacy(PROJECT_DIR / ds.legacy.get("unmatched", ""), False)
        cats = Counter()
        dist_by_kind: dict[str, list[float]] = {}
        lost = Counter()
        gained = Counter()
        rows = []
        for r in new:
            a = r["address"]
            new_ok = r["status"] == "matched"
            if a in old_m:
                lat, lon, kind = old_m[a]
                if new_ok:
                    d = _haversine(lat, lon, float(r["lat"]), float(r["lon"]))
                    dist_by_kind.setdefault(kind, []).append(d)
                    cats["both matched"] += 1
                    if d > 250:
                        rows.append([a, kind, r["match_type"], round(d), r["matched"], r["flags"]])
                else:
                    cats["old matched, new unmatched"] += 1
                    lost[(kind, reason_group(r["reason"]))] += 1
                    rows.append([a, kind, "unmatched", "", r["reason"], r.get("flags", "")])
            elif a in old_u:
                if new_ok:
                    cats["old unmatched, new matched"] += 1
                    gained[r["match_type"]] += 1
                else:
                    cats["both unmatched"] += 1
            else:
                cats["not in old outputs"] += 1
        lines = [f"# {ds.name}: new pipeline vs legacy outputs", "", "| comparison | records |", "|---|---:|"]
        lines += [f"| {k} | {v:,} |" for k, v in cats.most_common()]
        lines += ["", "Distance between old and new points (both matched), by old match kind:", "",
                  "| old kind | n | median m | share > 100 m | share > 250 m | share > 1 km |", "|---|---:|---:|---:|---:|---:|"]
        for kind, ds_ in sorted(dist_by_kind.items()):
            ds_.sort()
            n = len(ds_)
            lines.append(f"| {kind} | {n:,} | {ds_[n // 2]:.0f} | {sum(d > 100 for d in ds_) / n:.1%} | "
                         f"{sum(d > 250 for d in ds_) / n:.1%} | {sum(d > 1000 for d in ds_) / n:.1%} |")
        lines += ["", "Old matches the new pipeline leaves uncoded (old kind, new reason):", "",
                  "| old kind | new reason | records |", "|---|---|---:|"]
        lines += [f"| {k[0]} | {k[1]} | {v:,} |" for k, v in lost.most_common(15)]
        lines += ["", "Newly matched records by match type:", "", "| match type | records |", "|---|---:|"]
        lines += [f"| {k} | {v:,} |" for k, v in gained.most_common()]
        (out_dir / "compare.md").write_text("\n".join(lines) + "\n")
        with open(out_dir / "compare_disagreements.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["address", "old_kind", "new_match_type", "distance_m", "new_matched_or_reason", "flags"])
            w.writerows(rows)
        print("\n".join(lines[:4 + len(cats)]))


def main(argv=None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="nomgeo", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="geocode one or more datasets")
    r.add_argument("config", nargs="+")
    r.add_argument("--jobs", type=int, default=4, help="worker processes (default 4)")
    r.add_argument("--no-cache", action="store_true", help="ignore and rebuild all caches")
    r.add_argument("--limit", type=int, help="only the first N records (for testing)")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)
    e = sub.add_parser("explain", help="show how single addresses are parsed and matched")
    e.add_argument("config")
    e.add_argument("address", nargs="+")
    e.set_defaults(func=cmd_explain)
    c = sub.add_parser("compare", help="compare results with the legacy two-pass outputs")
    c.add_argument("config", nargs="+")
    c.set_defaults(func=cmd_compare)
    g = sub.add_parser("report", help="write the documentation tables (docs/_generated)")
    g.add_argument("config", nargs="+")
    g.add_argument("--out", default=str(PROJECT_DIR / "docs" / "_generated"))
    g.set_defaults(func=lambda a: __import__("nomgeo.report", fromlist=["build"]).build(
        _datasets(a.config), Path(a.out)))
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
