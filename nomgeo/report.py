"""Markdown tables for the documentation, built from the run outputs.

python -m nomgeo report configs/*.yaml   ->   docs/_generated/*.md
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from .cli import _haversine, _read_legacy
from .config import PROJECT_DIR, load_dataset
from .pipeline import reason_group

TYPES = ["house", "interpolated", "block", "between", "intersection", "near_intersection", "street", "poi"]
# Run times of the two-pass scripts, as published on the project website (2025).
LEGACY_TIME = {"berkeley": "2m04s", "cambridge": "5m02s", "louisville": "1h26m", "norfolk": "10m27s",
               "san_jose": "5h41m", "saint_paul": "37m30s", "virginia_beach": "1h24m"}
LABEL = {"berkeley": "Berkeley", "cambridge": "Cambridge", "lincoln": "Lincoln", "louisville": "Louisville",
         "norfolk": "Norfolk", "san_jose": "San Jose", "saint_paul": "Saint Paul",
         "virginia_beach": "Virginia Beach"}


def _table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def build(configs: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_dataset(c) for c in configs]
    res_rows, type_rows, reason_rows, legacy_rows, gap_rows, prec_rows = [], [], [], [], [], []
    tot = Counter()
    reasons_all = Counter()
    for ds in datasets:
        d = Path(ds.output_dir) / ds.name
        summary = json.loads((d / "summary.json").read_text())
        rows = list(csv.DictReader(open(d / "results.csv", newline="", encoding="utf-8")))
        n = len(rows)
        matched = [r for r in rows if r["status"] == "matched"]
        old_m = _read_legacy(PROJECT_DIR / ds.legacy.get("matches", ""), True)
        name = LABEL.get(ds.name, ds.name)
        legacy_ok = ds.name != "saint_paul"          # its legacy files are from an interrupted run
        old_rate = _pct(len(old_m), n) if legacy_ok and old_m else "-"
        res_rows.append([name, f"{n:,}", f"{len(matched):,}", _pct(len(matched), n), old_rate,
                         LEGACY_TIME.get(ds.name, "-"), f"{summary['seconds_total']:.0f} s"])
        tot["n"] += n
        tot["m"] += len(matched)

        by = Counter(r["match_type"] for r in matched)
        type_rows.append([name] + [f"{by.get(t, 0):,}" for t in TYPES])
        prec = {t: sorted(float(r["precision_m"]) for r in matched if r["match_type"] == t) for t in TYPES}
        prec_rows.append([name] + [f"{v[len(v) // 2]:.0f}" if v else "-" for v in prec.values()])

        reasons = Counter(reason_group(r.get("reason", "")) for r in rows if r["status"] != "matched")
        reasons_all.update(reasons)
        reason_rows.append((name, reasons))

        if legacy_ok and old_m:
            same = moved = lost = gained = 0
            fi_far = fi_n = 0
            for r in rows:
                o = old_m.get(r["address"])
                if o is None:
                    if r["status"] == "matched" and r["address"] not in old_m:
                        gained += 1
                    continue
                if r["status"] != "matched":
                    lost += 1
                    continue
                dist = _haversine(o[0], o[1], float(r["lat"]), float(r["lon"]))
                if o[2] == "false intersection":
                    fi_n += 1
                    fi_far += dist > 250
                if dist <= 100:
                    same += 1
                elif dist > 250:
                    moved += 1
            legacy_rows.append([name, f"{len(old_m):,}", _pct(same, len(old_m)), _pct(moved, len(old_m)),
                                _pct(lost, len(old_m)), f"{gained:,}"])

        gaps = [int(m.group(1)) for r in rows if r["status"] != "matched"
                for m in [re.search(r"streets do not meet \(closest (\d+) m\)", r.get("reason", ""))] if m]
        if gaps:
            gap_rows.append([name, f"{len(gaps):,}"] + [f"{sum(1 for g in gaps if g <= t):,}" for t in (60, 80, 100, 150, 200)])

    (out_dir / "results.md").write_text(_table(
        ["City", "Unique records", "Matched", "Match rate", "Legacy match rate", "Legacy run time", "New run time"],
        res_rows + [["**All**", f"**{tot['n']:,}**", f"**{tot['m']:,}**", f"**{_pct(tot['m'], tot['n'])}**", "", "", ""]]))
    (out_dir / "match_types.md").write_text(_table(["City"] + TYPES, type_rows))
    (out_dir / "precision.md").write_text(_table(["City"] + TYPES, prec_rows))
    groups = [g for g, _ in reasons_all.most_common()]
    (out_dir / "unmatched.md").write_text(_table(
        ["Reason"] + [n for n, _ in reason_rows] + ["All"],
        [[g] + [f"{r.get(g, 0):,}" for _, r in reason_rows] + [f"{reasons_all[g]:,}"] for g in groups],
        ["---"] + ["---:"] * (len(reason_rows) + 1)))
    (out_dir / "legacy.md").write_text(_table(
        ["City", "Legacy matches", "Same place (<= 100 m)", "Moved > 250 m", "Now unmatched", "Newly matched"],
        legacy_rows))
    (out_dir / "gap_tradeoff.md").write_text(_table(
        ["City", "Streets do not meet", "<= 60 m", "<= 80 m", "<= 100 m", "<= 150 m", "<= 200 m"], gap_rows))
    print(f"wrote {len(list(out_dir.glob('*.md')))} tables to {out_dir}")
