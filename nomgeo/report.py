"""Markdown tables for the documentation, built from the run outputs.

python -m nomgeo report configs/*.yaml   ->   docs/_generated/*.md
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from .config import load_dataset
from .pipeline import reason_group

TYPES = ["house", "interpolated", "block", "between", "intersection", "near_intersection", "street", "poi"]
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


def _duration(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f} s"
    s = round(seconds)
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h} h {m} min"
    return (f"{m} min {s} s" if s else f"{m} min") if m else f"{s} s"


def _timing_table(path: Path, names: list[str]) -> str:
    """Per-city run times from `nomgeo benchmark`."""
    if not path.exists():
        return "*Run `python -m nomgeo benchmark configs/*.yaml` to fill this table.*\n"
    b = json.loads(path.read_text())
    ph = b["phases"]
    rows = [[LABEL.get(n, n)] + [_duration(ph[p]["datasets"][n]) for p in ("cold", "rule_change", "unchanged")]
            for n in names if n in ph["cold"]["datasets"]]
    rows.append([f"**All ({b['records']:,} records)**"]
                + [f"**{_duration(ph[p]['seconds'])}**" for p in ("cold", "rule_change", "unchanged")])
    return _table(["City", "From scratch", "After a rule change", "Nothing changed"], rows)


def build(configs: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_dataset(c) for c in configs]
    res_rows, type_rows, reason_rows, gap_rows, prec_rows, near_rows = [], [], [], [], [], []
    tot = Counter()
    reasons_all = Counter()
    for ds in datasets:
        d = Path(ds.output_dir) / ds.name
        rows = list(csv.DictReader(open(d / "results.csv", newline="", encoding="utf-8")))
        n = len(rows)
        matched = [r for r in rows if r["status"] == "matched"]
        name = LABEL.get(ds.name, ds.name)
        res_rows.append([name, f"{n:,}", f"{len(matched):,}", _pct(len(matched), n)])
        tot["n"] += n
        tot["m"] += len(matched)

        by = Counter(r["match_type"] for r in matched)
        type_rows.append([name] + [f"{by.get(t, 0):,}" for t in TYPES])
        prec = {t: sorted(float(r["precision_m"]) for r in matched if r["match_type"] == t) for t in TYPES}
        prec_rows.append([name] + [f"{v[len(v) // 2]:.0f}" if v else "-" for v in prec.values()])

        reasons = Counter(reason_group(r.get("reason", "")) for r in rows if r["status"] != "matched")
        reasons_all.update(reasons)
        reason_rows.append((name, reasons))

        gaps = [int(m.group(1)) for r in rows if r["status"] != "matched"
                for m in [re.search(r"streets do not meet \(closest (\d+) m\)", r.get("reason", ""))] if m]
        if gaps:
            bands = [(0, 100), (100, 150), (150, 200), (200, 10 ** 9)]
            gap_rows.append([name, f"{len(gaps):,}"] + [f"{sum(1 for g in gaps if lo < g <= hi or (lo == 0 and g == 0)):,}"
                                                        for lo, hi in bands])
        near = [r for r in matched if r["match_type"] == "near_intersection"]
        near_gaps = [int(m.group(1)) for r in near for m in [re.search(r"gap (\d+) m", r.get("detail", ""))] if m]
        if near:
            near_prec = sorted(float(r["precision_m"]) for r in near)
            near_rows.append([name, f"{len(near):,}", f"{sum(g <= 40 for g in near_gaps):,}",
                              f"{sum(40 < g <= 70 for g in near_gaps):,}", f"{sum(70 < g for g in near_gaps):,}",
                              f"{sum('side_by_side' in (r.get('flags') or '') for r in near):,}",
                              f"{near_prec[len(near_prec) // 2]:.0f}"])

    (out_dir / "results.md").write_text(_table(
        ["City", "Unique records", "Matched", "Match rate"],
        res_rows + [["**All**", f"**{tot['n']:,}**", f"**{tot['m']:,}**", f"**{_pct(tot['m'], tot['n'])}**"]]))
    (out_dir / "timing.md").write_text(_timing_table(Path(datasets[0].output_dir) / "benchmark.json",
                                                     [ds.name for ds in datasets]))
    (out_dir / "match_types.md").write_text(_table(["City"] + TYPES, type_rows))
    (out_dir / "precision.md").write_text(_table(["City"] + TYPES, prec_rows))
    groups = [g for g, _ in reasons_all.most_common()]
    (out_dir / "unmatched.md").write_text(_table(
        ["Reason"] + [n for n, _ in reason_rows] + ["All"],
        [[g] + [f"{r.get(g, 0):,}" for _, r in reason_rows] + [f"{reasons_all[g]:,}"] for g in groups],
        ["---"] + ["---:"] * (len(reason_rows) + 1)))
    (out_dir / "gap_tradeoff.md").write_text(_table(
        ["City", "Streets do not meet", "within 100 m (side by side)", "100-150 m", "150-200 m", "over 200 m"],
        gap_rows))
    (out_dir / "near.md").write_text(_table(
        ["City", "Near intersections", "gap <= 40 m", "40-70 m", "70-100 m", "beside a cul-de-sac", "median precision m"],
        near_rows))
    print(f"wrote {len(list(out_dir.glob('*.md')))} tables to {out_dir}")
