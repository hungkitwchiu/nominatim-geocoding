"""Batch pipeline: read records, geocode each against the area index, write results."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import multiprocessing as mp
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import shapely
from tqdm import tqdm

from . import __version__
from .area import load_area
from .cache import KVCache
from .config import Dataset
from .db import connect, data_version
from .locate import Location, locate_intersection, locate_number, locate_street
from .nominatim import NominatimClient, find_poi
from .normalize import split_street
from .parse import (ADDRESS, BETWEEN, COORDINATES, INTERSECTION, STREET, Record, parse_record)
from .resolve import resolve, resolve_typo

LOG = logging.getLogger(__name__)

OUTPUT_FIELDS = ["address", "status", "match_type", "tier", "lat", "lon", "precision_m", "matched",
                 "source", "flags", "reason", "kind", "in_city", "penalty", "detail"]
# Confidence tiers.  A: the street(s) as written or after routine clean-up, and direct evidence.
# B: a correction or inference that the evidence confirms, or a location near rather than at
# the evidence.  C: nothing confirms the reading.
TIER_B_FLAGS = {"typo_1", "type_changed", "redacted", "truncated", "trailing_text", "second_chance",
                "alt_name", "route_number", "dominant_evidence", "block_of_exact_number",
                "between_known_numbers", "offset_of_intersection"}
TIER_C_FLAGS = {"typo_2", "bare_route_number", "number_not_found", "number_on_first_street",
                "cross_street_ignored", "extra_street_ignored"}
# Types that usually name a place rather than a street ("JEFFERSON PARK", "CENTRAL SQ"):
# never swap them for a different street type.
PLACE_TYPES = {"PARK", "PLZ", "SQ", "CTR", "MALL", "STA", "CMN", "CMNS", "GRN", "CRSE", "CLB"}


# Modules that cannot change a result; editing them keeps the cached results.
_NOT_FINGERPRINTED = {"__main__.py", "benchmark.py", "cli.py", "report.py"}


def code_fingerprint() -> str:
    h = hashlib.sha1(__version__.encode())
    for f in sorted(Path(__file__).parent.glob("*.py")):
        if f.name not in _NOT_FINGERPRINTED:
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def detect_numbering(records: list[Record]) -> str:
    """'hundred_block' when most house numbers are block-rounded (x00 or x50), else 'exact'."""
    nums = [r.number for r in records if r.kind == ADDRESS and r.number]
    if not nums:
        return "exact"
    share = sum(1 for n in nums if n % 100 in (0, 50)) / len(nums)
    return "hundred_block" if share >= 0.8 else "exact"


def _auto_numbering(addresses: list[str], ds: Dataset, cache: KVCache, group: str, use_cache: bool) -> str:
    """detect_numbering over the whole input. Parsing every record takes seconds, so the
    answer is cached with the results; it depends only on the input and the group."""
    key = "numbering:" + hashlib.sha1("\x1e".join(addresses).encode() + group.encode()).hexdigest()
    mode = cache.get(key) if use_cache else None
    if mode is None:
        mode = detect_numbering([parse_record(a, ds) for a in addresses])
        cache.put(key, mode, group)
    return mode


# --------------------------------------------------------------------------- one record
def tier(match_type: str, flags) -> str:
    """Confidence tier A, B or C of a match (see TIER_B_FLAGS)."""
    flags = set(flags)
    if match_type in ("street", "poi", "coordinates") or flags & TIER_C_FLAGS:
        return "C"
    if match_type in ("near_intersection", "between") or flags & TIER_B_FLAGS:
        return "B"
    return "A"


def _row(raw: str, rec: Record, loc: Location | None, reason: str | None, index, extra=()) -> dict:
    row = {"address": raw, "kind": rec.kind}
    if loc is None:
        row.update(status="unmatched", reason=reason or "no match",
                   flags=";".join(dict.fromkeys(list(rec.flags) + list(extra))))
        return row
    lat, lon = index.proj.inv(loc.x, loc.y)
    flags = list(rec.flags) + list(loc.flags) + list(extra)
    if index.core is not None and not loc.in_core:
        flags.append("outside_city")
    row.update(status="matched", match_type=loc.match_type, tier=tier(loc.match_type, flags),
               lat=round(lat, 7), lon=round(lon, 7),
               precision_m=round(loc.precision_m), matched=" & ".join(loc.names), source=loc.source,
               flags=";".join(dict.fromkeys(flags)), in_city=int(loc.in_core), penalty=loc.penalty,
               detail=loc.detail)
    return row


def _poi_allowed(text: str, ds) -> bool:
    """Only place-like text goes to the named-place search.

    'FRESH POND GOLF COURSE', 'PORTER SQ' and 'SENNOTT PARK' do; 'LEIGHTON ST' does not,
    or it would match businesses such as 'Leighton Street Car Wash'.
    """
    sn = split_street(text, ds.single_letter_suffixes)
    return not sn.suffix or bool(sn.suffix & PLACE_TYPES)


def _resolve_street(text, index, ds, truncated=False, wildcard=False):
    cands = resolve(text, index, ds, truncated=truncated, wildcard_first=wildcard)
    sn = split_street(text, ds.single_letter_suffixes)
    if sn.suffix & PLACE_TYPES:
        cands = [c for c in cands if "type_changed" not in c.flags]
    return cands


def _between(rec: Record, index, ds):
    """Lincoln "26TH & K-N ST": midpoint of the corners 26th & K and 26th & N."""
    main, a, b = rec.streets
    cands = [_resolve_street(t, index, ds) for t in (main, a, b)]
    missing = [t for t, c in zip((main, a, b), cands) if not c]
    if missing:
        return None, "street not found: " + " & ".join(missing), ()
    p1, r1 = locate_intersection(cands[0], cands[1], index, ds)
    p2, r2 = locate_intersection(cands[0], cands[2], index, ds)
    if not (p1 and p2):
        return None, f"corner not found ({r1 or r2})", ()
    d = math.hypot(p1.x - p2.x, p1.y - p2.y)
    x, y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
    names = tuple(dict.fromkeys(p1.names + p2.names))
    loc = Location(x, y, "between", max(p1.precision_m, p2.precision_m, d / 2), "osm_geometry",
                   names, p1.flags + p2.flags, p1.penalty + p2.penalty, index.in_core(x, y),
                   f"between corners {d:.0f} m apart")
    return loc, None, ()


def _attempt(rec: Record, index, ds, mode, client):
    if rec.kind == BETWEEN:
        return _between(rec, index, ds)
    if rec.kind == INTERSECTION:
        loc, reason, extra = _intersection(rec, index, ds, mode, client)
    else:
        loc, reason, extra = _single(rec, rec.streets[0], rec.number, index, ds, mode, client)
    if loc is None and rec.single_alternative:
        # "ROUGH AND READY RD" read as one street rather than an intersection.
        alt = parse_record(rec.single_alternative.replace(" AND ", " ANDX "), ds)
        if alt.kind in (ADDRESS, STREET) and alt.streets:
            text = alt.streets[0].replace("ANDX", "AND")
            loc2, reason2, extra2 = _single(rec, text, alt.number, index, ds, mode, client)
            if loc2:
                loc, reason, extra = loc2, None, extra2 + ("and_in_name",)
            elif reason and reason.startswith("street not found") and reason2 \
                    and not reason2.startswith("street not found"):
                reason = reason2           # the one-street reading got further
    return loc, reason, extra


def geocode_record(raw: str, index, ds: Dataset, mode: str, client) -> dict:
    rec = parse_record(raw, ds)
    if rec.kind == COORDINATES:
        if not ds.accept.get("coordinates_in_text", False):
            return _row(raw, rec, None, "coordinates in text (not accepted by policy)", index)
        lat, lon = rec.coords
        left, top, right, bottom = index.bbox
        if not (min(bottom, top) <= lat <= max(bottom, top) and min(left, right) <= lon <= max(left, right)):
            return _row(raw, rec, None, "coordinates in text lie outside the search area", index)
        x, y = index.proj.point(lon, lat)
        loc = Location(x, y, "coordinates", 25.0, "record", (), ("coordinates_in_text",), 0,
                       index.in_core(x, y), f"{lat}, {lon}")
        return _row(raw, rec, loc, None, index)
    if rec.kind not in (ADDRESS, STREET, INTERSECTION, BETWEEN):
        return _row(raw, rec, None, rec.note, index)

    loc, reason, extra = _attempt(rec, index, ds, mode, client)
    if loc is None and rec.alternative:
        # Berkeley's redaction reads two ways: "00 LOMA" is the 0-99 block of Loma or La Loma,
        # "2100 CENTER" is 21st Center or house 2100 on Center Street.
        alt = parse_record(raw, ds, reading=rec.alternative)
        if alt.kind in (ADDRESS, STREET, INTERSECTION):
            loc2, reason2, extra2 = _attempt(alt, index, ds, mode, client)
            if loc2:
                rec, loc, reason, extra = alt, loc2, None, extra2
            elif reason and reason.startswith("street not found") and reason2 \
                    and not reason2.startswith("street not found"):
                reason = reason2           # the other reading got further
    if loc is not None and "offset_of_intersection" in rec.flags:
        # "on A, south of B": somewhere along A near the corner, not at it.
        loc = replace(loc, precision_m=max(loc.precision_m, 100.0))
    if loc is not None:
        limit = float(ds.precision.get("max_precision_m", 250))
        if loc.precision_m > limit:
            return _row(raw, rec, None, f"too imprecise: {loc.match_type} on {' & '.join(loc.names)} "
                                        f"({loc.precision_m:.0f} m > {limit:.0f} m)", index)
        if not loc.in_core and index.core is not None and not ds.accept.get("outside_core", True):
            return _row(raw, rec, None, "outside the city boundary (not accepted by policy)", index)
    return _row(raw, rec, loc, reason, index, extra)


def _single(rec, text, number, index, ds, mode, client):
    cands = _resolve_street(text, index, ds, rec.truncated, rec.wildcard_first)
    if cands:
        if number is not None:
            loc, reason = locate_number(cands, number, mode, index, ds)
        else:
            loc, reason = locate_street(cands, index, ds)
        if loc:
            return loc, None, ()
        # The street exists, so never swap it for a place that shares its name
        # ("500 BROADWAY" is not the Broadway subway stop, "400 GILMAN" not "924 Gilman").
        # A place-like name that fits a street only with a word added or its type changed is a
        # guess, though: "450 RYLAND PARK" is not on Ryland Park Drive, so the park is tried.
        if not (_poi_allowed(text, ds) and all({"type_as_name", "type_changed"} & set(c.flags) for c in cands)):
            return None, reason, ()
    else:
        reason = f"street not found: {text}"
        if not _poi_allowed(text, ds):
            return None, reason, ()
    poi, poi_reason = find_poi(text, index, ds, client)
    if poi:
        return poi, None, (("number_ignored",) if number is not None else ())
    return None, poi_reason or reason, ()


def _intersection(rec, index, ds, mode, client):
    sides = []
    for i, part in enumerate(rec.streets):
        last = i == len(rec.streets) - 1
        sides.append((part, _resolve_street(part, index, ds, rec.truncated and last,
                                            rec.wildcard_first and i == 0)))
    found = [(p, c) for p, c in sides if c]
    missing = [p for p, c in sides if not c]
    loc = reason = None
    extra: tuple = ()
    if len(found) >= 2:
        for a in range(len(found)):
            for b in range(a + 1, len(found)):
                loc, reason = locate_intersection(found[a][1], found[b][1], index, ds)
                if loc or (reason and "ambiguous" in reason):
                    break
            if loc or (reason and "ambiguous" in reason):
                break
        if loc is None and reason and reason.startswith("streets do not meet") and len(found) == 2:
            # Second chance: a one-letter variant of either name that does cross the other
            # street (Cambridge "HAWTHORNE ST" = Hawthorn Street; the exact name is elsewhere).
            (pa, ca), (pb, cb) = found
            for ta, tb in ((resolve_typo(pa, index, ds), cb), (ca, resolve_typo(pb, index, ds))):
                if not ta or not tb:
                    continue
                loc2, _ = locate_intersection(ta, tb, index, ds)
                if loc2 and loc2.match_type == "intersection":
                    loc, reason = loc2, None
                    extra = ("second_chance",)
                    break
        if len(sides) > 2 and loc:
            extra = extra + ("extra_street_ignored",)
    elif len(found) == 1 and len(rec.streets) >= 2:
        reason = f"street not found: {missing[0]}"
    if loc is None and rec.number is not None and sides and sides[0][1]:
        loc2, _ = locate_number(sides[0][1], rec.number, mode, index, ds)
        if loc2 and loc2.match_type != "street":      # needs real number evidence
            return loc2, None, ("number_on_first_street",)
    if loc is None and not found:
        reason = "street not found: " + " & ".join(missing)
    if loc is None and missing:
        # "DANEHY PARK & SHERMAN ST": one side is a place, not a street.  The place must
        # touch the other street, or a same-named place elsewhere could be picked.
        others = [c for _, cs in found for c in cs]
        max_m = float(ds.precision.get("poi_max_radius_m", 150))
        for part in missing:
            if not _poi_allowed(part, ds):
                continue
            poi, poi_reason = find_poi(part, index, ds, client)
            if poi and others:
                p = shapely.Point(poi.x, poi.y)
                near = min(index.seg_geom[s].distance(p) for c in others for s in c.entry.segs)
                if near > max_m:
                    poi, poi_reason = None, f"named place '{part}' is {near:.0f} m from the cross street"
            if poi:
                return poi, None, ("poi_side", "cross_street_ignored")
            if poi_reason and reason and reason.startswith("street not found"):
                reason = poi_reason
    return loc, reason, extra


# ---------------------------------------------------------------------------- batch run
_G: dict = {}


def _work(chunk):
    index, ds, mode = _G["index"], _G["ds"], _G["mode"]
    if "client" not in _G:
        _G["client"] = NominatimClient(ds.nominatim_url, KVCache(Path(ds.cache_dir) / "nominatim.sqlite"),
                                       _G["dv"])
    out = []
    for raw in chunk:
        try:
            out.append(geocode_record(raw, index, ds, mode, _G["client"]))
        except Exception as exc:  # keep the batch going; report the record
            LOG.exception("failed on %r", raw)
            out.append({"address": raw, "status": "unmatched", "reason": f"internal error: {exc}"})
    return out


def read_addresses(ds: Dataset) -> list[str]:
    with open(ds.input, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        col = ds.address_column
        if col not in (reader.fieldnames or []):
            raise ValueError(f"{ds.input}: column '{col}' not found (have {reader.fieldnames})")
        seen, out = set(), []
        for row in reader:
            a = row[col] if row[col] is not None else ""
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out


def run(ds: Dataset, jobs: int = 4, use_cache: bool = True, limit: int | None = None,
        progress: bool = True) -> dict:
    t0 = time.time()
    addresses = read_addresses(ds)
    if limit:
        addresses = addresses[:limit]
    conn = connect(ds.database)
    dv = data_version(conn)
    index, area_info = load_area(conn, ds, dv, use_cache=use_cache)
    conn.close()
    t_index = time.time() - t0

    # Cached results are grouped by dataset and everything a result depends on apart from
    # the record itself. The area key covers the box, the city boundary, the data and the
    # index code, so a run whose boundary lookup came out differently never shares cached
    # results with this one.
    group = f"{ds.name}:" + hashlib.sha1(
        f"{ds.config_hash}|{code_fingerprint()}|{dv}|{area_info['key']}".encode()).hexdigest()[:16]
    rcache = KVCache(Path(ds.cache_dir) / "results.sqlite")
    mode = ds.numbering
    if mode == "auto":
        mode = _auto_numbering(addresses, ds, rcache, group, use_cache)
    fp = f"{group}|{mode}"
    keys = {a: hashlib.sha1(f"{a}\x1f{fp}".encode()).hexdigest() for a in addresses}
    cached = rcache.get_many(list(keys.values())) if use_cache else {}
    results = {a: cached[k] for a, k in keys.items() if k in cached}
    todo = [a for a in addresses if a not in results]

    t1 = time.time()
    _G.update(index=index, ds=ds, mode=mode, dv=dv)
    _G.pop("client", None)
    chunks = [todo[i:i + 250] for i in range(0, len(todo), 250)]
    bar = tqdm(total=len(todo), desc=f"{ds.name}", unit="rec", disable=not progress or not todo)
    new_rows = []
    if jobs > 1 and len(todo) > 1000:
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
            for rows in ex.map(_work, chunks):
                new_rows += rows
                bar.update(len(rows))
    else:
        for chunk in chunks:
            rows = _work(chunk)
            new_rows += rows
            bar.update(len(rows))
    bar.close()
    for r in new_rows:
        results[r["address"]] = r
    # Errors (a crash, a Nominatim outage) are retried next time rather than cached.
    rcache.put_many(((keys[r["address"]], r) for r in new_rows
                     if not re.search(r"internal error|nominatim error", r.get("reason") or "")), group)
    # Keep this dataset's three most recently used groups: going back to the previous rules
    # stays instant, and the cache does not grow with every change.
    rcache.keep_groups(f"{ds.name}:", group)
    t_geo = time.time() - t1

    rows = [results[a] for a in addresses]
    summary = write_outputs(ds, rows, {
        "dataset": ds.name, "records": len(addresses), "numbering": mode, "data_version": dv,
        "config_hash": ds.config_hash, "code": code_fingerprint(), "from_cache": len(addresses) - len(todo),
        "seconds_index": round(t_index, 1), "seconds_geocode": round(t_geo, 1),
        "seconds_total": round(time.time() - t0, 1), "area": area_info})
    return summary


# ------------------------------------------------------------------------------ outputs
def reason_group(reason: str) -> str:
    """Collapse specific reasons ('street not found: OAK ST') into report categories."""
    r = reason or ""
    for pat, label in [(r"same-named street outside the city", "number only on a same-named street outside the city"),
                       (r"^corner not found", "'between' record: a corner not found"),
                       (r"^named place .* from the cross street", "named place too far from the cross street"),
                       (r"^street not found", "street not found in area"),
                       (r"^house number not found", "house number not found on street"),
                       (r"^house number \d+ is outside", "number outside the street's known numbers"),
                       (r"^streets do not meet", "streets do not meet"),
                       (r"^ambiguous", "ambiguous (several places fit)"),
                       (r"^street too long", "street-only record on a long street"),
                       (r"interchange", "highway interchange (policy)"),
                       (r"motorway", "on a motorway (policy)"),
                       (r"ramp or direction|direction of travel", "ramp / direction of travel"),
                       (r"interstate", "interstate mile marker / interstate only"),
                       (r"named place too large", "named place too large"),
                       (r"^too imprecise", "evidence too imprecise"),
                       (r"placeholder|no street name", "placeholder / empty"),
                       (r"another (place|state)", "another city or state"),
                       (r"coordinates", "coordinates in text (policy)")]:
        if re.search(pat, r):
            return label
    return r or "other"


def write_outputs(ds: Dataset, rows: list[dict], meta: dict) -> dict:
    out_dir = Path(ds.output_dir) / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    unmatched = sorted((r for r in rows if r.get("status") != "matched"),
                       key=lambda r: (reason_group(r.get("reason", "")), r.get("reason", ""), r["address"]))
    with open(out_dir / "unmatched.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["address", "kind", "reason", "flags"], extrasaction="ignore")
        w.writeheader()
        w.writerows(unmatched)

    matched = [r for r in rows if r.get("status") == "matched"]
    by_type = Counter(r["match_type"] for r in matched)
    prec = {}
    for t in by_type:
        vals = sorted(float(r["precision_m"]) for r in matched if r["match_type"] == t)
        prec[t] = vals[len(vals) // 2]
    flags = Counter(f for r in matched for f in (r.get("flags") or "").split(";") if f)
    summary = dict(meta, matched=len(matched), unmatched=len(rows) - len(matched),
                   match_rate=round(100 * len(matched) / max(len(rows), 1), 2),
                   match_types=dict(by_type.most_common()),
                   tiers=dict(sorted(Counter(r.get("tier", "") for r in matched).items())),
                   median_precision_m=prec,
                   unmatched_reasons=dict(Counter(reason_group(r.get("reason", "")) for r in unmatched).most_common()),
                   flags=dict(flags.most_common()))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
