"""Turn resolved street candidates into one location (or an explicit reason why not).

Every function returns (locations, reason).  A location is only produced from real
evidence: an OSM address point, a house-number range, a geometric crossing, or a
short street.  When candidates point to different places the record is left
uncoded as ambiguous rather than picking one, unless one place on the street holds
nearly all of the evidence for the number (a stray point elsewhere is outvoted).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import shapely
from shapely.ops import linemerge, nearest_points
from shapely.strtree import STRtree

from .resolve import Candidate
from .streets import MOTORWAY_TYPES, AreaIndex, Cluster

KIND_RANK = {"house": 0, "interpolated": 1, "block": 1, "intersection": 0, "near_intersection": 1,
             "between": 1, "street": 2}


@dataclass(frozen=True)
class Location:
    x: float
    y: float
    match_type: str
    precision_m: float
    source: str
    names: tuple[str, ...]
    flags: tuple[str, ...]
    penalty: int
    in_core: bool
    detail: str = ""
    weight: float = 1.0          # how much evidence: the number of address points behind it


def rank(loc: Location) -> int:
    """Strength of the kind of evidence; a block used for an exact number ranks below it."""
    r = KIND_RANK.get(loc.match_type, 3)
    return r + 1 if "block_of_exact_number" in loc.flags else r


def _merge(locs: list[Location], extra_flags=()) -> Location:
    """One location from several that describe the same place, weighted by evidence."""
    xs = np.array([l.x for l in locs])
    ys = np.array([l.y for l in locs])
    w = np.array([l.weight for l in locs])
    cx, cy = float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())
    spread = float(np.max(np.hypot(xs - cx, ys - cy))) if len(locs) > 1 else 0.0
    best = min(locs, key=lambda l: (l.penalty, rank(l), l.precision_m))
    names = tuple(dict.fromkeys(n for l in locs for n in l.names))
    flags = tuple(dict.fromkeys(f for l in locs for f in l.flags + tuple(extra_flags)))
    return replace(best, x=cx, y=cy, precision_m=max(best.precision_m, spread), names=names,
                   flags=flags, in_core=any(l.in_core for l in locs), weight=float(w.sum()))


def _places(group: list[Location], radius: float) -> list[list[Location]]:
    """Locations grouped into places: chains of locations within `radius` of each other."""
    out: list[list[Location]] = []
    for l in group:
        hits = [g for g in out if any(math.hypot(l.x - m.x, l.y - m.y) <= radius for m in g)]
        out = [g for g in out if all(g is not h for h in hits)] + [[l] + [m for g in hits for m in g]]
    return out


def _dominant(group: list[Location], ds, radius: float) -> Location | None:
    """The place that holds nearly all of a street's evidence for a number, or None.

    A mis-numbered address point, or a TIGER range on the wrong piece of the street, can put a
    block in two places: Berkeley's 1600 block of Cedar Street has 25 address points in one
    place and 1 point 870 m away.  On one street, the main place wins if it lies in the city,
    holds at least `dominant_ratio` times the evidence of all other places together, and none
    of those has more than `dominant_minority_max` points.
    """
    if len({l.names for l in group}) != 1:
        return None
    ranked = sorted(((sum(l.weight for l in p), p) for p in _places(group, radius)), key=lambda t: -t[0])
    if len(ranked) < 2:
        return None
    (top, main), rest = ranked[0], ranked[1:]
    others = sum(w for w, _ in rest)
    if not (any(l.in_core for l in main)
            and top >= float(ds.precision.get("dominant_ratio", 5)) * others
            and max(w for w, _ in rest) <= float(ds.precision.get("dominant_minority_max", 4))):
        return None
    loc = _merge(main, ("dominant_evidence",))
    near = min(math.hypot(loc.x - l.x, loc.y - l.y) for _, p in rest for l in p)
    return replace(loc, detail=f"{loc.detail}; {others:.0f} conflicting point(s) or range(s) "
                               f"{near:.0f} m away ignored")


def choose(locs: list[Location], ds, what: str) -> tuple[Location | None, str | None]:
    """Pick the least-relaxed evidence; refuse when it points to different places, unless
    (for a number) one place on the street holds nearly all of the evidence."""
    if not locs:
        return None, None
    radius = float(ds.precision.get("ambiguity_radius_m", 80))

    def best_of(group):
        pen = min(l.penalty for l in group)
        tier = [l for l in group if l.penalty == pen]
        kind = min(rank(l) for l in tier)
        return [l for l in tier if rank(l) == kind]

    def compact(group):
        xs = np.array([l.x for l in group]); ys = np.array([l.y for l in group])
        return float(np.max(np.hypot(xs - xs.mean(), ys - ys.mean()))) <= radius

    def places(group):
        return len({(round(l.x / radius), round(l.y / radius)) for l in group})

    tier = best_of(locs)
    if not any(l.in_core for l in tier):
        # The least-relaxed reading lies outside the city.  For a department's own records an
        # in-city reading that needs a little more relaxation wins: "400 34TH ST, Norfolk" is
        # Norfolk's West 34th Street (direction left out), not Virginia Beach's 34th Street.
        in_city = [l for l in locs if l.in_core and l.penalty <= tier[0].penalty + 3]
        if in_city:
            alt = best_of(in_city)
            if compact(alt):
                return _merge(alt, ("preferred_in_city",)), None
            main = _dominant(alt, ds, radius) if what == "street/number" else None
            if main:
                return replace(main, flags=main.flags + ("preferred_in_city",)), None
            return None, f"ambiguous {what}: {places(alt)} different places in the city"
    if compact(tier):
        return _merge(tier), None
    core = [l for l in tier if l.in_core]
    if core and len(core) < len(tier) and compact(core):
        return _merge(core, ("preferred_in_city",)), None
    main = _dominant(tier, ds, radius) if what == "street/number" else None
    if main:
        return main, None
    return None, f"ambiguous {what}: {places(tier)} different places"


# --------------------------------------------------------------------------- numbers
def _robust_center(pts):
    """Centre of address points that ignores a few mis-numbered outliers.

    Returns (x, y, spread, n_dropped); spread is the 90th-percentile distance of the
    kept points from the centre, so one stray point cannot inflate it.
    """
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if len(pts) <= 2:
        cx, cy = xs.mean(), ys.mean()
        return cx, cy, float(np.max(np.hypot(xs - cx, ys - cy))), 0
    mx, my = np.median(xs), np.median(ys)
    d = np.hypot(xs - mx, ys - my)
    keep = d <= max(150.0, 3 * float(np.median(d)))
    cx, cy = xs[keep].mean(), ys[keep].mean()
    spread = float(np.percentile(np.hypot(xs[keep] - cx, ys[keep] - cy), 90))
    return float(cx), float(cy), spread, int((~keep).sum())


def _spatial_groups(pts, link_m: float = 150.0) -> list[list]:
    """Split points (x, y, ...) into groups that are far apart.

    A long street crossing town lines often restarts its numbering, so "0 MASSACHUSETTS AVE"
    has 0-99 blocks in Boston, Cambridge and Arlington: those are different places.
    """
    groups: list[list] = []
    for p in pts:
        hits = [g for g in groups if any(math.hypot(p[0] - q[0], p[1] - q[1]) <= link_m for q in g)]
        if not hits:
            groups.append([p])
        else:
            merged = [p] + [q for g in hits for q in g]
            groups = [g for g in groups if all(g is not h for h in hits)] + [merged]
    return groups


def _block_range(number: int, mode: str) -> tuple[int, int] | None:
    if mode == "hundred_block":
        if number % 100 == 0:
            return number, number + 99
        if number % 100 == 50:                 # midpoint convention (San Jose "3550")
            return number - 50, number + 49
    return None


def _interp_range(line, start: int, end: int, n: float):
    frac = 0.5 if end == start else min(1.0, max(0.0, (n - start) / (end - start)))
    p = line.interpolate(frac, normalized=True)
    return p.x, p.y


def _cluster_evidence(index: AreaIndex, cl: Cluster, number: int, mode: str, ds) -> list[tuple]:
    """Number evidence on one stretch of street, one item per separate place.

    Each item is (x, y, match_type, precision, source, detail, extra_flags).
    """
    pids = [index.seg_place[s] for s in cl.segs]
    pts = [p for pid in pids for p in index.points.get(pid, ())]
    rngs = [r for pid in pids for r in index.ranges.get(pid, ())]
    block = _block_range(number, mode)

    def from_points(points, kind, min_prec, detail, flags):
        out = []
        for g in _spatial_groups(points):
            cx, cy, spread, dropped = _robust_center(g)
            out.append((cx, cy, kind, max(min_prec, spread), "osm_address", detail.format(n=len(g)),
                        flags + (("outlier_points_dropped",) if dropped else ()), len(g) - dropped))
        return out

    if block is None:  # an exact house number
        same = [(x, y) for n, x, y in pts if n == number]
        if same:
            return from_points(same, "house", 15.0, "{n} address point(s)", ())
        hits = [(*_interp_range(line, s, e, number), src, s, e) for s, e, step, line, src in rngs
                if s <= number <= e and (number - s) % (step or 1) == 0]
        if hits:
            return [(g[0][0], g[0][1], "interpolated", 30.0, g[0][2], f"range {g[0][3]}-{g[0][4]}", (), 1)
                    for g in _spatial_groups(hits)]
        gap = int(ds.precision.get("number_neighbor_max_gap", 20))
        same_parity = [(n, x, y) for n, x, y in pts if n % 2 == number % 2]
        lo = max((p for p in same_parity if number - gap <= p[0] < number), default=None)
        hi = min((p for p in same_parity if number < p[0] <= number + gap), default=None)
        if lo and hi and math.hypot(hi[1] - lo[1], hi[2] - lo[2]) <= 300:
            f = (number - lo[0]) / (hi[0] - lo[0])
            x, y = lo[1] + f * (hi[1] - lo[1]), lo[2] + f * (hi[2] - lo[2])
            d = math.hypot(hi[1] - lo[1], hi[2] - lo[2])
            return [(x, y, "interpolated", max(25.0, d / 2), "osm_address",
                     f"between {lo[0]} and {hi[0]}", ("between_known_numbers",), 2)]
        block = (number - number % 100, number - number % 100 + 99)
        flags = ("block_of_exact_number",)
    else:
        flags = ()

    lo_n, hi_n = block
    inblock = [(x, y) for n, x, y in pts if lo_n <= n <= hi_n]
    if inblock:
        return from_points(inblock, "block", 30.0, "{n} address point(s) in " + f"{lo_n}-{hi_n}", flags)
    hits = []
    for s, e, step, line, src in rngs:
        a, b = max(s, lo_n), min(e, hi_n)
        if a > b:
            continue
        x, y = _interp_range(line, s, e, (a + b) / 2)
        part = line.length * ((b - a) / (e - s) if e > s else 1.0)
        hits.append((x, y, part, src))
    if hits:
        out = []
        for g in _spatial_groups(hits):
            cx = sum(h[0] for h in g) / len(g)
            cy = sum(h[1] for h in g) / len(g)
            prec = max(30.0, max(h[2] for h in g) / 2 + 10)
            src = "tiger" if any(h[3] == "tiger" for h in g) else g[0][3]
            out.append((cx, cy, "block", prec, src, f"range overlap {lo_n}-{hi_n}", flags, 1))
        return out
    # Interpolate between the nearest known numbers below and above the block, when they
    # are on the same stretch (close to each other), not in different towns.
    below = max((p for p in pts if hi_n - 400 <= p[0] < lo_n), default=None)
    above = min((p for p in pts if hi_n < p[0] <= lo_n + 400), default=None)
    if below and above and math.hypot(above[1] - below[1], above[2] - below[2]) <= 600:
        target = (lo_n + hi_n) / 2
        f = (target - below[0]) / (above[0] - below[0])
        x, y = below[1] + f * (above[1] - below[1]), below[2] + f * (above[2] - below[2])
        snapped = nearest_points(cl.geom, shapely.Point(x, y))[0]
        d = math.hypot(above[1] - below[1], above[2] - below[2])
        return [(snapped.x, snapped.y, "block", max(50.0, d / 4), "osm_address",
                 f"between {below[0]} and {above[0]}", flags + ("between_known_numbers",), 1)]
    return []


def _merge_same_street(locs: list[Location], within_m: float) -> list[Location]:
    """Merge evidence for one block that is split up on the same street.

    A street interrupted by a park or freeway is several pieces, and a long block can have
    its few address points at both ends; evidence of the same street and relaxation within
    `within_m` (block scale) is one place.  Numbering that restarts in another town is
    kilometres away and stays separate.  A lone stray point on the same stretch (within
    2 x `within_m`) is dropped when at least four times as many points agree there, but an
    in-city place is never dropped in favour of one outside the city.
    """
    out, by_name = [], {}
    for l in locs:
        by_name.setdefault((l.names, l.penalty, rank(l)), []).append(l)
    for group in by_name.values():
        strays, dominant = set(), set()
        for l in group:
            if l.weight > 2:
                continue
            for h in group:
                if (h is not l and h.weight >= max(4.0, 4 * l.weight) and (h.in_core or not l.in_core)
                        and math.hypot(h.x - l.x, h.y - l.y) <= 2 * within_m):
                    strays.add(id(l))
                    dominant.add(id(h))
                    break
        group = [replace(l, flags=l.flags + ("outlier_points_dropped",)) if id(l) in dominant else l
                 for l in group if id(l) not in strays]
        merged = [[l] for l in group]
        changed = True
        while changed:
            changed = False
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if any(math.hypot(a.x - b.x, a.y - b.y) <= within_m for a in merged[i] for b in merged[j]):
                        merged[i] += merged.pop(j)
                        changed = True
                        break
                if changed:
                    break
        out += [_merge(m) if len(m) > 1 else m[0] for m in merged]
    return out


def locate_number(cands: list[Candidate], number: int, mode: str, index: AreaIndex, ds):
    locs = []
    for c in cands:
        for cl in index.clusters(c.entry):
            evidence = _cluster_evidence(index, cl, number, mode, ds)
            if not evidence or _motorway_entry(index, c.entry, cl.segs):
                continue
            for x, y, mt, prec, src, detail, fl, weight in evidence:
                in_core = index.in_core(x, y)
                if "several_types" in c.flags and not in_core:
                    continue           # a type picked among several needs evidence in the city
                locs.append(Location(x, y, mt, prec, src, (c.entry.display,), c.flags + fl,
                                     c.penalty, in_core, detail, weight))
    locs = _merge_same_street(locs, float(ds.precision.get("same_street_merge_m", 400)))
    names = "/".join(sorted({c.entry.display for c in cands})[:3])
    if locs:
        loc, reason = choose(locs, ds, "street/number")
        if loc is None or loc.in_core or index.core is None:
            return loc, reason
        # The evidence lies outside the city.  If the record's street also exists inside the
        # city, the record means that street and its block is just missing from the data:
        # "1715 COMO AV, Saint Paul" is Saint Paul's Como Avenue, not Southeast Como Avenue in
        # Minneapolis.  (Within 300 m of the in-city street, e.g. the far side of a border
        # street, the evidence is kept.)
        in_city = [c for c in cands if c.penalty <= loc.penalty
                   and any(cl.in_core for cl in index.clusters(c.entry))]
        p = shapely.Point(loc.x, loc.y)
        if not in_city or min(index.seg_geom[s].distance(p) for c in in_city
                              for cl in index.clusters(c.entry) if cl.in_core for s in cl.segs) <= 300:
            return loc, reason
        cands = in_city
        names = "/".join(sorted({c.entry.display for c in cands})[:3])
        loc2, _ = (locate_street(cands, index, ds, flag="number_not_found")
                   if _fallback_allowed(index, cands, number) else (None, None))
        if loc2 and loc2.in_core:
            return loc2, None
        return None, (f"house number not found on {names} in the city "
                      f"(a same-named street outside the city has it)")
    if all(_motorway_entry(index, c.entry) for c in cands):
        return None, "address on a motorway"
    # No evidence for this number: fall back to a short street, but only when the street the
    # record most likely means does not contradict it ("1400 MASON ST" is not on a Mason Street
    # numbered 4-18, and must not move to some other short street instead).
    if not _fallback_allowed(index, cands, number):
        return None, f"house number {number} is outside the numbers known on {names}"
    loc, reason = locate_street(cands, index, ds, flag="number_not_found")
    if loc:
        return loc, None
    return None, f"house number not found on {names}" + (f" (street-level fallback: {reason})" if reason else "")


def _fallback_allowed(index: AreaIndex, cands, number: int) -> bool:
    """A street-level fallback needs every least-relaxed candidate to fit the number."""
    best = min(c.penalty for c in cands)
    return all(_number_plausible(index, c.entry, number) for c in cands if c.penalty == best)


def _number_plausible(index: AreaIndex, entry, number: int, margin: int = 100) -> bool:
    """True when a street has no known numbers, or `number` lies near the range it has."""
    known = []
    for s in entry.segs:
        pid = index.seg_place[s]
        known += [n for n, _, _ in index.points.get(pid, ())]
        known += [b for r in index.ranges.get(pid, ()) for b in (r[0], r[1])]
    return not known or min(known) - margin <= number <= max(known) + margin


# ------------------------------------------------------------------------ street only
# Relaxations too weak to stand on their own: with no house number or crossing to confirm
# them, they could name a different street.
WEAK_FLAGS = {"type_changed", "typo_2", "bare_route_number", "route_number"}


def locate_street(cands: list[Candidate], index: AreaIndex, ds, flag: str = "street_only"):
    if not ds.accept.get("street_only", True):
        return None, "street-only records not accepted"
    if not cands:
        return None, None
    strong = [c for c in cands if not WEAK_FLAGS & set(c.flags)]
    if not strong:
        return None, "street matched only through a typo or changed type, with nothing to confirm it"
    cands = strong
    best = min(c.penalty for c in cands)
    # Close variants count too: if "84TH ST" is a stub but "S 84TH ST" also exists, the
    # record could mean either, so all of them must agree on one place.  Pieces that touch
    # (a stub continuing its neighbour) are one place.
    items = []
    for c in cands:
        # Variants matter when the record is less specific than they are (no direction or
        # type given), not when the record's own direction matched exactly.
        looser = {"direction_added", "type_missing"} & set(c.flags)
        if c.penalty > best + 2 or (c.penalty > best and not looser):
            continue
        items += [(c, cl) for cl in index.clusters(c.entry)]
    # Connected groups of touching pieces (union-find, so the order of candidates is irrelevant).
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1].geom.distance(items[j][1].geom) <= 30.0:
                parent[find(i)] = find(j)
    by_root: dict[int, list] = {}
    for i, it in enumerate(items):
        by_root.setdefault(find(i), []).append(it)
    groups = list(by_root.values())
    extra: tuple = ()
    if len(groups) > 1:
        core = [g for g in groups if any(cl.in_core for _, cl in g)]
        if len(core) != 1:
            return None, f"ambiguous street: {len(groups)} separate streets with this name"
        groups, extra = core, ("preferred_in_city",)
    group = groups[0]
    c = min((t[0] for t in group), key=lambda c: c.penalty)
    if all(_motorway_entry(index, t[0].entry, t[1].segs) for t in group):
        return None, "motorway"
    if len(group) == 1:
        rep, radius, in_core = group[0][1].rep, group[0][1].radius, group[0][1].in_core
    else:
        geom = shapely.union_all([cl.geom for _, cl in group])
        p = nearest_points(geom, geom.centroid)[0]
        coords = shapely.get_coordinates(geom)
        rep = (p.x, p.y)
        radius = float(np.max(np.hypot(coords[:, 0] - p.x, coords[:, 1] - p.y)))
        in_core = index.in_core(p.x, p.y)
    limit = float(ds.precision.get("street_max_radius_m", 125))
    if radius > limit:
        return None, f"street too long for a street-level match ({radius:.0f} m > {limit:.0f} m)"
    names = tuple(dict.fromkeys(t[0].entry.display for t in group))
    loc = Location(rep[0], rep[1], "street", max(25.0, radius), "osm_geometry", names,
                   c.flags + (flag,) + extra, c.penalty, in_core, f"street radius {radius:.0f} m")
    return loc, None


# ----------------------------------------------------------------------- intersections
def _entry_tree(index: AreaIndex, entry):
    cache = index.__dict__.setdefault("_entry_trees", {})
    t = cache.get(entry.key)
    if t is None:
        t = cache[entry.key] = STRtree([index.seg_geom[s] for s in entry.segs])
    return t


# A gap this small between two streets is a junction drawn slightly apart.
SMALL_GAP_M = 15.0


def _local_street(index: AreaIndex, entry, p, radius: float):
    """The pieces of a street within `radius` of point p, as one geometry."""
    tree = _entry_tree(index, entry)
    hits = tree.query(p.buffer(radius))
    if len(hits) == 0:
        return None
    return shapely.union_all([tree.geometries[i] for i in hits])


def _near_stretch(index: AreaIndex, ea, pa, eb, gap: float):
    """How much of street A runs within (gap + 20 m) of street B, around their closest point.

    A street that ends at the gap or meets the other at an angle has a short stretch there
    (tens of metres); a cul-de-sac beside a street has its own length; two parallel streets
    have a long one.  Returns (length, middle point of that stretch) or (inf, None).
    """
    la = _local_street(index, ea, pa, 200.0)
    lb = _local_street(index, eb, pa, 300.0)
    if la is None or lb is None:
        return math.inf, None
    part = la.intersection(lb.buffer(gap + 20.0))
    if part.is_empty:
        return math.inf, None
    if part.geom_type != "LineString":
        # Only the piece at the closest point: a street cut by a traffic diverter has a stub
        # on each side of the cross street, which together are not a side-by-side stretch.
        merged = linemerge(part) if part.geom_type == "MultiLineString" else part
        pieces = list(getattr(merged, "geoms", [merged]))
        part = min((g for g in pieces if g.length > 0), key=lambda g: g.distance(pa), default=None)
        if part is None:
            return math.inf, None
    return part.length, nearest_points(part, part.centroid)[0]


def _near_miss(index: AreaIndex, ea, ga, eb, gb, max_stretch: float):
    """(x, y, gap, stretch) of a near miss between two street pieces, or None if parallel.

    If the streets run side by side for longer than `max_stretch`, their "closest point" could
    be anywhere along the block (parallel streets), so it is not a near miss.
    """
    pa, pb = nearest_points(ga, gb)
    gap = pa.distance(pb)
    la, ma = _near_stretch(index, ea, pa, eb, gap)
    lb, mb = _near_stretch(index, eb, pb, ea, gap)
    stretch, mid, other = (la, ma, eb) if la <= lb else (lb, mb, ea)
    if stretch > max_stretch:
        return None
    if stretch <= 40.0:                   # one street ends at, or points to, the gap
        return (pa.x + pb.x) / 2, (pa.y + pb.y) / 2, gap, stretch
    # A short street beside the other one (a cul-de-sac): the middle of its near stretch.
    q = nearest_points(_local_street(index, other, mid, 300.0), mid)[0]
    return (mid.x + q.x) / 2, (mid.y + q.y) / 2, gap, stretch


def _pair_points(index: AreaIndex, ea, eb, ds):
    """Crossing points (or near misses) between two named streets, as (x, y, kind, gap)."""
    cache = index.__dict__.setdefault("_pair_cache", {})
    key = (ea.key, eb.key)
    if key in cache:
        return cache[key]
    near_gap = float(ds.precision.get("near_intersection_max_gap_m", 100))
    merge_m = float(ds.precision.get("intersection_merge_m", 60))
    max_stretch = float(ds.precision.get("near_side_by_side_max_m", 100))
    geoms_a = [index.seg_geom[s] for s in ea.segs]
    tree_b = _entry_tree(index, eb)
    out = []
    ia, ib = tree_b.query(geoms_a, predicate="intersects")
    for i, j in zip(ia, ib):
        inter = geoms_a[i].intersection(tree_b.geometries[j])
        for part in getattr(inter, "geoms", [inter]):
            if not part.is_empty:
                p = part if part.geom_type == "Point" else part.centroid
                out.append((p.x, p.y, "intersection", 0.0, ea.segs[i], eb.segs[j], 0.0))
    # Near misses: judge each distinct place once, from its closest pair of pieces.  A gap of a
    # few metres is a junction drawn slightly apart (a traffic diverter) and counts as a
    # crossing; when the streets also cross elsewhere it is a second junction, which makes the
    # record ambiguous.  Wider gaps only count when the streets do not cross at all.
    crossed = bool(out)
    ia, ib = tree_b.query(geoms_a, predicate="dwithin", distance=SMALL_GAP_M if crossed else near_gap)
    pairs = []
    for i, j in zip(ia, ib):
        if geoms_a[i].intersects(tree_b.geometries[j]):
            continue
        pa, pb = nearest_points(geoms_a[i], tree_b.geometries[j])
        pairs.append((pa.distance(pb), (pa.x + pb.x) / 2, (pa.y + pb.y) / 2, i, j))
    judged = [(p[0], p[1]) for p in out]
    for gap, mx, my, i, j in sorted(pairs):
        if any(math.hypot(mx - x, my - y) <= merge_m for x, y in judged):
            continue
        judged.append((mx, my))
        near = _near_miss(index, ea, geoms_a[i], eb, tree_b.geometries[j], max_stretch)
        if not near:                      # parallel streets, not a near miss
            continue
        x, y, g, stretch = near
        if g <= SMALL_GAP_M and stretch <= 40.0:
            out.append((x, y, "intersection", g, ea.segs[i], eb.segs[j], 0.0))
        elif not crossed:
            out.append((x, y, "near_intersection", g, ea.segs[i], eb.segs[j], stretch))
    if any(p[2] == "intersection" for p in out):
        out = [p for p in out if p[2] == "intersection"]
    cache[key] = out
    return out


def _cluster_points(points, merge_m):
    groups: list[list] = []
    for p in points:
        for g in groups:
            if math.hypot(g[0][0] - p[0], g[0][1] - p[1]) <= merge_m:
                g.append(p)
                break
        else:
            groups.append([p])
    return groups


def locate_intersection(cands_a: list[Candidate], cands_b: list[Candidate], index: AreaIndex, ds):
    merge_m = float(ds.precision.get("intersection_merge_m", 60))
    locs = []
    for ca in cands_a:
        for cb in cands_b:
            if ca.entry.key == cb.entry.key:
                continue
            pts = _pair_points(index, ca.entry, cb.entry, ds)
            for group in _cluster_points(pts, merge_m):
                xs = [p[0] for p in group]; ys = [p[1] for p in group]
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                in_core = index.in_core(cx, cy)
                if not in_core and ("several_types" in ca.flags or "several_types" in cb.flags):
                    continue           # a type picked among several needs the crossing in the city
                spread = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))
                kind = group[0][2]
                gap = min(p[3] for p in group)
                stretch = max(p[6] for p in group)
                motorway = any(index.seg_type[p[4]] in MOTORWAY_TYPES or
                               index.seg_type[p[5]] in MOTORWAY_TYPES for p in group)
                if kind == "intersection":
                    prec = max(10.0, spread)
                elif stretch <= 40.0:
                    prec = gap / 2 + 10.0
                else:                     # a cul-de-sac beside the street: along it, too
                    prec = math.hypot(gap / 2, stretch / 2) + 10.0
                flags = ca.flags + cb.flags + (("motorway",) if motorway else ()) + (
                    ("side_by_side",) if kind != "intersection" and stretch > 40.0 else ()) + (
                    ("small_gap",) if kind == "intersection" and gap > 0 else ())
                detail = (f"{len(group)} crossing point(s)" if kind == "intersection" else
                          f"gap {gap:.0f} m" + (f", side by side for {stretch:.0f} m" if stretch > 40.0 else ""))
                locs.append(Location(cx, cy, kind, prec, "osm_geometry",
                                     (ca.entry.display, cb.entry.display), flags,
                                     ca.penalty + cb.penalty, in_core, detail))
    if not locs:
        return None, "streets do not meet" + _gap_note(cands_a, cands_b, index)
    loc, reason = choose(locs, ds, "intersection")
    if loc and "motorway" in loc.flags and not ds.accept.get("highway_interchange", False):
        return None, "highway interchange (not accepted by policy)"
    return loc, reason


def _gap_note(cands_a, cands_b, index: AreaIndex) -> str:
    """How far apart the closest candidate streets are, for the unmatched report."""
    best = None
    for ca in cands_a[:4]:
        ga = shapely.union_all([index.seg_geom[s] for s in ca.entry.segs])
        for cb in cands_b[:4]:
            gb = shapely.union_all([index.seg_geom[s] for s in cb.entry.segs])
            d = ga.distance(gb)
            best = d if best is None else min(best, d)
    return f" (closest {best:.0f} m)" if best is not None else ""


def _motorway_entry(index: AreaIndex, entry, segs=None) -> bool:
    """True when the matched street (or the pieces used) are all motorway carriageways."""
    types = {index.seg_type[s] for s in (segs or entry.segs)}
    return bool(types) and types <= MOTORWAY_TYPES
