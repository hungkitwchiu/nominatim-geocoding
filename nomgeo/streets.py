"""Per-area street index built from Nominatim's own tables.

For one search area (a bounding box, optionally with a core polygon such as the
city boundary) the index holds:

* every named street piece (OSM way) with its canonicalised name variants,
* OSM address points attached to those streets (`placex.housenumber`),
* house-number ranges from OSM interpolation lines and US Census TIGER,

all projected to a local metric plane so distances are in metres.  Building it
takes 5-40 seconds per city (most of it checking the TIGER ranges); it is pickled
to the cache directory and reused until the database or this code changes.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import shapely
from rapidfuzz.distance import OSA
from shapely import wkb
from shapely.ops import linemerge, nearest_points
from shapely.strtree import STRtree

from .normalize import StreetName, clean, parse_osm_ref, prefix_form, split_street

NAME_KEYS = ("name", "alt_name", "old_name", "official_name", "loc_name", "short_name", "name:en")
MOTORWAY_TYPES = {"motorway", "motorway_link"}
# Street pieces of the same name closer than this belong to one physical street.
CLUSTER_GAP_M = 30.0


class Projection:
    """Local equirectangular projection; < 0.5 % error across a metro area."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = math.cos(math.radians(lat0)) * 111_320.0
        self.ky = 110_574.0

    def fwd_xy(self, coords: np.ndarray) -> np.ndarray:
        out = np.empty_like(coords)
        out[:, 0] = (coords[:, 0] - self.lon0) * self.kx
        out[:, 1] = (coords[:, 1] - self.lat0) * self.ky
        return out

    def geom(self, g):
        return shapely.transform(g, self.fwd_xy)

    def point(self, lon: float, lat: float) -> tuple[float, float]:
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def inv(self, x: float, y: float) -> tuple[float, float]:
        """Return (lat, lon)."""
        return self.lat0 + y / self.ky, self.lon0 + x / self.kx


@dataclass
class Cluster:
    """One physically connected stretch of a named street."""
    segs: list[int]
    geom: object                     # merged projected geometry
    rep: tuple[float, float]         # representative point (middle of the street)
    radius: float                    # max distance from rep to the street's vertices
    in_core: bool


@dataclass
class StreetEntry:
    key: str                         # canonical key, e.g. "N 1ST ST"
    name: StreetName
    display: str                     # one OSM spelling, e.g. "North 1st Street"
    segs: list[int] = field(default_factory=list)
    alt: bool = True                 # True if only known through alt/old/official names
    is_ref: bool = False             # route-number entry ("REF I 80")
    _clusters: list[Cluster] | None = None


def parse_housenumbers(text: str) -> list[int]:
    """'1326;1320' -> [1326, 1320]; '750 b' -> [750]; '1001 1005' -> [1001, 1005]."""
    out = []
    for part in re.split(r"[;,]", text or ""):
        nums = [int(n) for n in re.findall(r"\d+", part)]
        if not nums:
            continue
        out.append(nums[0])
        if len(nums) == 2 and 0 < nums[1] - nums[0] <= 100:
            out.append(nums[1])
    return out


class AreaIndex:
    def __init__(self, bbox, core_wkb: bytes | None, data_version: str):
        left, top, right, bottom = bbox
        self.bbox = bbox
        self.data_version = data_version
        self.proj = Projection((left + right) / 2, (top + bottom) / 2)
        self.core = self.proj.geom(wkb.loads(core_wkb)) if core_wkb else None
        if self.core is not None:
            shapely.prepare(self.core)
        self.seg_place: list[int] = []
        self.seg_type: list[str] = []
        self.seg_geom: list = []
        self.seg_name: list[str] = []
        self.seg_keys: list[set[str]] = []
        self.dropped_points = 0
        self.dropped_tiger = 0
        self.entries: dict[str, StreetEntry] = {}
        self.by_base: dict[str, set[str]] = {}
        self.by_squash: dict[str, set[str]] = {}
        self.by_squash_base: dict[str, set[str]] = {}
        self.by_prefix_form: dict[str, set[str]] = {}
        self.by_ref: dict[tuple[str, str], set[str]] = {}
        self.first_tokens: set[str] = set()
        self.points: dict[int, list[tuple[int, float, float]]] = {}
        self.ranges: dict[int, list[tuple[int, int, int, object, str]]] = {}
        self.place_to_seg: dict[int, int] = {}
        self._tree: STRtree | None = None
        self._squash_sorted: list[str] | None = None
        self._base_choices: list[str] | None = None

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, conn, bbox, core_wkb: bytes | None, data_version: str) -> "AreaIndex":
        idx = cls(bbox, core_wkb, data_version)
        left, top, right, bottom = bbox
        with conn.cursor() as cur:
            cur.execute(
                """SELECT place_id, type, hstore_to_json(name)::text, ST_AsBinary(geometry)
                     FROM placex
                    WHERE osm_type = 'W' AND class = 'highway' AND rank_address IN (26, 27)
                      AND name IS NOT NULL
                      AND geometry && ST_MakeEnvelope(%s, %s, %s, %s, 4326)""",
                (left, bottom, right, top))
            rows = cur.fetchall()
        geoms = shapely.from_wkb([r[3] for r in rows])
        # Closed loops (a crescent mapped as one closed way) come back as polygons; use their
        # outline, or a crossing would be placed in the middle of the loop.
        geoms = [g.boundary if g.geom_type in ("Polygon", "MultiPolygon") else g for g in geoms]
        geoms = shapely.transform(geoms, idx.proj.fwd_xy)
        for (place_id, typ, names_json, _), g in zip(rows, geoms):
            names = json.loads(names_json)
            seg = len(idx.seg_place)
            idx.seg_place.append(place_id)
            idx.seg_type.append(typ)
            idx.seg_geom.append(g)
            idx.seg_name.append(names.get("name") or names.get("ref") or "")
            idx.seg_keys.append(set())
            idx.place_to_seg[place_id] = seg
            for k in NAME_KEYS:
                for value in (names.get(k) or "").split(";"):
                    if value.strip():
                        idx._add_name(value.strip(), seg, alt=(k != "name"))
            for net, num in parse_osm_ref(names.get("ref") or ""):
                idx._add_ref(net, num, seg)
        idx._load_numbers(conn)
        return idx

    def _add_name(self, display: str, seg: int, alt: bool) -> None:
        sn = split_street(display)
        if not sn.base:
            return
        key = sn.key()
        e = self.entries.get(key)
        if e is None:
            e = self.entries[key] = StreetEntry(key, sn, display)
            self.by_base.setdefault(sn.base_key, set()).add(key)
            self.by_squash.setdefault(sn.squash_key, set()).add(key)
            base_squash = "".join(sn.raw[1:] if sn.predir else sn.raw)
            self.by_squash_base.setdefault(base_squash, set()).add(key)
            self.by_prefix_form.setdefault(prefix_form(display), set()).add(key)
            toks = clean(display).split()
            if toks:
                self.first_tokens.add(toks[0])
        if not e.segs or e.segs[-1] != seg:
            e.segs.append(seg)
        self.seg_keys[seg].add(key)
        if not alt:
            if e.alt:
                e.display = display
            e.alt = False

    def _add_ref(self, net: str, num: str, seg: int) -> None:
        key = f"REF {net} {num}"
        e = self.entries.get(key)
        if e is None:
            e = self.entries[key] = StreetEntry(key, StreetName(None, (net, num)), f"{net} {num}",
                                                alt=False, is_ref=True)
        e.segs.append(seg)
        self.by_ref.setdefault((net, num), set()).add(key)

    def _load_numbers(self, conn) -> None:
        ids = self.seg_place
        if not ids:
            return
        verdict: dict[tuple[str, int], bool] = {}
        aliases: set[tuple[str, int]] = set()
        with conn.cursor() as cur:
            cur.execute("""SELECT parent_place_id, housenumber, address->'street', ST_X(centroid), ST_Y(centroid)
                             FROM placex
                            WHERE parent_place_id = ANY(%s) AND housenumber IS NOT NULL""", (ids,))
            for parent, hn, addr_street, lon, lat in cur:
                seg = self.place_to_seg.get(parent)
                if seg is not None and not addr_street and not self._clearly_nearest(seg, lon, lat):
                    # A number with no street name was attached to the nearest road; keep it only
                    # when that road is clearly the nearest (not a corner lot between two streets).
                    self.dropped_points += 1
                    continue
                if seg is not None and addr_street:
                    # Nominatim attaches a house to the nearest road when no street with its
                    # addr:street name is close; such numbers would land on the wrong street.
                    ok = verdict.get((addr_street, seg))
                    if ok is None:
                        ok, alias = self._addr_street_fits(addr_street, seg)
                        verdict[(addr_street, seg)] = ok
                        if alias:
                            aliases.add((addr_street, seg))
                    if not ok:
                        self.dropped_points += 1
                        continue
                x, y = self.proj.point(lon, lat)
                for n in parse_housenumbers(hn):
                    self.points.setdefault(parent, []).append((n, x, y))
        # Spellings used by the addresses become alternative names of their street, which
        # also covers typos in OSM's own street names ("Golden Turyle Circle").
        for addr_street, seg in sorted(aliases):
            # Only a spelling no real street in the area has: an addr:street typo must not
            # merge this street with a different street of that name elsewhere.
            existing = self.entries.get(split_street(addr_street).key())
            if existing is None or existing.alt:
                self._add_name(addr_street, seg, alt=True)
        with conn.cursor() as cur:
            for table, source in (("location_property_osmline", "osm_interpolation"),
                                  ("location_property_tiger", "tiger")):
                extra = " AND startnumber IS NOT NULL" if source == "osm_interpolation" else ""
                cur.execute(f"""SELECT parent_place_id, startnumber, endnumber, step, ST_AsBinary(linegeo)
                                  FROM {table} WHERE parent_place_id = ANY(%s){extra}""", (ids,))
                rows = cur.fetchall()
                if not rows:
                    continue
                lines = shapely.transform(shapely.from_wkb([r[4] for r in rows]), self.proj.fwd_xy)
                for (parent, start, end, step, _), line in zip(rows, lines):
                    if start is None or end is None or line is None:
                        continue
                    if source == "tiger" and not self._follows_street(line, parent):
                        self.dropped_tiger += 1
                        continue
                    self.ranges.setdefault(parent, []).append(
                        (min(start, end), max(start, end), step or 1, line, source))

    def _clearly_nearest(self, seg: int, lon: float, lat: float) -> bool:
        """Is street piece `seg` within 60 m of the point and at least 10 m nearer than any
        street with a different name?"""
        p = shapely.Point(*self.proj.point(lon, lat))
        d = self.seg_geom[seg].distance(p)
        if d > 60.0:
            return False
        keys = self.seg_keys[seg]
        for i in self.tree().query(p.buffer(d + 10.0)):
            if not (self.seg_keys[i] & keys) and self.seg_geom[i].distance(p) < d + 10.0:
                return False
        return True

    def _addr_street_fits(self, addr_street: str, seg: int) -> tuple[bool, bool]:
        """Is an address point's addr:street the street it is attached to?

        Returns (fits, is_new_spelling).  Same name with the direction moved or left out
        ("6th Street East" on East 6th Street, "Margaret Street" on East Margaret Street) fits;
        a small spelling difference fits and is remembered as an alternative name
        ("Oconner Crescent" on O Connor Crescent); a different name does not
        ("Terminal Drive" points attached to Crittenden Drive).
        """
        sn = split_street(addr_street)
        if not sn.base or sn.key() in self.seg_keys[seg]:
            return True, False
        a_sq = "".join(sn.raw[1:] if sn.predir else sn.raw)
        for k in sorted(self.seg_keys[seg]):
            e = self.entries.get(k)
            if e is None:
                continue
            en = e.name
            if en.base_key == sn.base_key:
                # Same name: the direction may be moved or left out, but the type must agree
                # ("Sunnyhills Court" numbers do not belong on Sunnyhills Drive).
                da = {d for d in (sn.predir, sn.postdir) if d}
                db = {d for d in (en.predir, en.postdir) if d}
                types_agree = not sn.suffix or not en.suffix or bool(sn.suffix & en.suffix)
                if (not da or not db or da == db) and types_agree:
                    return True, False
                continue
            b_sq = "".join(en.raw[1:] if en.predir else en.raw)
            if a_sq == b_sq:
                return True, True
            if (len(a_sq) >= 6 and re.findall(r"\d+", a_sq) == re.findall(r"\d+", b_sq)
                    and OSA.distance(a_sq, b_sq) <= (1 if len(a_sq) < 10 else 2)):
                return True, True
        return False, False

    def _follows_street(self, line, parent: int) -> bool:
        """Does a TIGER range run along the street Nominatim attached it to?

        Nominatim attaches a TIGER line to the nearest road when no street of the same name
        is close, which can put one street's numbers on a cross street.  Keep the range only
        if at least half of it lies within 30 m of a street sharing a name with the parent.
        """
        seg = self.place_to_seg.get(parent)
        if seg is None or line.length == 0:
            return False
        keys = self.seg_keys[seg]
        near = [i for i in self.tree().query(line.buffer(30.0)) if self.seg_keys[i] & keys]
        if not near:
            return False
        street = shapely.union_all([self.seg_geom[i] for i in near]).buffer(30.0)
        return line.intersection(street).length >= 0.5 * line.length

    # ------------------------------------------------------------- lookups
    def tree(self) -> STRtree:
        if self._tree is None:
            self._tree = STRtree(self.seg_geom)
        return self._tree

    def prefix_lookup(self, prefix: str) -> set[str]:
        """Entries whose spelled-out full name starts with `prefix` (for truncated fields)."""
        if self._squash_sorted is None:
            self._squash_sorted = sorted(self.by_prefix_form)
        keys = self._squash_sorted
        i = bisect.bisect_left(keys, prefix)
        out: set[str] = set()
        while i < len(keys) and keys[i].startswith(prefix):
            out |= self.by_prefix_form[keys[i]]
            i += 1
        return out

    def base_choices(self) -> list[str]:
        if self._base_choices is None:
            self._base_choices = sorted(self.by_base)
        return self._base_choices

    def in_core(self, x: float, y: float) -> bool:
        return True if self.core is None else bool(self.core.contains(shapely.Point(x, y)))

    def types_near(self, x: float, y: float, dist: float = 15.0) -> set[str]:
        hits = self.tree().query(shapely.Point(x, y).buffer(dist))
        return {self.seg_type[i] for i in hits}

    def clusters(self, entry: StreetEntry) -> list[Cluster]:
        """Split an entry's pieces into physically separate streets (same name, other town)."""
        if entry._clusters is not None:
            return entry._clusters
        segs = entry.segs
        parent = list(range(len(segs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        if len(segs) > 1:
            geoms = [self.seg_geom[s] for s in segs]
            local = STRtree(geoms)
            left, right = local.query(geoms, predicate="dwithin", distance=CLUSTER_GAP_M)
            for a, b in zip(left, right):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        groups: dict[int, list[int]] = {}
        for i, s in enumerate(segs):
            groups.setdefault(find(i), []).append(s)
        out = []
        for members in groups.values():
            geom = shapely.union_all([self.seg_geom[s] for s in members])
            merged = linemerge(geom) if geom.geom_type == "MultiLineString" else geom
            if merged.geom_type == "LineString":
                p = merged.interpolate(0.5, normalized=True)
            else:
                p = nearest_points(geom, geom.centroid)[0]
            coords = shapely.get_coordinates(geom)
            radius = float(np.max(np.hypot(coords[:, 0] - p.x, coords[:, 1] - p.y))) if len(coords) else 0.0
            out.append(Cluster(members, geom, (p.x, p.y), radius, self.in_core(p.x, p.y)))
        entry._clusters = out
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tree"] = None            # STRtrees are rebuilt lazily after unpickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.core is not None:
            shapely.prepare(self.core)
