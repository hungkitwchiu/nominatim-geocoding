"""Search areas: the city boundary ("core") plus a surrounding box ("outer").

Candidates inside the core are preferred; the outer box catches incidents just
across the city line (mutual aid, unincorporated pockets, mailing-address cities).
The boundary is looked up in Nominatim, so a new city needs no hand-made viewbox.
"""
from __future__ import annotations

import hashlib
import logging
import math
import pickle
from pathlib import Path

import requests

from . import __version__
from .streets import AreaIndex

LOG = logging.getLogger(__name__)

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico",
}


def find_boundary(conn, nominatim_url: str, city: str, state: str | None):
    """Return (osm_id, WKB polygon, (left, top, right, bottom)) of a city boundary, or None."""
    params = {"city": city, "countrycodes": "us", "format": "jsonv2", "limit": 10,
              "featureType": "settlement"}
    if state:
        params["state"] = STATE_NAMES.get(state.upper(), state)
    try:
        hits = requests.get(f"{nominatim_url}/search", params=params, timeout=30).json()
    except Exception as exc:  # pragma: no cover - network failure
        LOG.warning("boundary lookup failed for %s, %s: %s", city, state, exc)
        return None
    for h in hits:
        if h.get("osm_type") != "relation" or h.get("category") != "boundary":
            continue
        with conn.cursor() as cur:
            cur.execute("""SELECT ST_AsBinary(geometry), ST_XMin(geometry), ST_YMax(geometry),
                                  ST_XMax(geometry), ST_YMin(geometry)
                             FROM placex WHERE osm_type = 'R' AND osm_id = %s AND class = 'boundary'
                            ORDER BY rank_address DESC LIMIT 1""", (int(h["osm_id"]),))
            row = cur.fetchone()
        if row:
            return int(h["osm_id"]), bytes(row[0]), (row[1], row[2], row[3], row[4])
    return None


def buffered_bbox(bbox, km: float):
    left, top, right, bottom = bbox
    dlat = km / 110.574
    dlon = km / (111.320 * math.cos(math.radians((top + bottom) / 2)))
    return [left - dlon, top + dlat, right + dlon, bottom - dlat]


def load_area(conn, ds, data_version: str, use_cache: bool = True) -> tuple[AreaIndex, dict]:
    """Build (or load from cache) the street index for a dataset's area."""
    boundary = find_boundary(conn, ds.nominatim_url, ds.city, ds.state) if ds.city else None
    core_wkb = boundary[1] if boundary else None
    bbox = ds.bbox or (buffered_bbox(boundary[2], ds.buffer_km) if boundary else None)
    if bbox is None:
        raise ValueError(f"{ds.name}: no bbox given and no boundary found for {ds.city}, {ds.state}")
    info = {"bbox": [round(v, 5) for v in bbox], "core_osm_relation": boundary[0] if boundary else None}
    key = hashlib.sha1(json_key(bbox, info["core_osm_relation"], data_version).encode()).hexdigest()[:16]
    path = Path(ds.cache_dir) / "areas" / f"{ds.name}-{key}.pkl"
    if use_cache and path.exists():
        with open(path, "rb") as fh:
            index = pickle.load(fh)
        info["cache"] = "hit"
    else:
        index = AreaIndex.build(conn, bbox, core_wkb, data_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
        info["cache"] = "built"
    info.update(segments=len(index.seg_place), streets=sum(1 for e in index.entries.values() if not e.is_ref),
                address_points=sum(len(v) for v in index.points.values()),
                number_ranges=sum(len(v) for v in index.ranges.values()),
                tiger_ranges=sum(1 for v in index.ranges.values() for r in v if r[4] == "tiger"),
                dropped_points=index.dropped_points,
                dropped_tiger_ranges=getattr(index, "dropped_tiger", 0))
    return index, info


def _index_code_hash() -> str:
    """Hash of the code that shapes the index; editing it invalidates cached indexes."""
    here = Path(__file__).parent
    h = hashlib.sha1()
    for name in ("normalize.py", "streets.py"):
        h.update((here / name).read_bytes())
    return h.hexdigest()[:10]


def json_key(bbox, relation, data_version) -> str:
    return f"{[round(v, 5) for v in bbox]}|{relation}|{data_version}|{__version__}|{_index_code_hash()}"
