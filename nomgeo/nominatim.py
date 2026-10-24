"""Named places (POIs) through the Nominatim search API, validated before use.

Nominatim is only asked about records that do not resolve to a street.  A result
is accepted when its name actually matches the record, it lies in the area, it is
small enough for the precision target, and it is the only such place.
"""
from __future__ import annotations

import hashlib
import json
import math

import requests

from .cache import KVCache
from .locate import Location
from .normalize import CANONICAL_TOKEN, clean

_STOP = {"THE", "OF", "AND", "AT", "IN", "ON", "A"}
_SKIP_CATEGORIES = {"highway", "boundary", "place"}


class NominatimClient:
    def __init__(self, base_url: str, cache: KVCache | None, data_version: str):
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.data_version = data_version
        self._session = None

    def search(self, **params) -> list[dict]:
        key = hashlib.sha1((json.dumps(sorted(params.items())) + self.data_version).encode()).hexdigest()
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        if self._session is None:
            self._session = requests.Session()
        resp = self._session.get(f"{self.base_url}/search", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if self.cache:
            self.cache.put(key, data)
        return data


def _tokens(text: str) -> set[str]:
    """Words of a name with abbreviations unified ('SQ' and 'SQUARE' -> 'SQ')."""
    return {CANONICAL_TOKEN.get(t, t) for t in clean(text).split() if t not in _STOP}


def name_matches(query: str, names: list[str]) -> bool:
    q = _tokens(query)
    if not q:
        return False
    for n in names:
        t = _tokens(n)
        if not t:
            continue
        # The place must be called what the record says (or the record must spell out the
        # place's full multi-word name), not merely contain one of its words: "GILMAN" is not
        # the "924 Gilman" club and "BROADWAY" not the Broadway subway stop.
        if t == q or (len(t) >= 2 and t <= q) or "".join(clean(n).split()) == "".join(clean(query).split()):
            return True
    return False


def find_poi(text: str, index, ds, client: NominatimClient | None):
    """Return (Location, None) for a unique small named place, else (None, reason or None)."""
    if client is None or not ds.accept.get("poi", True):
        return None, None
    q = clean(text)
    if len(q.replace(" ", "")) < 4:
        return None, None
    left, top, right, bottom = index.bbox
    try:
        results = client.search(q=q, format="jsonv2", limit=10, bounded=1, namedetails=1, dedupe=1,
                                viewbox=f"{left},{top},{right},{bottom}")
    except Exception as exc:
        return None, f"nominatim error: {exc}"
    limit = float(ds.precision.get("poi_max_radius_m", 150))
    valid = []
    for r in results:
        cat, typ = r.get("category"), r.get("type")
        # place=* nodes stand for areas of unknown size (localities, hamlets); squares are small.
        if (cat in _SKIP_CATEGORIES and not (cat == "place" and typ == "square")) \
                or int(r.get("place_rank", 30)) < 21:
            continue
        names = list((r.get("namedetails") or {}).values()) or [r.get("name") or ""]
        if not name_matches(q, names):
            continue
        lat, lon = float(r["lat"]), float(r["lon"])
        s, n, w, e = map(float, r["boundingbox"])
        radius = 0.5 * math.hypot((n - s) * 110_574, (e - w) * 111_320 * math.cos(math.radians(lat)))
        x, y = index.proj.point(lon, lat)
        valid.append((x, y, radius, r, names[0] if names else r.get("display_name", "")))
    if not valid:
        return None, None
    # A place called exactly what the record says beats places that merely contain it
    # ("KENDALL SQ": Kendall Square, not Kendall Square Cinema or Kendall Square Garage).
    exact = [v for v in valid if any(_tokens(n) == _tokens(q) for n in [v[4]] + list(
        (v[3].get("namedetails") or {}).values()))]
    if exact:
        valid = exact
    far = [v for v in valid if math.hypot(v[0] - valid[0][0], v[1] - valid[0][1]) > limit]
    if far:
        return None, f"ambiguous place name: {1 + len(far)} places called '{q.title()}'"
    x, y, radius, r, name = valid[0]
    if radius > limit:
        return None, f"named place too large for the precision target ({radius:.0f} m)"
    return Location(x, y, "poi", max(25.0, radius), "nominatim", (name,),
                    (f"poi:{r.get('category')}/{r.get('type')}",), 0, index.in_core(x, y),
                    r.get("display_name", "")[:120]), None
