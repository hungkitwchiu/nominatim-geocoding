"""Resolve a street string against the streets that exist in the area.

Matching proceeds through tiers and stops at the first tier that finds anything,
so a looser rule is only tried when every stricter one failed:

  exact name  ->  spacing ("VANDORN" = "Van Dorn", "NIEMANBL" = "Nieman Blvd")
  ->  extra trailing type  ->  type as part of the name ("CEDAR DALE" = "Cedardale")
  ->  words after the type  ->  a town name after the street  ->  unit number after the name
  ->  redaction wildcard ("KAI00" = "Kains")  ->  truncated field ("SILVERCREEK VALL")
  ->  first word left out ("RANCHO VERDE" = "El Rancho Verde"), with typo candidates
  ->  typo tolerance (1-2 edits by length, unique, digits must agree)

Within a tier, directional and street-type differences are scored, and only the
least-relaxed variants survive (an exact "MAIN ST" hides "MAIN AVE").  A changed type on
the exact name competes with the other readings of the whole text, which rank first.
Every relaxation is recorded as a flag and adds to a penalty so results can be audited.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import process
from rapidfuzz.distance import OSA

from .normalize import (DIRECTIONS, HIGHWAY_TYPES, ROAD_TYPES, StreetName, clean, parse_route,
                        prefix_form, split_street, suffix_options)
from .streets import AreaIndex, StreetEntry

TIER_PENALTY = {"exact": 0, "spacing": 1, "extra_type": 1, "type_as_name": 1, "leading_word_added": 2,
                "redacted": 2, "truncated": 2, "fuzzy1": 3, "fuzzy_short": 4, "fuzzy2": 5}
# One edit in a name of 5+ letters, two edits in 7+ letters, one edit in a 4-letter name.
FUZZY_FLAG = {"fuzzy1": "typo_1", "fuzzy2": "typo_2", "fuzzy_short": "typo_short"}
# Relaxations not to be stacked on top of another one.
WEAK_RELAXATIONS = {"type_changed", "typo_1", "typo_2", "typo_short", "truncated", "redacted",
                    "leading_word_added"}
# A fused type abbreviation at the end of a single token ("NIEMANBL" = Nieman Boulevard).
_FUSED_TYPE = re.compile(r"([A-Z]{5,})(AVE|AV|BLVD|BL|DR|CT|WAY|WY|ST|RD|LN|CL|PL|TR|CR)")


@dataclass(frozen=True)
class Candidate:
    entry: StreetEntry
    penalty: int
    flags: tuple[str, ...]

    @property
    def key(self) -> str:
        return self.entry.key


def _dir_score(inp: StreetName, ent: StreetName):
    di = {d for d in (inp.predir, inp.postdir) if d}
    de = {d for d in (ent.predir, ent.postdir) if d}
    if di == de:
        if (inp.predir, inp.postdir) == (ent.predir, ent.postdir):
            return 0, ()
        return 1, ("direction_moved",)
    if not di:
        return 2, ("direction_added",)
    if not de:
        return 2, ("direction_dropped",)
    return None, ()                       # conflicting directions: a different street


def _suffix_score(inp: StreetName, ent: StreetName, unreliable=frozenset()):
    es = next(iter(ent.suffix)) if ent.suffix else None
    if not inp.suffix:
        return (1, ("type_missing",)) if es else (0, ())
    if es in inp.suffix:
        return (0, ("type_ambiguous",)) if len(inp.suffix) > 1 else (0, ())
    if es is None:
        return 1, ("type_extra",)
    if inp.suffix & unreliable:
        # The dataset writes this type for streets of any type (Lincoln: "HUNTINGTON ST").
        return 1, ("type_unreliable",)
    return 3, ("type_changed",)


def _score_entries(sn: StreetName, keys, index: AreaIndex, tier: str, extra_flags=(),
                   unreliable=frozenset()):
    """Score candidate entries for direction/type agreement and keep the least relaxed."""
    scored = []
    for key in sorted(keys):            # sorted: the same result on every run
        e = index.entries[key]
        if e.is_ref:
            continue
        dpen, dflags = _dir_score(sn, e.name)
        if dpen is None:
            continue
        spen, sflags = _suffix_score(sn, e.name, unreliable)
        scored.append((e, dpen, spen, dflags + sflags))
    if not scored:
        return []
    # A compatible street type hides a changed one ("MAIN ST" never becomes "MAIN AVE" when a
    # Main St exists).  Directional variants all stay: records drop directionals so often that
    # the geometry or house numbers decide between "5TH ST", "N 5TH ST" and "S 5TH ST", in
    # penalty order.
    if any(s < 3 for _, _, s, _ in scored):
        scored = [t for t in scored if t[2] < 3]
    elif len({e.name.suffix for e, _, _, _ in scored}) > 1:
        # Only a changed type is left.  When the name has one type in the area that is a fair
        # guess ("FONTANOSO RD" = Fontanoso Way; "CORNHUSKER RD" = Cornhusker Highway and its
        # West part).  When it has several, the house number or the cross street must pick
        # one, inside the city (locate.py): "2100 SHATTUCK BLVD" is on Shattuck Avenue, not
        # Shattuck Place.  Only between plain road types and on the exact name: "MAGAZINE
        # BEACH" is not Magazine Street, and a highway's type is never changed.
        if tier != "exact" or not sn.suffix or not sn.suffix <= ROAD_TYPES:
            return []
        scored = [t for t in scored if t[0].name.suffix and t[0].name.suffix <= ROAD_TYPES | HIGHWAY_TYPES]
        extra_flags = tuple(extra_flags) + ("several_types",)
    out = []
    for e, dpen, spen, flags in scored:
        pen = TIER_PENALTY[tier] + dpen + spen + (1 if e.alt else 0)
        fl = tuple(extra_flags) + flags + (("alt_name",) if e.alt else ())
        if tier != "exact":
            fl = (FUZZY_FLAG.get(tier, tier),) + fl
        out.append(Candidate(e, pen, fl))
    return out


def _typo_distance(a: StreetName, b: StreetName) -> int:
    """Edits between two names, word by word where the words line up.

    The keys abbreviate type words inside names ("SALT CREEK" is SALT CRK, "Salt Lake" is
    SALT LK), which can make different words look alike; so two words that were both
    abbreviated are compared as written, and any other pair in whichever form is closer
    ("KEYS" and Keyes as written, "EX" and Expressway as EX and EXPY).
    """
    ra, rb = (a.raw[1:] if a.predir else a.raw), (b.raw[1:] if b.predir else b.raw)
    if len(ra) == len(rb) == len(a.base) == len(b.base):
        total = 0
        for wa, ca, wb, cb in zip(ra, a.base, rb, b.base):
            full_a, full_b = wa != ca and len(wa) >= 4, wb != cb and len(wb) >= 4
            total += OSA.distance(wa, wb) if full_a and full_b else min(OSA.distance(wa, wb), OSA.distance(ca, cb))
        return total
    return min(OSA.distance(" ".join(ra), " ".join(rb)), OSA.distance(a.base_key, b.base_key))


def _fuzzy_keys(sn: StreetName, index: AreaIndex, min_len: int):
    q = sn.base_key
    # Also as written: a type word inside the name is abbreviated in the key ("KEYS" is "KYS"),
    # which would put "Keyes" two edits away instead of one.
    raw = " ".join(sn.raw[1:] if sn.predir else sn.raw)
    letters = max(q.replace(" ", ""), raw.replace(" ", ""), key=len)
    if len(letters) < min_len:
        return [], None
    max_d = 1 if len(letters) < 7 else 2
    found: dict[str, int] = {}
    for query in {q, raw}:
        for c, d, _ in process.extract(query, index.base_choices(), scorer=OSA.distance,
                                       score_cutoff=max_d, limit=25):
            found[c] = min(d, found.get(c, d))
    digits = re.findall(r"\d+", q)
    hits = []
    for c, d in found.items():
        if d == 0 or re.findall(r"\d+", c) != digits:
            continue
        keys = {k for k in index.by_base[c] if _typo_distance(sn, index.entries[k].name) <= max_d}
        if keys:
            hits.append((d, keys))
    if not hits:
        return [], None
    best = min(d for d, _ in hits)
    keys = set().union(*[ks for d, ks in hits if d == best])
    return keys, "fuzzy_short" if len(letters) == 4 else f"fuzzy{best}"


def resolve(text: str, index: AreaIndex, ds, truncated: bool = False,
            wildcard_first: bool = False) -> list[Candidate]:
    """All plausible streets for `text`, least relaxed first."""
    return sorted(_resolve_tiers(text, index, ds, truncated, wildcard_first), key=lambda c: c.penalty)


def resolve_typo(text: str, index: AreaIndex, ds) -> list[Candidate]:
    """Second chance: streets one edit away, even when the exact name exists elsewhere.

    Used when the exactly-named streets of an intersection do not meet: Cambridge's
    "HAWTHORNE ST & BRATTLE ST" names Hawthorn Street, while "Hawthorne Street"
    exists only in neighbouring towns.
    """
    sn = split_street(text, ds.single_letter_suffixes)
    if len(sn.base_key.replace(" ", "")) < int(ds.precision.get("fuzzy_min_length", 5)):
        return []
    hits = process.extract(sn.base_key, index.base_choices(), scorer=OSA.distance, score_cutoff=1, limit=25)
    digits = re.findall(r"\d+", sn.base_key)
    keys = set()
    for c, d, _ in hits:
        if d == 1 and re.findall(r"\d+", c) == digits:
            keys |= index.by_base[c]
    cands = _score_entries(sn, keys, index, "fuzzy1", unreliable=ds.unreliable_types)
    return sorted((c for c in cands if c.penalty <= int(ds.precision.get("max_penalty", 6))),
                  key=lambda c: c.penalty)


def resolve_other_types(text: str, index: AreaIndex, ds, exclude: set[str],
                        other_bases: set[str] = frozenset()) -> list[Candidate]:
    """Second chance: same-named streets of another road type, for an intersection whose
    exactly-named streets do not meet ("STEVENS CT & MT ISABEL DR": Stevens Lane crosses
    Mount Isabel Drive).  Only between ordinary road types, keeping the record's direction,
    and never a street named like the other side ("WITCHDUCK CT" is not Witchduck Road meeting
    North Witchduck Road); the caller must confirm the result with a real crossing.
    """
    sn = split_street(text, ds.single_letter_suffixes)
    if not sn.base or not sn.suffix or not sn.suffix <= ROAD_TYPES or sn.base_key in other_bases:
        return []
    out = []
    for key in sorted(index.by_base.get(sn.base_key, set())):
        e = index.entries[key]
        if e.is_ref or key in exclude or not e.name.suffix or e.name.suffix & sn.suffix \
                or not e.name.suffix <= ROAD_TYPES | HIGHWAY_TYPES:
            continue
        if (sn.predir and e.name.predir and sn.predir != e.name.predir) or \
                (sn.postdir and e.name.postdir and sn.postdir != e.name.postdir):
            continue
        out.append(Candidate(e, 3, ("type_changed", "second_chance")))
    return out


def _resolve_tiers(text: str, index: AreaIndex, ds, truncated: bool = False,
                   wildcard_first: bool = False, whole_text_only: bool = False) -> list[Candidate]:
    """The candidates of the first tier that finds any.  `whole_text_only` leaves out the exact
    name and the tiers that drop words or allow typos (to look past a changed type)."""
    min_len = int(ds.precision.get("fuzzy_min_length", 5))
    max_pen = int(ds.precision.get("max_penalty", 6))
    out: list[Candidate] = []

    route = parse_route(text, ds.state)
    if route:
        net, num = route
        keys = set()
        for (n2, num2), ks in index.by_ref.items():
            if num2 == num and (n2 == net or (net is None and n2 in ("US", "SR"))):
                keys |= ks
        for k in sorted(keys):
            out.append(Candidate(index.entries[k], 0 if net else 1, ("route_number",)))

    sn = split_street(text, ds.single_letter_suffixes)
    if not sn.base:
        return out

    def first(cands):
        return [c for c in cands if c.penalty <= max_pen]

    def score(name, keys, tier):
        return first(_score_entries(name, keys, index, tier, unreliable=ds.unreliable_types))

    def found(cands):
        # Looking past a changed type (whole_text_only), only a reading that keeps the type counts.
        return any(not whole_text_only or "type_changed" not in c.flags for c in cands)

    # Tier: exact base name.
    cands = [] if whole_text_only else score(sn, index.by_base.get(sn.base_key, set()), "exact")
    if cands:
        if any("type_changed" not in c.flags for c in cands):
            return out + cands
        # Only a changed type fits the name as written.  Readings of the whole text that keep
        # the record's type compete with it and rank first ("VISTAGLEN DR & BERONA WY" is Vista
        # Glen Drive, not Vistaglen Court), but the number or the cross street decides: "2700
        # BLUEBILL DR" stays on Bluebill Road when Blue Bill Drive has no 2700 block.  Readings
        # that drop words or need a typo do not compete.
        other = [c for c in _resolve_tiers(text, index, ds, truncated, wildcard_first, whole_text_only=True)
                 if not c.entry.is_ref and "type_changed" not in c.flags]
        return out + other + cands

    # Tier: spacing / punctuation ("VANDORN" = "VAN DORN", "EASTSHORE" = "EAST SHORE").
    keys = index.by_squash.get(sn.squash_key, set()) | index.by_squash_base.get(
        "".join(sn.raw[1:] if sn.predir else sn.raw), set())
    cands = score(sn, keys, "spacing")
    if found(cands):
        return out + cands
    # ... including a type word fused into the name ("IVANWAY ST" = "Ivan Way").
    toks = clean(text).split()
    if len(toks) >= 2 and suffix_options(toks[-1]):
        keys = index.by_prefix_form.get(prefix_form(" ".join(toks[:-1])), set())
        loose = StreetName(sn.predir, sn.base, frozenset(), sn.postdir, sn.raw)
        cands = score(loose, keys, "spacing")
        if found(cands):
            return out + cands
    # ... and a single token with the type fused on ("NIEMANBL" = "Nieman Boulevard").
    fused = _FUSED_TYPE.fullmatch(toks[-1]) if toks and (
        len(toks) == 1 or (len(toks) == 2 and toks[0] in DIRECTIONS)) else None
    if fused and not whole_text_only:
        cands = [Candidate(c.entry, c.penalty + 1, ("spacing",) + c.flags)
                 for c in resolve(" ".join(toks[:-1] + [fused.group(1), fused.group(2)]), index, ds)
                 if not WEAK_RELAXATIONS & set(c.flags)]
        if first(cands):
            return out + first(cands)

    # Tier: an extra trailing type ("UNIVERSITY AVE ST" -> "UNIVERSITY AVE").  The remaining
    # type must agree: "ELDERBERRY RIDGE LN" must not become "Elderberry Lane".
    if not whole_text_only and len(sn.base) >= 2 and suffix_options(sn.base[-1]):
        sn2 = StreetName(sn.predir, sn.base[:-1], suffix_options(sn.base[-1]), sn.postdir, sn.raw[:-1])
        cands = score(sn2, index.by_base.get(sn2.base_key, set()), "extra_type")
        cands = [c for c in cands if "type_changed" not in c.flags]
        if cands:
            return out + cands

    # Tier: the "type" is part of the name and the real type is missing
    # ("PINE LAKE" = "Pine Lake Road", "ROSA PARKS" = "Rosa Parks Way"), written apart or
    # fused ("CEDAR DALE" = "Cedardale Drive", "00 OAK RIDGE" = Oak Ridge Road or Oakridge
    # Path: the number decides).
    if sn.suffix:
        keys = set()
        for opt in sn.suffix:
            keys |= index.by_base.get(" ".join(sn.base + (opt,)), set())
        if not sn.postdir and toks and suffix_options(toks[-1]):
            squash = "".join(sn.raw[1:] if sn.predir else sn.raw) + toks[-1]
            keys |= {k for k in index.by_squash_base.get(squash, set())
                     if len(index.entries[k].name.base) <= len(sn.base)}
        loose = StreetName(sn.predir, sn.base, frozenset(), sn.postdir, sn.raw)
        cands = score(loose, keys, "type_as_name")
        if found(cands):
            return out + cands

    # Tier: words after the street type ("S 13TH ST ANTIQUES", "ELKCREST DR HEALTH G").
    type_pos = [i for i, t in enumerate(toks[1:-1], 1) if suffix_options(t)]
    if type_pos and not whole_text_only:
        k = type_pos[-1]
        tail = toks[k + 1:]
        if not any(suffix_options(t) or t in DIRECTIONS for t in tail):
            head = " ".join(toks[:k + 1])
            cands = [Candidate(c.entry, c.penalty + 1, ("trailing_text",) + c.flags)
                     for c in resolve(head, index, ds) if not WEAK_RELAXATIONS & set(c.flags)]
            if first(cands):
                return out + first(cands)

    # Tier: a town name after the street ("350 S ABEL IN MILPITAS", "950 CLYDE AV SANTA CL"):
    # a town in the box, written in full or cut off after 5+ letters, after a type or "IN".
    towns = getattr(index, "towns", ())
    if towns and not whole_text_only and len(toks) >= 3:
        for k in (2, 1):
            if len(toks) <= k + 1:
                continue
            tail, head = " ".join(toks[-k:]), toks[:-k]
            if not any(t == tail or (len(tail) >= 5 and t.startswith(tail)) for t in towns):
                continue
            if head[-1] == "IN":
                head = head[:-1]
            elif not suffix_options(head[-1]):
                continue
            cands = [Candidate(c.entry, c.penalty + 1, ("town_removed",) + c.flags)
                     for c in (resolve(" ".join(head), index, ds) if head else [])
                     if not WEAK_RELAXATIONS & set(c.flags)]
            if first(cands):
                return out + first(cands)
            break

    # Tier: a unit number after a name written without its type ("DWIGHT 417" = Dwight Way,
    # unit 417).  Only reached when the text as written matched nothing, so a name that ends
    # in a number ("PIER 2") is unaffected.
    if (not whole_text_only and len(toks) >= 2 and re.fullmatch(r"\d{1,5}[A-Z]?", toks[-1])
            and any(re.search(r"[A-Z]{2}", t) and t not in DIRECTIONS for t in toks[:-1])):
        cands = [Candidate(c.entry, c.penalty + 1, ("unit_removed",) + c.flags)
                 for c in resolve(" ".join(toks[:-1]), index, ds) if not WEAK_RELAXATIONS & set(c.flags)]
        if first(cands):
            return out + first(cands)

    # Tier: redacted first token (Berkeley "KAI00 AVE", "S00 PABLO AVE", "00 LOMA").
    if wildcard_first:
        cands = first(_resolve_wildcard(text, index, ds))
        if found(cands):
            return out + cands

    # Tier: field cut off at a fixed width (San Jose "SILVERCREEK VALL", "BIRKHAVEN P").
    if truncated:
        prefix = prefix_form(text)
        if len(prefix) >= 5:
            keys = index.prefix_lookup(prefix)
            # The cut may have removed the type or directional, so only conflicts count.
            loose = StreetName(sn.predir, sn.base, frozenset(), None, sn.raw)
            cands = score(loose, keys, "truncated")
            if found(cands):
                return out + cands

    # Typo tolerance, only for a name that does not exist here at all.  If the name exists
    # but was refused above (several types, a conflicting direction), a street one letter
    # away is not a better reading ("GORDON AV" must not become "Dordon Street"), and a typo
    # is never combined with a changed type.
    def typos():
        if whole_text_only or index.by_base.get(sn.base_key):
            return []
        keys, tier = _fuzzy_keys(sn, index, min_len)
        return [c for c in score(sn, keys, tier) if "type_changed" not in c.flags] if keys else []

    # Tier: the first word of the name was left out ("RANCHO VERDE" = "El Rancho Verde Drive",
    # "CINCO DE MAYO" = "Via Cinco de Mayo"), when exactly one such name exists in the area.
    # A typo reading competes, and the number or the cross street decides ("6900 BONNET DR"
    # is Bonnot Drive, which has the block, not Blue Bonnet Drive).
    bases = getattr(index, "by_tail", {}).get(sn.base_key, set())
    if not whole_text_only and len(bases) == 1:
        base = next(iter(bases))
        longer = StreetName(sn.predir, tuple(base.split()), sn.suffix, sn.postdir, sn.raw)
        cands = [c for c in score(longer, index.by_base.get(base, set()), "leading_word_added")
                 if "type_changed" not in c.flags]
        if cands:
            return out + cands + typos()

    # Tier: typo tolerance.
    cands = typos()
    if cands:
        return out + cands

    # Last resort: a bare 2-3 digit number with no such numbered street is a route ("87" = CA 87).
    if not out and not whole_text_only and re.fullmatch(r"\d{2,3}", text.strip()):
        for (net, num), ks in index.by_ref.items():
            if num == text.strip():
                out += [Candidate(index.entries[k], 2, ("bare_route_number",)) for k in sorted(ks)]
    return out


def _resolve_wildcard(text: str, index: AreaIndex, ds) -> list[Candidate]:
    toks = clean(text).split()
    if not toks:
        return []
    m = re.fullmatch(r"([A-Z0-9]*)00", toks[0])
    if not m:
        return []
    prefix, length = m.group(1), len(toks[0])
    out = []
    for word in sorted(index.first_tokens):
        if len(word) == length and word.startswith(prefix) and word != toks[0]:
            for c in resolve(" ".join([word] + toks[1:]), index, ds):
                if "fuzzy" not in "".join(c.flags) and "typo" not in "".join(c.flags):
                    out.append(Candidate(c.entry, c.penalty + TIER_PENALTY["redacted"],
                                         ("redacted",) + c.flags))
    return out
