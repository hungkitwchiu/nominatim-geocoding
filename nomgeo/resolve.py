"""Resolve a street string against the streets that exist in the area.

Matching proceeds through tiers and stops at the first tier that finds anything,
so a looser rule is only tried when every stricter one failed:

  exact name  ->  spacing ("VANDORN" = "Van Dorn")  ->  extra trailing type
  ->  redaction wildcard ("KAI00" = "Kains")  ->  truncated field ("SILVERCREEK VALL")
  ->  typo tolerance (1-2 edits, unique, digits must agree)

Within a tier, directional and street-type differences are scored, and only the
least-relaxed variants survive (an exact "MAIN ST" hides "MAIN AVE").  Every
relaxation is recorded as a flag and adds to a penalty so results can be audited.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import process
from rapidfuzz.distance import OSA

from .normalize import (DIRECTIONS, StreetName, clean, parse_route, prefix_form, split_street,
                        suffix_options)
from .streets import AreaIndex, StreetEntry

TIER_PENALTY = {"exact": 0, "spacing": 1, "extra_type": 1, "type_as_name": 1, "redacted": 2,
                "truncated": 2, "fuzzy1": 3, "fuzzy2": 5}
# Relaxations not to be stacked on top of another one.
WEAK_RELAXATIONS = {"type_changed", "typo_1", "typo_2", "truncated", "redacted"}


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
        # Only a changed type is left.  That is a fair guess when the name has one type in
        # the area ("FONTANOSO RD" = Fontanoso Way; "CORNHUSKER RD" = Cornhusker Highway and
        # its West part) but not when it has several ("PARK AV" with no Park Ave but Park
        # Street, Park Terrace, Park Circle ...).
        return []
    out = []
    for e, dpen, spen, flags in scored:
        pen = TIER_PENALTY[tier] + dpen + spen + (1 if e.alt else 0)
        fl = tuple(extra_flags) + flags + (("alt_name",) if e.alt else ())
        if tier != "exact":
            fl = (tier if not tier.startswith("fuzzy") else "typo_" + tier[-1],) + fl
        out.append(Candidate(e, pen, fl))
    return out


def _fuzzy_keys(sn: StreetName, index: AreaIndex, min_len: int):
    q = sn.base_key
    letters = q.replace(" ", "")
    if len(letters) < min_len:
        return [], None
    max_d = 1 if len(letters) < 9 else 2
    hits = process.extract(q, index.base_choices(), scorer=OSA.distance, score_cutoff=max_d,
                           limit=25)
    digits = re.findall(r"\d+", q)
    hits = [(c, d) for c, d, _ in hits if d > 0 and re.findall(r"\d+", c) == digits]
    if not hits:
        return [], None
    best = min(d for _, d in hits)
    keys = set()
    for c, d in hits:
        if d == best:
            keys |= index.by_base[c]
    return keys, f"fuzzy{best}"


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


def _resolve_tiers(text: str, index: AreaIndex, ds, truncated: bool = False,
                   wildcard_first: bool = False) -> list[Candidate]:
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

    # Tier: exact base name.
    cands = score(sn, index.by_base.get(sn.base_key, set()), "exact")
    if cands:
        return out + cands

    # Tier: spacing / punctuation ("VANDORN" = "VAN DORN", "EASTSHORE" = "EAST SHORE").
    keys = index.by_squash.get(sn.squash_key, set()) | index.by_squash_base.get(
        "".join(sn.raw[1:] if sn.predir else sn.raw), set())
    cands = score(sn, keys, "spacing")
    if cands:
        return out + cands
    # ... including a type word fused into the name ("IVANWAY ST" = "Ivan Way").
    toks = clean(text).split()
    if len(toks) >= 2 and suffix_options(toks[-1]):
        keys = index.by_prefix_form.get(prefix_form(" ".join(toks[:-1])), set())
        loose = StreetName(sn.predir, sn.base, frozenset(), sn.postdir, sn.raw)
        cands = score(loose, keys, "spacing")
        if cands:
            return out + cands

    # Tier: an extra trailing type ("UNIVERSITY AVE ST" -> "UNIVERSITY AVE").  The remaining
    # type must agree: "ELDERBERRY RIDGE LN" must not become "Elderberry Lane".
    if len(sn.base) >= 2 and suffix_options(sn.base[-1]):
        sn2 = StreetName(sn.predir, sn.base[:-1], suffix_options(sn.base[-1]), sn.postdir, sn.raw[:-1])
        cands = score(sn2, index.by_base.get(sn2.base_key, set()), "extra_type")
        cands = [c for c in cands if "type_changed" not in c.flags]
        if cands:
            return out + cands

    # Tier: the "type" is part of the name and the real type is missing
    # ("PINE LAKE" = "Pine Lake Road", "ROSA PARKS" = "Rosa Parks Way").
    if sn.suffix:
        keys = set()
        for opt in sn.suffix:
            keys |= index.by_base.get(" ".join(sn.base + (opt,)), set())
        loose = StreetName(sn.predir, sn.base, frozenset(), sn.postdir, sn.raw)
        cands = score(loose, keys, "type_as_name")
        if cands:
            return out + cands

    # Tier: words after the street type ("S 13TH ST ANTIQUES", "ELKCREST DR HEALTH G").
    type_pos = [i for i, t in enumerate(toks[1:-1], 1) if suffix_options(t)]
    if type_pos:
        k = type_pos[-1]
        tail = toks[k + 1:]
        if not any(suffix_options(t) or t in DIRECTIONS for t in tail):
            head = " ".join(toks[:k + 1])
            cands = [Candidate(c.entry, c.penalty + 1, ("trailing_text",) + c.flags)
                     for c in resolve(head, index, ds) if not WEAK_RELAXATIONS & set(c.flags)]
            if first(cands):
                return out + first(cands)

    # Tier: redacted first token (Berkeley "KAI00 AVE", "S00 PABLO AVE", "00 LOMA").
    if wildcard_first:
        cands = _resolve_wildcard(text, index, ds)
        if cands:
            return out + first(cands)

    # Tier: field cut off at a fixed width (San Jose "SILVERCREEK VALL", "BIRKHAVEN P").
    if truncated:
        prefix = prefix_form(text)
        if len(prefix) >= 5:
            keys = index.prefix_lookup(prefix)
            # The cut may have removed the type or directional, so only conflicts count.
            loose = StreetName(sn.predir, sn.base, frozenset(), None, sn.raw)
            cands = score(loose, keys, "truncated")
            if cands:
                return out + cands

    # Tier: typo tolerance.
    keys, tier = _fuzzy_keys(sn, index, min_len)
    if keys:
        cands = score(sn, keys, tier)
        if cands:
            return out + cands

    # Last resort: a bare 2-3 digit number with no such numbered street is a route ("87" = CA 87).
    if not out and re.fullmatch(r"\d{2,3}", text.strip()):
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
