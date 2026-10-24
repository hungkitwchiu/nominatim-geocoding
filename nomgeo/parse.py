"""Turn one raw record ("1300 9TH ST, BERKELEY, CA") into a structured request.

The parser only *classifies and cleans*; it never decides which street is meant.
That happens in resolve.py against the streets that actually exist in the area.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalize import DIRECTIONS as _DIRECTIONS
from .normalize import US_STATES, clean, ordinal, suffix_options

# Record kinds -------------------------------------------------------------
ADDRESS = "address"            # house/block number + one street
STREET = "street"              # one street, no number
INTERSECTION = "intersection"  # two (or more) streets
BETWEEN = "between"            # Lincoln "26TH & K-N ST": 26th St between K St and N St
HIGHWAY = "highway"            # ramp, direction of travel, interstate mile marker...
JUNK = "junk"                  # placeholder text ("UNKNOWN")
EMPTY = "empty"                # nothing but a city/number
OTHER_PLACE = "other_place"    # names another city/state
COORDINATES = "coordinates"    # lat/lon typed into the address field


@dataclass
class Record:
    raw: str
    street_text: str = ""
    city: str | None = None
    state: str | None = None
    kind: str = EMPTY
    number: int | None = None
    streets: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    note: str | None = None
    truncated: bool = False            # last street may be cut at the dataset's field width
    wildcard_first: bool = False       # first token of first street is redacted ("KAI00")
    single_alternative: str | None = None  # "ROUGH AND READY RD": try as one street too
    redacted_alternative: str | None = None  # "00 LOMA": 0-block of Loma, or "LA LOMA"?
    coords: tuple[float, float] | None = None


_JUNK_RE = re.compile(
    r"^(UNKNOWN|UNK|NONE|NA|N A|TEST|VARIOUS|MISC|CITYWIDE|CITY WIDE|NO ADDRESS)\b|"
    r"\b(UNKNOWN|NOT A VALID|MISCELLANEOUS|LOCATION UNKNOWN|UNK LOCATION|"
    r"COMMUNITY AT LARGE|CITY LINC|CITY LINCOLN|UNKNOWN LOCATION|NO LOCATION)\b")
_COORD_RE = re.compile(r"(-?\d{2,3}\.\d{4,})\s*[;, ]\s*(-?\d{2,3}\.\d{4,})")
_RAMP_RE = re.compile(r"\b(RAMP|RP|ONRAMP|OFFRAMP|OFFI|EXIT|OVERPASS|INTERCHANGE|HOV)\b|"
                      r"\b(ON|OFF) RAMP\b|\b\d{2,3}[NSEW]\b|\bI\s?\d+[NSEW]\b|"
                      r" TO | FROM ")
_TRAVEL_DIR_RE = re.compile(r"\b(NB|SB|EB|WB)\b")
_INTERSTATE_RE = re.compile(r"(^|[\s&])(I\s?0?\d{1,3}|INTERSTATE\s?\d{1,3})\b")
_UNIT_PATTERNS = [
    r"\(.*?\)", r"\(.*$", r"#\s*[A-Z0-9-]*",
    r"\b(APT|APARTMENT|UNIT|STE|SUITE|SPC|SPACE|RM|ROOM|BLDG|BUILDING|TRLR)\b\.?\s*[A-Z0-9-]*",
    r"\b\d+(ST|ND|RD|TH)\s+(FL|FLR|FLOOR)\b", r"\b(FLOOR|FLR)\s*\d+\b",
    r"\b(REAR|BLOCK|BLK)\b", r"\s-\s.*$", r"\b[A-Z0-9]{1,3}\s+(FLOOR|FLR)\b",
]
_UNIT_RE = re.compile("|".join(f"(?:{p})" for p in _UNIT_PATTERNS))
_DEAD_END = {"DEAD END", "END", "DE00 END", "DEAD-END"}
# Types that are followed by a route number rather than a unit ("COUNTY RD 12", "HWY 2").
_ROUTE_TYPES = {"RD", "ROAD", "HWY", "HY", "HIGHWAY", "RTE", "ROUTE", "FWY", "FREEWAY", "EXPY",
                "PIKE", "TPKE", "BYP", "CR", "SR", "US"}
_NUMBER_RE = re.compile(r"^(\d+)(?:\s*-\s*\d+)?[A-Z]?\s+(?:1/2\s+)?(?=[^&\s])")


def _number_is_name(rest_tokens: list[str]) -> bool:
    """True when a leading number is the street's name: "7 ST E" is Seventh Street East.

    That is the case when only one street type (and maybe a direction) follows the
    number; "100 PARK AVE" and "500 E ST" are ordinary house numbers.
    """
    return (len(rest_tokens) in (1, 2) and bool(suffix_options(rest_tokens[0]))
            and (len(rest_tokens) == 1 or rest_tokens[1] in _DIRECTIONS))


_HYPHEN_PAIR = re.compile(r"^\s*([A-Z0-9][A-Z0-9 .]*?)\s*-\s*([A-Z0-9][A-Z0-9 .]*?)\s*$")


def _between(text: str) -> list[str] | None:
    """'26TH & K-N ST' -> ['26TH', 'K ST', 'N ST']; None when the record is not of that form."""
    parts = [p.strip() for p in text.split("&")]
    if len(parts) != 2:
        return None
    hits = [_HYPHEN_PAIR.match(p) for p in parts]
    if sum(1 for h in hits if h) != 1:
        return None
    i = 0 if hits[0] else 1
    a, b = (clean(g) for g in hits[i].groups())
    main = clean(re.sub(r"^\d+\s+", "", parts[1 - i]))
    if not (a and b and main):
        return None
    if re.fullmatch(r"I|US|SR|[A-Z]{2}", a) and b.split()[0].isdigit():
        return None                 # a route number such as "I-80" or "KY-841", not a pair
    # "N-O ST": the type written once applies to both.
    a_toks, b_toks = a.split(), b.split()
    if len(b_toks) >= 2 and suffix_options(b_toks[-1]) and not suffix_options(a_toks[-1]):
        a = f"{a} {b_toks[-1]}"
    return [main, a, b]


def _split_city_state(raw: str, default_state: str | None):
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) >= 3:
        return ", ".join(parts[:-2]), parts[-2] or None, parts[-1] or default_state
    if len(parts) == 2:
        return parts[0], parts[1] or None, default_state
    return raw.strip(), None, default_state


def apply_rules(text: str, rules) -> str:
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def parse_record(raw: str, ds, redacted_name: bool = False) -> Record:
    """Classify and clean one record.  `ds` is the dataset config (see config.py).

    `redacted_name=True` reads a leading "00" as a redacted name ("00 LOMA" = "LA LOMA")
    instead of the default reading as a redacted 1-2 digit house number (0-99 block).
    """
    rec = Record(raw=raw)
    street, rec.city, rec.state = _split_city_state(raw or "", ds.state)
    if rec.state:
        rec.state = rec.state.strip().upper()

    # Extra comma-separated pieces naming a state ("GRETNA, NE, Lincoln, NE").
    pieces = [p.strip().upper() for p in street.split(",")]
    if len(pieces) > 1 and any(p in US_STATES or len(p) > 2 and p.split()[-1] in US_STATES
                               for p in pieces[1:]):
        rec.kind, rec.note = OTHER_PLACE, "names another place"
        return rec
    street = " ".join(pieces)

    street = apply_rules(street, ds.rules)
    for delim in ds.strip_after:
        if delim in street:
            street = street.split(delim, 1)[0]
            rec.flags.append("note_removed")
    m = _COORD_RE.search(street)
    if m:
        rec.kind, rec.note = COORDINATES, "coordinates in text"
        rec.coords = (float(m.group(1)), float(m.group(2)))
        return rec

    text = street.upper()
    text = re.sub(r"\s*//\s*", " & ", text)
    text = re.sub(r"(?<!\d)/(?!\d)", " & ", text)      # "A ST/B ST" but not "42 1/2"
    text = re.sub(r"^\s*@\s*", "", text)
    text = re.sub(r"\s@\s?|\s?@\s", " & ", text)
    text = re.sub(r"\s+AT\s+", " & ", text)
    unit_free = _UNIT_RE.sub(" ", text)
    if unit_free.split() != text.split():
        rec.flags.append("unit_removed")
    between = _between(unit_free) if ds.between_hyphen else None
    text = clean(unit_free)
    rec.street_text = text
    rec.truncated = bool(ds.truncation_length) and len(street) >= ds.truncation_length - 1

    if not text or re.fullmatch(r"[\d\s&]*", text):
        rec.kind, rec.note = EMPTY, "no street name"
        return rec
    if _JUNK_RE.search(text):
        rec.kind, rec.note = JUNK, "placeholder text"
        return rec
    if (re.search(r"\b(FL|TX|IA|KS|MO|CO|AZ|NV|GUAT|MEXICO)$", text)
            and not suffix_options(text.split()[-1])):
        rec.kind, rec.note = OTHER_PLACE, "names another state"
        return rec
    if _RAMP_RE.search(text):
        rec.kind, rec.note = HIGHWAY, "ramp or direction of travel"
        return rec
    if between:
        rec.kind, rec.streets = BETWEEN, between
        return rec
    if _TRAVEL_DIR_RE.search(text):
        # "CORNHUSKER HWY EB & N 23RD ST": the direction qualifies a lane, not the place.
        text = " ".join(_TRAVEL_DIR_RE.sub(" ", text).split())
        rec.flags.append("travel_direction_removed")
        if not text.replace("&", "").strip():
            rec.kind, rec.note = HIGHWAY, "direction of travel only"
            return rec

    # San Jose "VINE ST S & O GOODYEAR ST" = on Vine St, south of Goodyear St.
    m = re.match(r"^(.+?) (N|S|E|W) & O (.+)$", text)
    if m:
        text = f"{m.group(1)} & {m.group(3)}"
        rec.flags.append("offset_of_intersection")

    has_and = " AND " in f" {text} "
    if has_and:
        rec.single_alternative = text
        text = re.sub(r"\s+AND\s+", " & ", text)

    # Berkeley-style redaction: the first token had its last two characters replaced by
    # "00", meant for house numbers ("1234" -> "1200") but also hit names and ordinals.
    if ds.redacted_first_token:
        first, *rest = text.split()
        nxt = rest[0] if rest else ""
        if re.fullmatch(r"[1-9]\d*00", first) and (not rest or nxt == "&" or
                                                   (suffix_options(nxt) and (len(rest) == 1 or rest[1] == "&"))):
            text = " ".join([ordinal(int(first[:-2]))] + rest)
            rec.flags.append("redacted_ordinal")
        elif first == "00" and rest and not redacted_name:
            # A one- or two-digit house number redacted to "00": the 0-99 block.
            text = " ".join(["0"] + rest)
            rec.flags.append("redacted_zero_block")
            rec.redacted_alternative = raw
        elif re.fullmatch(r"[A-Z0-9]*[A-Z][A-Z0-9]*00|00", first):
            rec.wildcard_first = True
            rec.flags.append("redacted_name")

    # Lincoln "000000 4800 DORIS BAIR CR": a zero padding field in front of the real number.
    text = re.sub(r"^0+\s+(?=\d+\s)", "", text)
    number = None
    m = _NUMBER_RE.match(text)
    if m and not rec.wildcard_first:
        if not _number_is_name(text[m.end():].split("&")[0].split()):
            number = int(m.group(1))
            text = text[m.end():]

    parts = [p.strip() for p in text.split("&")]
    parts = [p for p in parts if p]
    if len(parts) > 1 and any(p in _DEAD_END for p in parts):
        parts = [p for p in parts if p not in _DEAD_END]
        rec.flags.append("dead_end")
    # Strip house numbers glued to later parts ("8100 DIVEN ARCH AND 8100 NORFOLK CROSSING"),
    # but keep numbered street names ("ROBERT ST & 7 ST E").
    parts = [p if _number_is_name(p.split()[1:]) else re.sub(r"^\d+\s+(?=\D)", "", p) for p in parts]
    # Unit numbers written after the street type ("UNIVERSITY AVE 211", "MLK JR WAY 2E").
    stripped = []
    for p in parts:
        toks = p.split()
        unit_number = re.fullmatch(r"\d{1,5}[A-Z]?|[A-Z]\d{1,4}", toks[-1]) if toks else None
        # A lone letter after the type ("HARWOOD CT G") is a unit, unless the dataset uses
        # single letters as types or it is a directional ("UNIVERSITY AVE E").
        unit_letter = (re.fullmatch(r"[A-Z]", toks[-1]) and not ds.single_letter_suffixes
                       and toks[-1] not in _DIRECTIONS) if toks else None
        if (len(toks) >= 3 and (unit_number or unit_letter)
                and suffix_options(toks[-2]) and toks[-2] not in _ROUTE_TYPES):
            p = " ".join(toks[:-1])
            if "unit_removed" not in rec.flags:
                rec.flags.append("unit_removed")
        stripped.append(p)
    parts = stripped
    if len(parts) >= 2 and len(set(parts)) == 1:
        parts = parts[:1]
        rec.flags.append("same_street_twice")

    interstate = any(_INTERSTATE_RE.search(f" {p}") for p in parts)
    if interstate:
        rec.kind = HIGHWAY
        rec.note = ("interchange" if len(parts) > 1 else
                    "interstate mile marker" if number is not None else "interstate only")
        rec.streets = parts
        return rec

    rec.streets = parts
    rec.number = number
    if not parts:
        rec.kind, rec.note = EMPTY, "no street name"
    elif len(parts) >= 2:
        rec.kind = INTERSECTION
        if number is not None:
            rec.flags.append("number_with_intersection")
    elif number is not None:
        rec.kind = ADDRESS
    else:
        rec.kind = STREET
    return rec
