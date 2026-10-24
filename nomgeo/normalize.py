"""Text normalisation shared by the record parser and the street index.

Input records and OSM street names go through the *same* functions, so the two
compare equal whenever they differ only by case, punctuation, abbreviation
("AVE" vs "Avenue"), directional spelling ("N" vs "North") or ordinal spelling
("7TH" vs "Seventh").  Nothing here knows about any particular city; city-specific
fixes belong in the dataset rule files.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Street suffixes
# ---------------------------------------------------------------------------
# USPS Publication 28, Appendix C1: standard abbreviation -> accepted spellings.
_USPS_C1 = {
    "ALY": "ALLEY ALLEE ALLY ALY", "ANX": "ANEX ANNEX ANNX ANX", "ARC": "ARC ARCADE",
    "AVE": "AV AVE AVEN AVENU AVENUE AVN AVNUE", "BYU": "BAYOO BAYOU BYU",
    "BCH": "BCH BEACH", "BND": "BEND BND", "BLF": "BLF BLUF BLUFF", "BLFS": "BLUFFS BLFS",
    "BTM": "BOT BTM BOTTM BOTTOM", "BLVD": "BLVD BOUL BOULEVARD BOULV",
    "BR": "BR BRNCH BRANCH", "BRG": "BRDGE BRG BRIDGE", "BRK": "BRK BROOK", "BRKS": "BROOKS BRKS",
    "BG": "BURG BG", "BGS": "BURGS BGS", "BYP": "BYP BYPA BYPAS BYPASS BYPS",
    "CP": "CAMP CP CMP", "CYN": "CANYN CANYON CNYN CYN", "CPE": "CAPE CPE",
    "CSWY": "CAUSEWAY CAUSWA CSWY", "CTR": "CEN CENT CENTER CENTR CENTRE CNTER CNTR CTR",
    "CTRS": "CENTERS CTRS", "CIR": "CIR CIRC CIRCL CIRCLE CRCL CRCLE", "CIRS": "CIRCLES CIRS",
    "CLF": "CLF CLIFF", "CLFS": "CLFS CLIFFS", "CLB": "CLB CLUB", "CMN": "COMMON CMN",
    "CMNS": "COMMONS CMNS", "COR": "COR CORNER", "CORS": "CORNERS CORS", "CRSE": "COURSE CRSE",
    "CT": "COURT CT", "CTS": "COURTS CTS", "CV": "COVE CV", "CVS": "COVES CVS",
    "CRK": "CREEK CRK", "CRES": "CRESCENT CRES CRSENT CRSNT", "CRST": "CREST CRST",
    "XING": "CROSSING CRSSNG XING", "XRD": "CROSSROAD XRD", "XRDS": "CROSSROADS XRDS",
    "CURV": "CURVE CURV", "DL": "DALE DL", "DM": "DAM DM", "DV": "DIV DIVIDE DV DVD",
    "DR": "DR DRIV DRIVE DRV", "DRS": "DRIVES DRS", "EST": "EST ESTATE", "ESTS": "ESTATES ESTS",
    "EXPY": "EXP EXPR EXPRESS EXPRESSWAY EXPW EXPY", "EXT": "EXT EXTENSION EXTN EXTNSN EXTENDED",
    "EXTS": "EXTS EXTENSIONS", "FALL": "FALL", "FLS": "FALLS FLS", "FRY": "FERRY FRRY FRY",
    "FLD": "FIELD FLD", "FLDS": "FIELDS FLDS", "FLT": "FLAT FLT", "FLTS": "FLATS FLTS",
    "FRD": "FORD FRD", "FRDS": "FORDS FRDS", "FRST": "FOREST FORESTS FRST",
    "FRG": "FORG FORGE FRG", "FRGS": "FORGES FRGS", "FRK": "FORK FRK", "FRKS": "FORKS FRKS",
    "FT": "FORT FRT FT", "FWY": "FREEWAY FREEWY FRWAY FRWY FWY",
    "GDN": "GARDEN GARDN GRDEN GRDN GDN", "GDNS": "GARDENS GDNS GRDNS",
    "GTWY": "GATEWAY GATEWY GATWAY GTWAY GTWY", "GLN": "GLEN GLN", "GLNS": "GLENS GLNS",
    "GRN": "GREEN GRN", "GRNS": "GREENS GRNS", "GRV": "GROV GROVE GRV", "GRVS": "GROVES GRVS",
    "HBR": "HARB HARBOR HARBR HBR HRBOR", "HBRS": "HARBORS HBRS", "HVN": "HAVEN HVN",
    "HTS": "HT HTS HEIGHTS", "HWY": "HIGHWAY HIGHWY HIWAY HIWY HWAY HWY",
    "HL": "HILL HL", "HLS": "HILLS HLS", "HOLW": "HLLW HOLLOW HOLLOWS HOLW HOLWS",
    "INLT": "INLT INLET", "IS": "IS ISLAND ISLND", "ISS": "ISLANDS ISLNDS ISS", "ISLE": "ISLE ISLES",
    "JCT": "JCT JCTION JCTN JUNCTION JUNCTN JUNCTON", "JCTS": "JCTNS JCTS JUNCTIONS",
    "KY": "KEY KY", "KYS": "KEYS KYS", "KNL": "KNL KNOL KNOLL", "KNLS": "KNLS KNOLLS",
    "LK": "LK LAKE", "LKS": "LKS LAKES", "LAND": "LAND", "LNDG": "LANDING LNDG LNDNG",
    "LN": "LANE LN", "LGT": "LGT LIGHT", "LGTS": "LIGHTS LGTS", "LF": "LF LOAF",
    "LCK": "LCK LOCK", "LCKS": "LCKS LOCKS", "LDG": "LDG LDGE LODG LODGE", "LOOP": "LOOP LOOPS",
    "MALL": "MALL", "MNR": "MNR MANOR", "MNRS": "MANORS MNRS", "MDW": "MEADOW",
    "MDWS": "MDW MDWS MEADOWS MEDOWS", "MEWS": "MEWS", "ML": "MILL", "MLS": "MILLS",
    "MSN": "MISSN MSSN MISSION", "MTWY": "MOTORWAY MTWY", "MT": "MNT MT MOUNT",
    "MTN": "MNTAIN MNTN MOUNTAIN MOUNTIN MTIN MTN", "MTNS": "MNTNS MOUNTAINS MTNS",
    "NCK": "NCK NECK", "ORCH": "ORCH ORCHARD ORCHRD", "OVAL": "OVAL OVL", "OPAS": "OVERPASS OPAS",
    "PARK": "PARK PRK PARKS", "PKWY": "PARKWAY PARKWY PKWAY PKWY PKY PARKWAYS PKWYS",
    "PASS": "PASS", "PSGE": "PASSAGE PSGE", "PATH": "PATH PATHS", "PIKE": "PIKE PIKES",
    "PNE": "PINE PNE", "PNES": "PINES PNES", "PL": "PL PLACE", "PLN": "PLAIN PLN",
    "PLNS": "PLAINS PLNS", "PLZ": "PLAZA PLZ PLZA", "PT": "POINT PT", "PTS": "POINTS PTS",
    "PRT": "PORT PRT", "PRTS": "PORTS PRTS", "PR": "PR PRAIRIE PRR",
    "RADL": "RAD RADIAL RADIEL RADL", "RAMP": "RAMP", "RNCH": "RANCH RANCHES RNCH RNCHS",
    "RPD": "RAPID RPD", "RPDS": "RAPIDS RPDS", "RST": "REST RST", "RDG": "RDG RDGE RIDGE",
    "RDGS": "RDGS RIDGES", "RIV": "RIV RIVER RVR RIVR", "RD": "RD ROAD", "RDS": "ROADS RDS",
    "RTE": "ROUTE RTE", "ROW": "ROW", "RUE": "RUE", "RUN": "RUN", "SHL": "SHL SHOAL",
    "SHLS": "SHLS SHOALS", "SHR": "SHOAR SHORE SHR", "SHRS": "SHOARS SHORES SHRS",
    "SKWY": "SKYWAY SKWY", "SPG": "SPG SPNG SPRING SPRNG", "SPGS": "SPGS SPNGS SPRINGS SPRNGS",
    "SPUR": "SPUR SPURS", "SQ": "SQ SQR SQRE SQU SQUARE", "SQS": "SQRS SQUARES SQS",
    "STA": "STA STATION STATN STN", "STRA": "STRA STRAV STRAVEN STRAVENUE STRAVN STRVN STRVNUE",
    "STRM": "STREAM STREME STRM", "ST": "STREET STRT ST STR", "STS": "STREETS STS",
    "SMT": "SMT SUMIT SUMITT SUMMIT", "TER": "TER TERR TERRACE", "TRWY": "THROUGHWAY TRWY",
    "TRCE": "TRACE TRACES TRCE", "TRAK": "TRACK TRACKS TRAK TRK TRKS", "TRFY": "TRAFFICWAY TRFY",
    "TRL": "TRAIL TRAILS TRL TRLS", "TRLR": "TRAILER TRLR TRLRS",
    "TUNL": "TUNEL TUNL TUNLS TUNNEL TUNNELS TUNNL", "TPKE": "TRNPK TURNPIKE TURNPK TPKE",
    "UPAS": "UNDERPASS UPAS", "UN": "UN UNION", "UNS": "UNIONS UNS",
    "VLY": "VALLEY VALLY VLLY VLY", "VLYS": "VALLEYS VLYS", "VIA": "VDCT VIA VIADCT VIADUCT",
    "VW": "VIEW VW", "VWS": "VIEWS VWS", "VLG": "VILL VILLAG VILLAGE VILLG VILLIAGE VLG",
    "VLGS": "VILLAGES VLGS", "VL": "VILLE VL", "VIS": "VIS VIST VISTA VST VSTA",
    "WALK": "WALK WALKS", "WALL": "WALL", "WAY": "WY WAY", "WAYS": "WAYS", "WL": "WELL WL",
    "WLS": "WELLS WLS",
}

# Street-type words that occur in local OSM names but are not in USPS C1.  They
# are treated exactly like suffixes so "Kempsville Close" splits as name+type.
_EXTRA_TYPES = {
    "ARCH": "ARCH AR", "CHASE": "CHASE CHAS", "CLOSE": "CLOSE CLOS", "QUAY": "QUAY",
    "REACH": "REACH RECH", "DOWNS": "DOWNS DONS", "WALKWAY": "WALKWAY WW",
    "CONN": "CONNECTOR CONN", "ACCESS": "ACCESS", "STEPS": "STEPS STAIRS",
    "POINTE": "POINTE", "WYND": "WYND WYNDE", "GATE": "GATE", "TURN": "TURN",
    "ALCOVE": "ALCOVE", "CROSS": "CROSS", "GREENWAY": "GREENWAY", "BUSWAY": "BUSWAY",
    "RDWY": "ROADWAY RDWY",
}

# Abbreviations used by police records management systems.  A token may stand for
# several types ("CR" = Circle or Crescent); the area's actual streets decide.
_POLICE_TYPES = {
    "AV": {"AVE"}, "BL": {"BLVD"}, "BD": {"BLVD"}, "CI": {"CIR"}, "CR": {"CIR", "CRES"},
    "CL": {"CIR", "CLOSE"}, "CM": {"CMN", "CMNS"}, "CMMN": {"CMN"}, "COM": {"CMN"},
    "CY": {"CT"}, "EX": {"EXPY", "EXT"}, "FY": {"FWY"}, "HY": {"HWY"}, "LP": {"LOOP"},
    "LD": {"LNDG"}, "PA": {"PKWY", "PATH"}, "PW": {"PKWY"}, "PY": {"PKWY"}, "PK": {"PARK", "PIKE"},
    "PZ": {"PLZ"}, "TE": {"TER"}, "TR": {"TER", "TRL"}, "TRKE": {"TPKE"}, "TPK": {"TPKE"},
    "WK": {"WALK"}, "SQR": {"SQ"}, "CRT": {"CT"}, "DRV": {"DR"}, "LA": {"LN"},
    "RW": {"ROW"}, "HW": {"HWY"}, "AL": {"ALY"}, "BV": {"BLVD"}, "BLV": {"BLVD"}, "TL": {"TRL"}, "STRE": {"ST"},
}

# Single letters some departments use as types ("OLIVE L").  Only honoured when a
# dataset enables `single_letter_suffixes`, because Lincoln has an "L Street".
SINGLE_LETTER_TYPES = {
    "A": {"AVE"}, "B": {"BLVD"}, "C": {"CT", "CIR"}, "D": {"DR"}, "L": {"LN", "LOOP"},
    "P": {"PL", "PARK", "PKWY"}, "R": {"RD"}, "S": {"ST"}, "T": {"TER", "TRL"}, "W": {"WAY"},
}

SUFFIX_OPTIONS: dict[str, frozenset] = {}
for _std, _variants in list(_USPS_C1.items()) + list(_EXTRA_TYPES.items()):
    for _v in _variants.split():
        SUFFIX_OPTIONS.setdefault(_v, set()).add(_std)
for _tok, _stds in _POLICE_TYPES.items():
    SUFFIX_OPTIONS.setdefault(_tok, set()).update(_stds)
SUFFIX_OPTIONS = {k: frozenset(v) for k, v in SUFFIX_OPTIONS.items()}

# Words that end a street name but should not be read as a type in the *middle*
# of a name.  They still canonicalise inside the base (both sides alike).
CANONICAL_TOKEN = {tok: next(iter(stds)) for tok, stds in SUFFIX_OPTIONS.items() if len(stds) == 1}

# Types that say no more than "a road": a record can give the wrong one without meaning a
# different street.  Words that name a kind of place (SQ, PARK, BCH, BRG, HL, LNDG, XING ...)
# are not among them.  A record's type may be changed *to* a highway type, never *from* one.
ROAD_TYPES = frozenset({"ST", "STS", "AVE", "BLVD", "RD", "DR", "LN", "CT", "PL", "CIR", "CRES", "WAY",
                        "TER", "PKWY", "TRL", "LOOP", "ALY", "ROW", "WALK", "PATH", "CV", "RUN"})
HIGHWAY_TYPES = frozenset({"HWY", "FWY", "EXPY", "TPKE", "PIKE", "BYP"})

# Suffixes after which a bare number is an ordinal ("N 27 ST" -> "N 27TH ST").
_ORDINAL_CONTEXT = {"ST", "AVE", "PL", "TER", "CT", "DR", "LN", "WAY", "BLVD", "CIR", "RD",
                    "ALY", "TRL", "PKWY", "SQ", "LOOP", "VW", "PATH"}

# ---------------------------------------------------------------------------
# Directionals and ordinals
# ---------------------------------------------------------------------------
DIRECTIONS = {
    "N": "N", "NORTH": "N", "S": "S", "SOUTH": "S", "SO": "S", "E": "E", "EAST": "E",
    "W": "W", "WEST": "W", "NE": "NE", "NORTHEAST": "NE", "NW": "NW", "NORTHWEST": "NW",
    "SE": "SE", "SOUTHEAST": "SE", "SW": "SW", "SOUTHWEST": "SW",
}
DIRECTION_WORD = {"N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST",
                  "NE": "NORTHEAST", "NW": "NORTHWEST", "SE": "SOUTHEAST", "SW": "SOUTHWEST"}

_UNIT_ORD = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6,
             "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9}
_TEEN_ORD = {"TENTH": 10, "ELEVENTH": 11, "TWELFTH": 12, "THIRTEENTH": 13, "FOURTEENTH": 14,
             "FIFTEENTH": 15, "SIXTEENTH": 16, "SEVENTEENTH": 17, "EIGHTEENTH": 18,
             "NINETEENTH": 19}
_TENS_ORD = {"TWENTIETH": 20, "THIRTIETH": 30, "FORTIETH": 40, "FIFTIETH": 50, "SIXTIETH": 60,
             "SEVENTIETH": 70, "EIGHTIETH": 80, "NINETIETH": 90, "HUNDREDTH": 100}
_TENS = {"TWENTY": 20, "THIRTY": 30, "FORTY": 40, "FIFTY": 50, "SIXTY": 60, "SEVENTY": 70,
         "EIGHTY": 80, "NINETY": 90}

# Leading words that expand when they start a multi-word name ("ST CLAIR" -> SAINT CLAIR).
_LEADING_EXPANSIONS = {"ST": "SAINT", "STE": "SAINTE", "MT": "MOUNT", "FT": "FORT",
                       "PT": "POINT", "DR": "DOCTOR"}
# Honorific/abbreviation tokens unified on both sides.
_TOKEN_ALIASES = {"JUNIOR": "JR", "JNR": "JR", "SENIOR": "SR", "SAINT": "SAINT"}


def ordinal(n: int) -> str:
    """1 -> '1ST', 2 -> '2ND', 11 -> '11TH', 23 -> '23RD'."""
    if 10 <= n % 100 <= 20:
        return f"{n}TH"
    return f"{n}" + {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")


_NUM_ORD_RE = re.compile(r"^(\d+)(ST|ND|RD|TH)$")


def _ordinalise(tokens: list[str]) -> list[str]:
    """Spelled-out ordinals become numeric ('TWENTY FIRST' -> '21ST'); fix '3TH' -> '3RD'."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if t in _TENS and nxt in _UNIT_ORD:
            out.append(ordinal(_TENS[t] + _UNIT_ORD[nxt]))
            i += 2
            continue
        if t in _UNIT_ORD:
            out.append(ordinal(_UNIT_ORD[t]))
        elif t in _TEEN_ORD:
            out.append(ordinal(_TEEN_ORD[t]))
        elif t in _TENS_ORD:
            out.append(ordinal(_TENS_ORD[t]))
        else:
            m = _NUM_ORD_RE.match(t)
            out.append(ordinal(int(m.group(1))) if m else t)
        i += 1
    return out


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
_APOSTROPHES = re.compile(r"[’‘'`´]")
_NON_WORD = re.compile(r"[^A-Z0-9/& ]+")


def clean(text: str) -> str:
    """Upper-case, strip accents, drop apostrophes, turn other punctuation into spaces.

    '&' and '/' survive because the record parser uses them as connectors.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).upper()
    text = _APOSTROPHES.sub("", text)
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Street names
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StreetName:
    """A street name split into USPS-style components.

    `suffix` is a set because an input abbreviation can stand for several types;
    names read from OSM always carry exactly one (or none).
    """
    predir: str | None
    base: tuple[str, ...]
    suffix: frozenset = field(default_factory=frozenset)
    postdir: str | None = None
    raw: tuple[str, ...] = ()        # cleaned tokens before canonicalisation (for squash keys)

    @property
    def base_key(self) -> str:
        return " ".join(self.base)

    @property
    def squash_key(self) -> str:
        """Name without type and spaces, directionals spelled out ('EASTSHORE')."""
        return "".join(self.raw)

    def key(self, suffix: str | None = None) -> str:
        parts = [self.predir] if self.predir else []
        parts += list(self.base)
        sfx = suffix if suffix is not None else (next(iter(self.suffix)) if len(self.suffix) == 1 else None)
        if sfx:
            parts.append(sfx)
        if self.postdir:
            parts.append(self.postdir)
        return " ".join(parts)

    def __str__(self) -> str:
        sfx = "/".join(sorted(self.suffix)) if self.suffix else ""
        return " ".join(p for p in [self.predir or "", self.base_key, sfx, self.postdir or ""] if p)


def prefix_form(text: str) -> str:
    """Whole name squashed with abbreviations spelled out, for matching cut-off fields.

    'Mount Rushmore Drive' -> 'MOUNTRUSHMOREDRIVE', and the truncated record
    'MT RUSHMOR' -> 'MOUNTRUSHMOR', which is a prefix of it.
    """
    toks = _ordinalise(clean(text).split())
    if len(toks) >= 2 and toks[0] in _LEADING_EXPANSIONS:
        toks[0] = _LEADING_EXPANSIONS[toks[0]]
    return "".join(DIRECTION_WORD.get(t, t) for t in toks)


def suffix_options(token: str, single_letter: bool = False) -> frozenset:
    opts = SUFFIX_OPTIONS.get(token)
    if opts:
        return opts
    if single_letter and token in SINGLE_LETTER_TYPES:
        return frozenset(SINGLE_LETTER_TYPES[token])
    return frozenset()


def _collapse_repeated_suffix(toks: list[str]) -> list[str]:
    """'CEDAR STREET STREET' -> 'CEDAR STREET'; 'BERKLEY AVENUE EX AVENUE' -> 'BERKLEY AVENUE EX'."""
    while len(toks) >= 3:
        last = SUFFIX_OPTIONS.get(toks[-1])
        if not last:
            break
        prev = SUFFIX_OPTIONS.get(toks[-2])
        prev2 = SUFFIX_OPTIONS.get(toks[-3]) if len(toks) >= 4 else None
        if prev and prev & last:
            toks = toks[:-1]
        elif prev and prev2 and prev2 & last:
            toks = toks[:-1]
        else:
            break
    return toks


def _canonical_base(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Return (canonical base tokens, raw tokens for the squash key)."""
    toks = _ordinalise([_TOKEN_ALIASES.get(t, t) for t in tokens])
    # Half streets: "21ST 1/2" and OSM's "21 1/2" are the same street.
    toks = [re.sub(r"(ST|ND|RD|TH)$", "", t) if i + 1 < len(toks) and toks[i + 1] == "1/2"
            and _NUM_ORD_RE.match(t) else t for i, t in enumerate(toks)]
    # Expand MLK-style abbreviations to the long form used in OSM.
    expanded: list[str] = []
    for t in toks:
        if t == "MLK":
            expanded += ["MARTIN", "LUTHER", "KING"]
        else:
            expanded.append(t)
    toks = expanded
    # A generational suffix is often left out ("MLK WAY" = Martin Luther King Jr Way).
    toks = [t for i, t in enumerate(toks) if not (i >= 2 and t in ("JR", "SR"))]
    if len(toks) >= 2 and toks[0] in _LEADING_EXPANSIONS:
        toks = [_LEADING_EXPANSIONS[toks[0]]] + toks[1:]
    if len(toks) == 1 and toks[0].isdigit() and int(toks[0]) < 500:
        toks = [ordinal(int(toks[0]))]
    raw = [DIRECTION_WORD.get(t, t) for t in toks]
    canon = [CANONICAL_TOKEN.get(t, t) for t in toks]
    return canon, raw


def split_street(text_or_tokens, single_letter_suffixes: bool = False) -> StreetName:
    """Split a street name into (predirectional, base, type, postdirectional).

    Works the same for an input string ("N 27 ST") and an OSM name ("North 27th Street").
    A directional or type is only peeled off when a non-empty name remains, which keeps
    "N Street", "West Street" and "Avenue E" intact.
    """
    toks = clean(text_or_tokens).split() if isinstance(text_or_tokens, str) else list(text_or_tokens)
    toks = [t for t in toks if t not in ("&", "/")]
    if not toks:
        return StreetName(None, ())
    toks = _collapse_repeated_suffix(toks)
    pre = post = None
    sfx: frozenset = frozenset()
    i, j = 0, len(toks)
    if j - i >= 2 and toks[j - 1] in DIRECTIONS:
        rest = toks[i:j - 1]
        if not (len(rest) == 1 and suffix_options(rest[0], single_letter_suffixes)):
            post, j = DIRECTIONS[toks[j - 1]], j - 1
    if j - i >= 2:
        opts = suffix_options(toks[j - 1], single_letter_suffixes)
        if opts:
            sfx, j = opts, j - 1
    if j - i >= 2 and toks[i] in DIRECTIONS:
        pre, i = DIRECTIONS[toks[i]], i + 1
    base, raw = _canonical_base(toks[i:j])
    if pre:
        raw = [DIRECTION_WORD[pre]] + raw
    return StreetName(pre, tuple(base), sfx, post, tuple(raw))


# ---------------------------------------------------------------------------
# Route numbers ("I-80", "HWY 2", "US 101")
# ---------------------------------------------------------------------------
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY", "PR",
}
_ROUTE_RE = re.compile(
    r"^(?:(?P<net>I|IH|INTERSTATE|US|USHWY|SR|STATE ROUTE|STATE HWY|STATE HIGHWAY|RT|RTE|ROUTE|"
    r"HWY|HY|HIGHWAY|CR|COUNTY ROAD|COUNTY RD|[A-Z]{2})\s*(?:HWY|HIGHWAY|ROUTE|RTE)?\s*)"
    r"(?P<num>\d{1,4})(?:\s*(?P<dir>[NSEW]|NB|SB|EB|WB|NORTH|SOUTH|EAST|WEST))?$")


def parse_route(text: str, state: str | None = None) -> tuple[str | None, str] | None:
    """'I-80' -> ('I', '80'); 'HWY 2' -> (None, '2'); 'CA 87' -> ('SR', '87'); else None."""
    t = clean(text).replace("-", " ")
    t = re.sub(r"^I(\d)", r"I \1", t)
    t = re.sub(r"^U S\b", "US", t)
    m = _ROUTE_RE.match(t)
    if not m:
        return None
    net = m.group("net").strip()
    num = m.group("num").lstrip("0") or "0"
    if net in ("I", "IH", "INTERSTATE"):
        return ("I", num)
    if net in ("US", "USHWY"):
        return ("US", num)
    if net in ("SR", "STATE ROUTE", "STATE HWY", "STATE HIGHWAY") or (state and net == state):
        return ("SR", num)
    if net in ("CR", "COUNTY ROAD", "COUNTY RD"):
        return ("CR", num)
    if net in ("RT", "RTE", "ROUTE", "HWY", "HY", "HIGHWAY"):
        return (None, num)
    return None


def parse_osm_ref(ref: str) -> list[tuple[str, str]]:
    """OSM `ref` ('I 80;I 580', 'CA 87', 'US 101') -> [('I','80'), ('I','580')]."""
    out = []
    for part in ref.split(";"):
        p = clean(part).replace("-", " ")
        m = re.match(r"^([A-Z]+)\s*(\d{1,4})", p)
        if not m:
            continue
        net, num = m.groups()
        if net in ("I", "US", "SR", "CR"):
            out.append((net, num))
        elif net in US_STATES:
            out.append(("SR", num))
    return out
