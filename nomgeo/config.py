"""Dataset configuration.

A dataset is described by a small YAML file (see configs/*.yaml).  Anything not
set there falls back to configs/defaults.yaml, so the thresholds that decide
what counts as a match live in one documented place.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULTS_FILE = PROJECT_DIR / "configs" / "defaults.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_rules(path: Path | None) -> tuple[list, str]:
    """Read a rule CSV (columns match, replace, notes) into compiled substitutions.

    Same semantics as the original name_cleanup_rules.csv: case-insensitive, whole
    words unless the notes contain "no-boundary", and `match` may be a regex.
    """
    if not path:
        return [], ""
    text = Path(path).read_text(encoding="utf-8")
    rules = []
    for row in csv.DictReader(text.splitlines()):
        match = (row.get("match") or "").strip()
        if not match or match.startswith("#"):
            continue
        replace = (row.get("replace") or "").strip()
        notes = (row.get("notes") or "").lower()
        pattern = match if "no-boundary" in notes else rf"(?<![A-Za-z0-9]){match}(?![A-Za-z0-9])"
        rules.append((re.compile(pattern, re.IGNORECASE), f" {replace} " if replace else " "))
    return rules, text


@dataclass
class Dataset:
    name: str
    input: Path
    address_column: str
    city: str | None
    state: str | None
    bbox: list[float] | None
    buffer_km: float
    numbering: str
    rules: list
    rules_text: str
    strip_after: list[str]
    truncation_length: int | None
    redacted_first_token: bool
    single_letter_suffixes: bool
    between_hyphen: bool
    unreliable_types: frozenset
    precision: dict
    accept: dict
    database: dict
    nominatim_url: str
    cache_dir: Path
    output_dir: Path
    legacy: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        """Changes whenever a setting or rule that can change a result changes."""
        relevant = {k: self.raw.get(k) for k in ("area", "numbering", "parsing", "precision", "accept")}
        blob = json.dumps(relevant, sort_keys=True, default=str) + self.rules_text
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    defaults = yaml.safe_load(DEFAULTS_FILE.read_text()) if DEFAULTS_FILE.exists() else {}
    cfg = _merge(defaults, yaml.safe_load(path.read_text()) or {})
    base = path.parent

    def resolve(p):
        if p is None:
            return None
        p = Path(os.path.expanduser(str(p)))
        if p.is_absolute():
            return p
        # Relative paths in a dataset file are relative to the project directory.
        return (PROJECT_DIR / p) if (PROJECT_DIR / p).exists() or not (base / p).exists() else base / p

    area = cfg.get("area", {})
    parsing = cfg.get("parsing", {})
    bbox = area.get("bbox")
    if bbox:
        # Accept the corners in any order; store as left, top, right, bottom.
        xs, ys = sorted(map(float, bbox[0::2])), sorted(map(float, bbox[1::2]))
        bbox = [xs[0], ys[1], xs[1], ys[0]]
    rules, rules_text = load_rules(resolve(cfg.get("rules")))
    db = dict(cfg.get("database", {}))
    for key in ("host", "port", "dbname", "user", "password"):
        env = os.environ.get(f"NOMGEO_DB_{key.upper()}")
        if env:
            db[key] = env
    return Dataset(
        name=cfg["name"],
        input=resolve(cfg.get("input")),
        address_column=cfg.get("address_column", "address"),
        city=area.get("city"),
        state=(area.get("state") or "").upper() or None,
        bbox=bbox,
        buffer_km=float(area.get("buffer_km", 3)),
        numbering=cfg.get("numbering", "auto"),
        rules=rules,
        rules_text=rules_text,
        strip_after=list(parsing.get("strip_after") or []),
        truncation_length=parsing.get("truncation_length"),
        redacted_first_token=bool(parsing.get("redacted_first_token", False)),
        single_letter_suffixes=bool(parsing.get("single_letter_suffixes", False)),
        between_hyphen=bool(parsing.get("between_hyphen", False)),
        unreliable_types=frozenset(t.upper() for t in parsing.get("unreliable_types") or []),
        precision=cfg.get("precision", {}),
        accept=cfg.get("accept", {}),
        database=db,
        nominatim_url=os.environ.get("NOMGEO_NOMINATIM_URL", cfg.get("nominatim_url", "http://localhost/nominatim")),
        cache_dir=resolve(cfg.get("cache_dir", "cache")),
        output_dir=resolve(cfg.get("output_dir", "outputs")),
        legacy=cfg.get("legacy", {}) or {},
        raw=cfg,
    )
