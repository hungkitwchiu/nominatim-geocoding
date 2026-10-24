# nominatim-geocoding

Batch geocoding of US police incident addresses on a self-hosted
[Nominatim](https://nominatim.org/) database, built for research that maps incidents
to census geographies.

The geocoder is the `nomgeo` package. It matches each record against the streets,
address points and house-number ranges that exist in the police department's area, and
labels every result with what was matched, how precise it is, and how much interpretation
it needed (tier A, B or C). Records that are not addresses, or that fit more than one place,
stay uncoded with a stated reason.

Full documentation: [`docs/geocoding.qmd`](docs/geocoding.qmd) (rendered as `docs/geocoding.html`).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# one dataset (see configs/berkeley.yaml for the format)
.venv/bin/python -m nomgeo run configs/berkeley.yaml

# why did a record match (or not)?
.venv/bin/python -m nomgeo explain configs/berkeley.yaml "1300 9TH ST, BERKELEY, CA"
```

Outputs go to `outputs/<dataset>/`: `results.csv` (one row per input address),
`unmatched.csv` (sorted by reason) and `summary.json`.

## Layout

| Path | What it is |
|---|---|
| `nomgeo/` | the geocoder (parsing, street index, matching, caching, command line) |
| `configs/defaults.yaml` | thresholds and policies shared by all datasets |
| `configs/<city>.yaml`, `configs/rules/<city>.csv` | one dataset: area, parsing options, local name fixes |
| `tests/` | unit tests and regression cases (`python -m unittest discover -s tests -t .`) |
| `docs/` | documentation (Quarto) |
| `legacy/` | the earlier two-pass scripts and their documentation, kept for reference (tag `two-pass-final`) |

Address files and outputs are not version-controlled (see `.gitignore`).

## License

MIT, see [`LICENSE`](LICENSE).
