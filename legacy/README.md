# The earlier two-pass scripts

These files are the first version of this project, kept for reference and for the
comparisons in the [current documentation](../docs/geocoding.qmd). They are superseded by
the `nomgeo` package and are no longer maintained.

| File | What it is |
|---|---|
| `geocode.py`, `geocode_pass2.py`, `geofunctions.py` | the two-pass geocoder (Nominatim queries, then intersections from an `osm_raw` osm2pgsql database) |
| `name_cleanup_rules.csv`, `city_viewboxes.csv` | its name rules and search boxes; the city-specific rows now live in `configs/rules/` and `configs/*.yaml` |
| `extractcol.py`, `countsuffix.py` | helpers for re-running unmatched addresses and finding unhandled suffixes |
| `doc.geocode.qmd` | its documentation, published at <https://hungkitwchiu.github.io/doc.geocode.html> |

The scripts read their rule files and address CSVs from the working directory, so run
them from this folder. The last version before `nomgeo` is tagged `two-pass-final`.
