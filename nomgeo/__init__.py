"""nomgeo: reference-driven batch geocoding of US addresses on a local Nominatim database.

The package matches each record against the streets, address points and
house-number ranges stored in Nominatim's own tables, and labels every result
with what was matched and how precise it is.  See docs/geocoding.qmd.
"""

__version__ = "0.1.0"
