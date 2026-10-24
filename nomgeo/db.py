"""Database access helpers."""
from __future__ import annotations

import psycopg


def connect(db: dict) -> psycopg.Connection:
    conn = psycopg.connect(host=db.get("host", "localhost"), port=db.get("port", 5432),
                           dbname=db.get("dbname", "nominatim"), user=db.get("user"),
                           password=db.get("password"), autocommit=True)
    return conn


def data_version(conn) -> str:
    """A fingerprint of the reference data; caches are discarded when it changes.

    Combines the OSM import/replication date with the identity of the TIGER table,
    which Nominatim recreates on every `add-data --tiger-data`.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT lastimportdate FROM import_status LIMIT 1")
        row = cur.fetchone()
        osm = row[0].isoformat() if row and row[0] else "unknown"
        cur.execute("SELECT relfilenode FROM pg_class WHERE relname = 'location_property_tiger'")
        row = cur.fetchone()
        tiger = str(row[0]) if row else "none"
    return f"osm:{osm}|tiger:{tiger}"
