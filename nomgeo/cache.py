"""A small key-value cache in SQLite, safe to use from several processes.

Two caches use it:
* results.sqlite   - final result per record, keyed by (address, config, code, data version);
                     a re-run with nothing changed is answered entirely from here.
* nominatim.sqlite - raw Nominatim responses, keyed by request parameters and data version;
                     survives config changes, so only new queries hit the server.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path


class KVCache:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._pid = None
        self._lock = threading.Lock()

    def _db(self) -> sqlite3.Connection:
        # One connection per process: connections must not cross a fork().
        if self._conn is None or self._pid != os.getpid():
            self._conn = sqlite3.connect(self.path, timeout=120, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, t REAL)")
            self._pid = os.getpid()
        return self._conn

    def get(self, key: str):
        with self._lock:
            row = self._db().execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_many(self, keys: list[str]) -> dict:
        out = {}
        with self._lock:
            db = self._db()
            for i in range(0, len(keys), 900):
                chunk = keys[i:i + 900]
                q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(chunk))})"
                for k, v in db.execute(q, chunk):
                    out[k] = json.loads(v)
        return out

    def put(self, key: str, value) -> None:
        self.put_many([(key, value)])

    def put_many(self, items) -> None:
        now = time.time()
        rows = [(k, json.dumps(v), now) for k, v in items]
        if not rows:
            return
        with self._lock:
            db = self._db()
            db.executemany("INSERT OR REPLACE INTO kv (k, v, t) VALUES (?, ?, ?)", rows)
            db.commit()

    def clear(self) -> None:
        with self._lock:
            db = self._db()
            db.execute("DELETE FROM kv")
            db.commit()
