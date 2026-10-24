"""End-to-end regression cases against the live database (tests/regression_cases.csv).

Each row records a case that was once wrong, with the expected outcome.  Skipped when
the database is not reachable.  Run with:  .venv/bin/python -m unittest tests.test_regression -v
"""
import csv
import unittest
from pathlib import Path

from nomgeo.area import load_area
from nomgeo.cache import KVCache
from nomgeo.config import PROJECT_DIR, load_dataset
from nomgeo.db import connect, data_version
from nomgeo.nominatim import NominatimClient
from nomgeo.parse import parse_record
from nomgeo.pipeline import detect_numbering, geocode_record, read_addresses

CASES = Path(__file__).with_name("regression_cases.csv")


class Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CASES.open(encoding="utf-8") as fh:
            cls.rows = list(csv.DictReader(fh))
        cls.env = {}
        try:
            first = load_dataset(PROJECT_DIR / "configs" / f"{cls.rows[0]['dataset']}.yaml")
            connect(first.database).close()
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"database not reachable: {exc}")

    def _env(self, name):
        if name not in self.env:
            ds = load_dataset(PROJECT_DIR / "configs" / f"{name}.yaml")
            conn = connect(ds.database)
            dv = data_version(conn)
            index, _ = load_area(conn, ds, dv)
            conn.close()
            mode = ds.numbering if ds.numbering != "auto" else detect_numbering(
                [parse_record(a, ds) for a in read_addresses(ds)])
            client = NominatimClient(ds.nominatim_url, KVCache(Path(ds.cache_dir) / "nominatim.sqlite"), dv)
            self.env[name] = (ds, index, mode, client)
        return self.env[name]

    def test_cases(self):
        for case in self.rows:
            with self.subTest(case["address"]):
                ds, index, mode, client = self._env(case["dataset"])
                row = geocode_record(case["address"], index, ds, mode, client)
                if case["expect_status"]:
                    self.assertEqual(row["status"], case["expect_status"], f"{case['why']} -> {row}")
                if case["expect_type"]:
                    self.assertEqual(row.get("match_type"), case["expect_type"], f"{case['why']} -> {row}")
                if case["expect_name"]:
                    self.assertIn(case["expect_name"], row.get("matched", ""), f"{case['why']} -> {row}")
                if case.get("forbid_name"):
                    self.assertNotIn(case["forbid_name"], row.get("matched") or "", f"{case['why']} -> {row}")
                if case.get("expect_in_city"):
                    self.assertEqual(str(row.get("in_city")), case["expect_in_city"], f"{case['why']} -> {row}")
                if case.get("expect_tier"):
                    self.assertEqual(row.get("tier"), case["expect_tier"], f"{case['why']} -> {row}")


if __name__ == "__main__":
    unittest.main()
