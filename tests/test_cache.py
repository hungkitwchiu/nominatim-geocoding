"""Unit tests for the caches' pruning (no database needed).

Run with:  .venv/bin/python -m unittest tests.test_cache -v
"""
import os
import tempfile
import unittest
from pathlib import Path

from nomgeo.area import _prune_areas
from nomgeo.cache import KVCache


class CacheGroups(unittest.TestCase):
    def test_only_recent_groups_of_a_dataset_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = KVCache(Path(tmp) / "results.sqlite")
            c.put("old", 1)                                   # written before groups existed
            for i in range(5):
                c.put(f"a{i}", i, f"san_jose:{i}")
                c.keep_groups("san_jose:", f"san_jose:{i}", keep=3)
            c.put("other", 9, "san:1")                        # another dataset sharing a prefix
            c.keep_groups("san:", "san:1", keep=3)
            self.assertEqual(c.get_many(["old", "a0", "a1", "a2", "a3", "a4", "other"]),
                             {"a2": 2, "a3": 3, "a4": 4, "other": 9})

    def test_reusing_a_group_keeps_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = KVCache(Path(tmp) / "results.sqlite")
            for g in ["x:1", "x:2", "x:1", "x:3", "x:4"]:     # back to the first rules once
                c.put(g, g, g)
                c.keep_groups("x:", g, keep=3)
            self.assertEqual(sorted(c.get_many(["x:1", "x:2", "x:3", "x:4"])), ["x:1", "x:3", "x:4"])


class AreaPruning(unittest.TestCase):
    def test_keeps_newest_indexes_of_that_dataset_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            names = [f"san-{i:016x}.pkl" for i in range(5)] + ["san-jose-0000000000000000.pkl", "san-notes.pkl"]
            for t, n in enumerate(names):
                (d / n).write_bytes(b"")
                os.utime(d / n, (1000 + t, 1000 + t))
            _prune_areas(d, "san", keep=3)
            self.assertEqual(sorted(p.name for p in d.iterdir()),
                             sorted(names[2:5] + ["san-jose-0000000000000000.pkl", "san-notes.pkl"]))


if __name__ == "__main__":
    unittest.main()
