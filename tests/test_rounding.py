"""Unit tests for block labels and the rounding convention (no database needed).

Run with:  .venv/bin/python -m unittest discover -s tests -t .
"""
import unittest

from nomgeo.locate import _block_range
from nomgeo.pipeline import rounding_from_counts


class BlockLabels(unittest.TestCase):
    def test_floor_and_nearest(self):
        self.assertEqual(_block_range(1300, "hundred_block"), (1300, 1399))
        self.assertEqual(_block_range(1300, "hundred_block", "floor"), (1300, 1399))
        self.assertEqual(_block_range(1300, "hundred_block", "nearest"), (1250, 1349))

    def test_midpoint_labels_are_the_same_block_either_way(self):
        self.assertEqual(_block_range(3550, "hundred_block", "floor"), (3500, 3599))
        self.assertEqual(_block_range(3550, "hundred_block", "nearest"), (3500, 3599))

    def test_exact_numbers_have_no_block(self):
        self.assertIsNone(_block_range(1395, "exact", "nearest"))
        self.assertIsNone(_block_range(1300, "exact"))


class RoundingDecision(unittest.TestCase):
    def test_nearest_when_the_next_hundred_is_common_at_half_open_ends(self):
        used, info = rounding_from_counts(344, 119, 378, 14)        # Cambridge
        self.assertEqual(used, "nearest")
        self.assertEqual(info["next_hundred_share_50_99"], 0.346)

    def test_floor_when_rare_in_both_groups(self):
        self.assertEqual(rounding_from_counts(420, 2, 702, 1)[0], "floor")       # Berkeley
        self.assertEqual(rounding_from_counts(8497, 81, 6627, 31)[0], "floor")   # San Jose

    def test_floor_when_the_baseline_is_high_too(self):
        self.assertEqual(rounding_from_counts(500, 60, 500, 55)[0], "floor")

    def test_too_few_records_means_floor_with_a_note(self):
        used, info = rounding_from_counts(40, 20, 300, 5)
        self.assertEqual(used, "floor")
        self.assertIn("too few", info["note"])


if __name__ == "__main__":
    unittest.main()
