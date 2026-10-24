"""Unit tests for the confidence tier of a match (no database needed).

Run with:  .venv/bin/python -m unittest discover -s tests -t .
"""
import unittest

from nomgeo.pipeline import tier


class Tier(unittest.TestCase):
    def test_direct_evidence_with_routine_clean_up(self):
        self.assertEqual(tier("block", []), "A")
        self.assertEqual(tier("intersection", ["type_missing", "direction_added", "small_gap"]), "A")
        self.assertEqual(tier("house", ["unit_removed", "spacing", "city_repeated"]), "A")

    def test_confirmed_corrections_and_near_locations(self):
        self.assertEqual(tier("block", ["several_types", "type_changed"]), "B")
        self.assertEqual(tier("block", ["dominant_evidence"]), "B")
        self.assertEqual(tier("intersection", ["typo_1"]), "B")
        self.assertEqual(tier("near_intersection", []), "B")
        self.assertEqual(tier("between", []), "B")
        self.assertEqual(tier("block", ["block_of_exact_number"]), "B")

    def test_nothing_confirms_the_reading(self):
        self.assertEqual(tier("street", []), "C")
        self.assertEqual(tier("poi", ["poi:leisure/park"]), "C")
        self.assertEqual(tier("block", ["typo_2"]), "C")
        self.assertEqual(tier("street", ["number_not_found"]), "C")
        self.assertEqual(tier("block", ["number_on_first_street", "type_changed"]), "C")


if __name__ == "__main__":
    unittest.main()
