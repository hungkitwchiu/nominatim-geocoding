"""Unit tests for the ambiguity decisions in locate.py (no database needed).

`choose` decides whether several candidate locations are one place, a lopsided set with one
dominant place, variants of one name, several crossings, or ambiguous; `_merge_same_street`
merges split evidence for one block and drops stray points.  Both take plain Locations.

Run with:  .venv/bin/python -m unittest discover -s tests -t .
"""
import unittest
from types import SimpleNamespace

from nomgeo.locate import Location, _merge_same_street, choose

DS = SimpleNamespace(precision={})          # the defaults written in the code


def loc(x, y, match_type="block", penalty=0, in_core=True, weight=1.0, names=("Main Street",), flags=()):
    return Location(x, y, match_type, 30.0, "osm_address", tuple(names), tuple(flags), penalty, in_core,
                    "", weight)


class Choose(unittest.TestCase):
    def test_one_place_is_merged_weighted_by_evidence(self):
        got, why = choose([loc(0, 0), loc(40, 0, weight=3.0)], DS, "street/number")
        self.assertIsNone(why)
        self.assertEqual((got.x, got.y), (30.0, 0.0))

    def test_two_places_far_apart_are_ambiguous(self):
        got, why = choose([loc(0, 0), loc(2000, 0)], DS, "street/number")
        self.assertIsNone(got)
        self.assertEqual(why, "ambiguous street/number: 2 different places")

    def test_lopsided_evidence_on_one_street_wins(self):
        got, why = choose([loc(0, 0, weight=25.0), loc(870, 0, weight=1.0)], DS, "street/number")
        self.assertEqual((got.x, got.y), (0.0, 0.0))
        self.assertIn("dominant_evidence", got.flags)
        # ... unless the minority is big enough to be a real second place.
        got, why = choose([loc(0, 0, weight=25.0), loc(870, 0, weight=5.0)], DS, "street/number")
        self.assertIsNone(got)

    def test_in_city_reading_beats_a_less_relaxed_one_outside(self):
        outside = loc(0, 0, penalty=0, in_core=False, names=("34th Street",))
        got, why = choose([outside, loc(5000, 0, penalty=2, names=("West 34th Street",))], DS, "street/number")
        self.assertEqual(got.names, ("West 34th Street",))
        self.assertIn("preferred_in_city", got.flags)
        # Four steps of relaxation more is too many: the reading outside the city stands.
        got, why = choose([outside, loc(5000, 0, penalty=4, names=("West 34th Street",))], DS, "street/number")
        self.assertEqual(got.names, ("34th Street",))

    def test_variants_of_one_name_close_together_meet_in_the_middle(self):
        got, why = choose([loc(0, 0, names=("North Almaden Boulevard",)),
                           loc(210, 0, names=("South Almaden Boulevard",))], DS, "street/number")
        self.assertEqual(got.x, 105.0)
        self.assertIn("variants_merged", got.flags)

    def test_several_crossings_close_together_meet_in_the_middle(self):
        names = ("Camden Avenue", "Branham Lane")
        a = loc(0, 0, "intersection", names=names)
        got, why = choose([a, loc(270, 0, "intersection", names=names)], DS, "intersection")
        self.assertEqual(got.x, 135.0)
        self.assertIn("several_crossings", got.flags)
        got, why = choose([a, loc(900, 0, "intersection", names=names)], DS, "intersection")
        self.assertIsNone(got)


class MergeSameStreet(unittest.TestCase):
    def test_split_evidence_for_one_block_becomes_the_middle_of_the_stretch(self):
        out = _merge_same_street([loc(0, 0, weight=8.0), loc(500, 0, weight=6.0)], 700.0, 80.0)
        self.assertEqual([(l.x, l.precision_m) for l in out], [(250.0, 250.0)])

    def test_numbering_in_another_town_stays_separate(self):
        self.assertEqual(len(_merge_same_street([loc(0, 0), loc(3000, 0)], 700.0, 80.0)), 2)

    def test_a_stray_point_near_many_agreeing_points_is_dropped(self):
        out = _merge_same_street([loc(0, 0, weight=8.0), loc(750, 0, weight=1.0)], 700.0, 80.0)
        self.assertEqual([(l.x, l.weight) for l in out], [(0.0, 8.0)])
        self.assertIn("outlier_points_dropped", out[0].flags)
        # Never across the city line: places on either side stay apart for the in-city preference.
        out = _merge_same_street([loc(0, 0, weight=8.0, in_core=False), loc(750, 0, weight=1.0)], 700.0, 80.0)
        self.assertEqual(len(out), 2)

    def test_an_exact_number_in_two_places_is_flagged(self):
        far = _merge_same_street([loc(0, 0, "interpolated"), loc(600, 0, "interpolated")], 700.0, 80.0)
        self.assertEqual((far[0].x, far[0].precision_m), (300.0, 300.0))
        self.assertIn("number_in_two_places", far[0].flags)
        # Within one place it is one spot; and a block is a stretch of street by nature.
        near = _merge_same_street([loc(0, 0, "interpolated"), loc(100, 0, "interpolated")], 700.0, 80.0)
        blocks = _merge_same_street([loc(0, 0), loc(600, 0)], 700.0, 80.0)
        self.assertNotIn("number_in_two_places", near[0].flags + blocks[0].flags)


if __name__ == "__main__":
    unittest.main()
