"""Unit tests for name normalisation and record parsing (no database needed).

Run with:  .venv/bin/python -m unittest discover -s tests -v
"""
import unittest
from types import SimpleNamespace

from nomgeo.normalize import ordinal, parse_osm_ref, parse_route, split_street
from nomgeo.parse import ADDRESS, BETWEEN, COORDINATES, EMPTY, HIGHWAY, INTERSECTION, JUNK, \
    OTHER_PLACE, STREET, parse_record


def ds(**kw):
    base = dict(state="CA", city=None, rules=[], strip_after=[], truncation_length=None,
                redacted_first_token=False, single_letter_suffixes=False, between_hyphen=False,
                unreliable_types=frozenset())
    base.update(kw)
    return SimpleNamespace(**base)


class SplitStreet(unittest.TestCase):
    def same(self, a, b):
        self.assertEqual(split_street(a).key(), split_street(b).key(), f"{a!r} vs {b!r}")

    def test_record_and_osm_spellings_agree(self):
        for a, b in [("N 27 ST", "North 27th Street"), ("7TH ST", "Seventh Street"),
                     ("1ST VIEW ST", "First View Street"), ("UNIVERSITY AV E", "University Avenue East"),
                     ("ST CLAIR AV", "St. Clair Avenue"), ("MLK JR WAY", "Martin Luther King Junior Way"),
                     ("DR W J HODGE ST", "Dr. W.J. Hodge Street"), ("OCONNER CRES", "O'Conner Crescent"),
                     ("SIXTY-SECOND ST", "62nd Street"), ("21ST 1/2 ST", "21 1/2 Street"),
                     ("MLK WAY", "Martin Luther King Jr Way"),
                     ("MARTIN LUTHER KING WAY", "Martin Luther King Junior Way")]:
            self.same(a, b)

    def test_letter_and_direction_names_survive(self):
        self.assertEqual(split_street("N ST").base_key, "N")          # Lincoln's N Street
        self.assertEqual(split_street("E ST").base_key, "E")
        self.assertEqual(split_street("West Street").base_key, "WEST")
        self.assertEqual(split_street("AVENUE E").base_key, "AVE E")

    def test_repeated_type_collapses(self):
        self.same("CEDAR STREET STREET", "Cedar Street")
        self.same("A AVENUE AVENUE", "A Avenue")

    def test_ambiguous_police_types_keep_options(self):
        self.assertEqual(split_street("KEMPS CL").suffix, frozenset({"CIR", "CLOSE"}))
        self.assertEqual(split_street("OLIVE L", single_letter_suffixes=True).suffix, frozenset({"LN", "LOOP"}))
        self.assertEqual(split_street("OLIVE L").suffix, frozenset())

    def test_ordinals(self):
        self.assertEqual([ordinal(n) for n in (1, 2, 3, 11, 12, 13, 21, 22, 103)],
                         ["1ST", "2ND", "3RD", "11TH", "12TH", "13TH", "21ST", "22ND", "103RD"])

    def test_routes(self):
        self.assertEqual(parse_route("I-80"), ("I", "80"))
        self.assertEqual(parse_route("I-264W"), ("I", "264"))
        self.assertEqual(parse_route("HWY 2"), (None, "2"))
        self.assertEqual(parse_route("U S HIGHWAY 42"), ("US", "42"))
        self.assertEqual(parse_route("CA 87", state="CA"), ("SR", "87"))
        self.assertIsNone(parse_route("MAIN ST"))
        self.assertEqual(parse_osm_ref("I 80;I 580"), [("I", "80"), ("I", "580")])


class ParseRecord(unittest.TestCase):
    def kind(self, raw, **kw):
        return parse_record(raw, ds(**kw)).kind

    def test_kinds(self):
        self.assertEqual(self.kind("1300 9TH ST, BERKELEY, CA"), ADDRESS)
        self.assertEqual(self.kind("UTOPIA PL & PEAK DR, San Jose, CA"), INTERSECTION)
        self.assertEqual(self.kind("PHILHOWER DR, Norfolk, VA"), STREET)
        self.assertEqual(self.kind(", San Jose, CA"), EMPTY)
        self.assertEqual(self.kind("3100, Norfolk, VA"), EMPTY)
        self.assertEqual(self.kind("Unknown & Unknown, Cambridge, MA"), JUNK)
        self.assertEqual(self.kind("GRETNA, NE, Lincoln, NE"), OTHER_PLACE)
        self.assertEqual(self.kind("200 37.8696807843512;-122.319669759703, BERKELEY, CA"), COORDINATES)
        self.assertEqual(self.kind("NB 280 TO NB 280 RP & NB 280 T, San Jose, CA"), HIGHWAY)
        self.assertEqual(self.kind("136 I65S, Louisville, KY"), HIGHWAY)
        self.assertEqual(self.kind("ASHBY AVE & I-80 HWY, BERKELEY, CA"), HIGHWAY)

    def test_numbered_street_is_not_a_house_number(self):
        r = parse_record("7 ST E & ROBERT ST N, Saint Paul, MN", ds(state="MN"))
        self.assertEqual((r.kind, r.number, r.streets), (INTERSECTION, None, ["7 ST E", "ROBERT ST N"]))
        r = parse_record("1100 7 ST E, Saint Paul, MN", ds(state="MN"))
        self.assertEqual((r.number, r.streets), (1100, ["7 ST E"]))
        r = parse_record("100 PARK AVE, Saint Paul, MN", ds(state="MN"))
        self.assertEqual((r.number, r.streets), (100, ["PARK AVE"]))

    def test_units_removed(self):
        for raw, street in [("6050 MERIDIAN AVE # 1, San Jose, CA", "MERIDIAN AVE"),
                            ("1800 UNIVERSITY AVE 211, BERKELEY, CA", "UNIVERSITY AVE"),
                            ("1700 SAN PABLO (REAR) AVE, BERKELEY, CA", "SAN PABLO AVE"),
                            ("100 HARWOOD CT G, Lincoln, NE", "HARWOOD CT"),
                            ("435 UNIVERSITY AV E, Saint Paul, MN", "UNIVERSITY AV E")]:
            self.assertEqual(parse_record(raw, ds()).streets, [street], raw)

    def test_berkeley_redaction(self):
        b = ds(redacted_first_token=True)
        self.assertEqual(parse_record("6700 ST & SAN PABLO AVE, BERKELEY, CA", b).streets, ["67TH ST", "SAN PABLO AVE"])
        self.assertEqual(parse_record("400 ST & GILMAN ST, BERKELEY, CA", b).streets, ["4TH ST", "GILMAN ST"])
        r = parse_record("00 DARTMOUTH ST, BERKELEY, CA", b)
        self.assertEqual((r.number, r.streets), (0, ["DARTMOUTH ST"]))
        self.assertTrue(parse_record("KAI00 AVE, BERKELEY, CA", b).wildcard_first)

    def test_berkeley_redaction_other_readings(self):
        b = ds(redacted_first_token=True)
        raw = "2100 CENTER & OXFORD ST, BERKELEY, CA"
        r = parse_record(raw, b)
        self.assertEqual((r.streets, r.alternative), (["21ST CENTER", "OXFORD ST"], "number"))
        r = parse_record(raw, b, reading="number")
        self.assertEqual((r.kind, r.number, r.streets), (INTERSECTION, 2100, ["CENTER", "OXFORD ST"]))
        raw = "00 ROY AVE & VIRGINIA ST, BERKELEY, CA"
        self.assertEqual(parse_record(raw, b).alternative, "name")
        r = parse_record(raw, b, reading="name")
        self.assertEqual((r.streets, r.wildcard_first), (["00 ROY AVE", "VIRGINIA ST"], True))

    def test_repeated_city_and_glued_zip(self):
        r = parse_record("2400 PIEDMONT, BERKELEY, BERKELEY, CA", ds(city="Berkeley"))
        self.assertEqual((r.number, r.streets), (2400, ["PIEDMONT"]))
        self.assertIn("city_repeated", r.flags)
        r = parse_record("1600 MARTIN LUTHER KING JR94709 WAY, BERKELEY, CA", ds())
        self.assertEqual((r.number, r.streets), (1600, ["MARTIN LUTHER KING JR WAY"]))
        self.assertIn("zip_removed", r.flags)
        self.assertEqual(parse_record("GRETNA, NE, Lincoln, NE", ds(city="Lincoln")).kind, OTHER_PLACE)

    def test_split_ordinals(self):
        for raw, number, streets in [("1600 8 TH, BERKELEY, CA", 1600, ["8TH"]),
                                     ("8600 S 33 RD CIRC, Lincoln, NE", 8600, ["S 33RD CIRC"]),
                                     ("22 ND ST & PORTLAND AVE, Louisville, KY", None, ["22ND ST", "PORTLAND AVE"]),
                                     ("5800 N 33 RD, Lincoln, NE", 5800, ["N 33 RD"]),       # maybe 33 Road
                                     ("1 ST JAMES PL, Louisville, KY", 1, ["ST JAMES PL"]),
                                     ("N 11 ND ST & O ST, Lincoln, NE", None, ["N 11 ND ST", "O ST"])]:  # 11TH
            r = parse_record(raw, ds())
            self.assertEqual((r.number, r.streets), (number, streets), raw)

    def test_lincoln_between(self):
        r = parse_record("26TH & K-N ST, Lincoln, NE", ds(state="NE", between_hyphen=True))
        self.assertEqual((r.kind, r.streets), (BETWEEN, ["26TH", "K ST", "N ST"]))
        r = parse_record("HWY 77 & I-80, Lincoln, NE", ds(state="NE", between_hyphen=True))
        self.assertEqual(r.kind, HIGHWAY)

    def test_san_jose_offset(self):
        for raw, streets in [("VINE ST S & O GOODYEAR ST, San Jose, CA", ["VINE ST", "GOODYEAR ST"]),
                             ("S 5TH N & OF KEYES ST, San Jose, CA", ["S 5TH", "KEYES ST"]),
                             ("ALMA & W OF MONTEREY, San Jose, CA", ["ALMA", "MONTEREY"]),
                             ("MONTEREY RD N OF LEWIS, San Jose, CA", ["MONTEREY RD", "LEWIS"])]:
            r = parse_record(raw, ds())
            self.assertEqual((r.kind, r.streets), (INTERSECTION, streets), raw)
            self.assertIn("offset_of_intersection", r.flags, raw)
        # "OF" inside a name, or after a number with no street before it, is not an offset.
        r = parse_record("2950 VALLEY OF HEARTS DELIGHT PL, San Jose, CA", ds())
        self.assertEqual((r.number, r.streets), (2950, ["VALLEY OF HEARTS DELIGHT PL"]))
        self.assertNotIn("offset_of_intersection", parse_record("50 NB S OF ALMADEN, San Jose, CA", ds()).flags)

    def test_san_jose_number_at_street(self):
        for raw, number, street in [("150 AT BLOSSOM HILL, San Jose, CA", 150, "BLOSSOM HILL"),
                                    ("250 NB AT MERIDIAN, San Jose, CA", 250, "MERIDIAN"),
                                    ("1300 BLK OF AIRPORT, San Jose, CA", 1300, "AIRPORT")]:
            r = parse_record(raw, ds())
            self.assertEqual((r.kind, r.number, r.streets), (ADDRESS, number, [street]), raw)
        r = parse_record("50 ST AT 280, San Jose, CA", ds())          # a street "at" another
        self.assertEqual((r.kind, r.streets), (INTERSECTION, ["50 ST", "280"]))

    def test_dead_end(self):
        r = parse_record("MARTWOOD WY & DEAD END, San Jose, CA", ds())
        self.assertEqual((r.kind, r.streets, r.dead_end), (STREET, ["MARTWOOD WY"], True))
        r = parse_record("DEAD END & E ALMA AV, San Jose, CA", ds())
        self.assertEqual((r.streets, r.dead_end), (["E ALMA AV"], True))
        r = parse_record("A ST & DEAD END & B ST, San Jose, CA", ds())
        self.assertEqual((r.kind, r.dead_end), (INTERSECTION, False))


if __name__ == "__main__":
    unittest.main()
