"""Regression tests for the normalization stage.

Run with:
    cd code/business_entity_resolution
    python -m unittest tests.test_normalization -v
"""
import unittest

from src.normalization import normalize_address, normalize_address_fields, normalize_name_fields


class TestDrAmbiguity(unittest.TestCase):
    """'Dr' is ambiguous: it can be the title 'Doctor' inside a person's name
    that a road is named after, or the street-type abbreviation for 'Drive'.
    These are the four cases explicitly called out in the bug report."""

    def test_dr_as_doctor_title_not_expanded(self):
        # "Dr Rajkumar Road" must NOT become "drive rajkumar road".
        result = normalize_address("Dr Rajkumar Road")
        self.assertEqual(result, "dr rajkumar road")
        self.assertNotIn("drive", result)

    def test_dr_as_doctor_title_with_street_number_not_expanded(self):
        result = normalize_address("123 Dr Martin Luther King Jr Blvd")
        self.assertIn("dr martin luther king jr", result)
        self.assertNotIn("drive", result)
        # "Blvd" (unambiguous) should still expand normally.
        self.assertIn("boulevard", result)

    def test_dr_as_drive_at_end_of_segment_is_expanded(self):
        # A genuine street-type "Dr" at the end of an address segment should
        # still normalize to "drive".
        result = normalize_address("123 Main Dr")
        self.assertEqual(result, "123 main drive")

    def test_dr_as_drive_before_comma_is_expanded(self):
        result = normalize_address("123 Main Dr, Springfield, IL")
        self.assertTrue(result.split(",")[0].strip().endswith("drive"))

    def test_dr_with_period_as_doctor_title_not_expanded(self):
        result = normalize_address("Dr. ABC Road")
        self.assertNotIn("drive", result)
        self.assertIn("abc road", result)


class TestExistingBehaviorUnaffected(unittest.TestCase):
    """Sanity checks that the Dr fix did not disturb previously-validated
    behavior in the same code path (address abbreviation expansion)."""

    def test_st_saint_guard_still_works(self):
        result = normalize_address("St Louis, MO")
        self.assertTrue(result.startswith("st louis"))

    def test_st_street_expansion_still_works_mid_address(self):
        result = normalize_address("100 Hill Street, Athens, TN")
        self.assertIn("hill street", result)

    def test_rd_ave_still_expand(self):
        result = normalize_address("12 Green Needles Road, Littleton, MA")
        self.assertIn("road", result)

    def test_no_number_expansion_still_works(self):
        result = normalize_address("House No 58, Pune")
        self.assertIn("number 58", result)

    def test_none_literal_still_missing(self):
        self.assertIsNone(normalize_address("None"))
        self.assertIsNone(normalize_address("null"))

    def test_legal_suffix_core_still_stripped(self):
        fields = normalize_name_fields("Sai Systems Private Limited")
        self.assertEqual(fields["name_core"], "sai systems")

    def test_non_latin_name_preserved_intact(self):
        fields = normalize_name_fields("राम मार्केटिंग प्राइवेट लिमिटेड")
        self.assertEqual(fields["name_tokens"], ["राम", "मार्केटिंग", "प्राइवेट", "लिमिटेड"])

    def test_component_level_missing_marker_dropped(self):
        result = normalize_address("Shop No. 49, <NULL>, Hyderabad, Andhra Pradesh")
        self.assertNotIn("null", result)
        self.assertIn("hyderabad", result)


class TestOriginalFieldUntouched(unittest.TestCase):
    def test_normalize_address_fields_does_not_mutate_input(self):
        raw = "Dr Rajkumar Road"
        fields = normalize_address_fields(raw)
        self.assertEqual(raw, "Dr Rajkumar Road")
        self.assertEqual(fields["address_normalized"], "dr rajkumar road")


if __name__ == "__main__":
    unittest.main()
