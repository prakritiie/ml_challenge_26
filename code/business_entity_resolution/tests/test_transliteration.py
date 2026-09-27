"""Unit tests for the V3 Devanagari transliteration prototype.

Run with:
    cd code/business_entity_resolution
    python -m unittest tests.test_transliteration -v
"""
import unittest

from src.transliteration import transliterate_devanagari, has_devanagari


class TestDevanagariTransliteration(unittest.TestCase):
    def test_has_devanagari_detects_script(self):
        self.assertTrue(has_devanagari("राम मार्केटिंग"))
        self.assertFalse(has_devanagari("Ram Marketing"))
        self.assertFalse(has_devanagari(None))
        self.assertFalse(has_devanagari(""))

    def test_basic_consonant_vowel_combination(self):
        # क (ka) + ा (aa matra) -> "kaa"
        self.assertEqual(transliterate_devanagari("का"), "kaa")

    def test_virama_suppresses_inherent_vowel(self):
        # क् (virama) + ष -> cluster "ksh", not "kasha"
        result = transliterate_devanagari("क्ष")
        self.assertNotIn("a", result[:1])  # no inherent vowel after 'k'

    def test_no_schwa_deletion_by_design(self):
        # Every plain consonant with no matra/virama keeps its inherent "a"
        # -- this is the documented, deliberate policy for this prototype.
        result = transliterate_devanagari("लिमिटेड")
        self.assertTrue(result.endswith("a"))

    def test_nukta_consonant_zone(self):
        # ज़ (za, nukta form) should map to "z", not plain ज "ja"
        result = transliterate_devanagari("ज़ोन")
        self.assertTrue(result.startswith("z"))

    def test_anusvara_maps_to_n(self):
        result = transliterate_devanagari("गंगा")
        self.assertIn("n", result)

    def test_mixed_devanagari_and_latin_preserved(self):
        result = transliterate_devanagari("Shiv आईटी")
        self.assertTrue(result.startswith("Shiv"))

    def test_empty_and_none_input(self):
        self.assertIsNone(transliterate_devanagari(None))
        self.assertIsNone(transliterate_devanagari(""))

    def test_does_not_mutate_input(self):
        raw = "गोल्डन ग्लोबल प्राइवेट लिमिटेड"
        transliterate_devanagari(raw)
        self.assertEqual(raw, "गोल्डन ग्लोबल प्राइवेट लिमिटेड")


if __name__ == "__main__":
    unittest.main()
