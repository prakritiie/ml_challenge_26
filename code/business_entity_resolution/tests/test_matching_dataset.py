"""Unit tests for matching_dataset.py (labeling, split, feature engineering)
and candidate_generation.py, using tiny synthetic in-memory tables.

Run with:
    cd code/business_entity_resolution
    python -m unittest tests.test_matching_dataset -v
"""
import unittest

import duckdb

from src.matching_dataset import (
    FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS, REMOVED_FEATURES,
    ID_COLUMNS, LABEL_COLUMN, SPLIT_COLUMN, SPLIT_TRAIN_FRAC,
)
from src.candidate_generation import ALL_RULES, V1_RULES, NEW_RULES

# Feature pairs known (from feature_validation, see README.md) to be exact
# duplicates -- at most one member of each pair may survive into the clean
# model feature list.
KNOWN_DUPLICATE_PAIRS = [
    ("exact_normalized_name", "blocked_exact_name"),
    ("exact_core_name", "blocked_core_name"),
    ("exact_sorted_tokens", "blocked_sorted_tokens"),
]
KNOWN_CONSTANT_FEATURES = ["country_equal"]


class TestModelFeatureCleanup(unittest.TestCase):
    def test_constant_feature_removed(self):
        for f in KNOWN_CONSTANT_FEATURES:
            self.assertIn(f, REMOVED_FEATURES)
            self.assertNotIn(f, MODEL_FEATURE_COLUMNS)

    def test_duplicate_pairs_reduced_to_one_survivor(self):
        for kept, dropped in KNOWN_DUPLICATE_PAIRS:
            self.assertIn(dropped, REMOVED_FEATURES)
            self.assertNotIn(dropped, MODEL_FEATURE_COLUMNS)
            self.assertIn(kept, MODEL_FEATURE_COLUMNS)

    def test_model_feature_count_is_27(self):
        self.assertEqual(len(MODEL_FEATURE_COLUMNS), 27)
        self.assertEqual(len(FEATURE_COLUMNS) - len(REMOVED_FEATURES), len(MODEL_FEATURE_COLUMNS))

    def test_no_duplicate_names_in_model_features(self):
        self.assertEqual(len(MODEL_FEATURE_COLUMNS), len(set(MODEL_FEATURE_COLUMNS)))

    def test_removed_features_are_subset_of_raw_features(self):
        self.assertTrue(set(REMOVED_FEATURES).issubset(set(FEATURE_COLUMNS)))

    def test_no_id_or_label_or_split_columns_in_model_features(self):
        for col in ID_COLUMNS + [LABEL_COLUMN, SPLIT_COLUMN]:
            self.assertNotIn(col, MODEL_FEATURE_COLUMNS)

    def test_no_country_specific_hardcoding_in_feature_names(self):
        # Sanity: no feature name bakes in a specific country/script value
        # (would indicate hardcoding rather than a general comparison).
        for f in MODEL_FEATURE_COLUMNS:
            self.assertNotIn("india", f.lower())
            self.assertNotIn("us_", f.lower())


class TestFeatureColumnsNoLeakage(unittest.TestCase):
    def test_no_id_columns_in_features(self):
        for col in FEATURE_COLUMNS:
            self.assertNotIn("entity_id", col)
            self.assertNotIn("source1", col)
            self.assertNotIn("candidate_id", col)

    def test_label_not_in_features(self):
        self.assertNotIn("label", FEATURE_COLUMNS)

    def test_no_duplicate_feature_names(self):
        self.assertEqual(len(FEATURE_COLUMNS), len(set(FEATURE_COLUMNS)))

    def test_all_ten_blocking_provenance_flags_present(self):
        expected = {
            "blocked_exact_name", "blocked_core_name", "blocked_sorted_tokens", "blocked_strong_token",
            "blocked_ngram", "blocked_postal", "blocked_city_street",
            "blocked_rare_address_token", "blocked_address_ngram", "blocked_name_prefix",
        }
        self.assertTrue(expected.issubset(set(FEATURE_COLUMNS)))


class TestRuleSetConsistency(unittest.TestCase):
    def test_all_rules_is_union_of_v1_and_new(self):
        self.assertEqual(set(ALL_RULES), set(V1_RULES) | set(NEW_RULES))

    def test_ten_rules_total(self):
        self.assertEqual(len(ALL_RULES), 10)


class TestEntitySplitDeterminism(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute("""
            CREATE TEMP TABLE fake_s1 AS SELECT * FROM (VALUES
                ('S1-1'), ('S1-2'), ('S1-3'), ('S1-4'), ('S1-5'),
                ('S1-6'), ('S1-7'), ('S1-8'), ('S1-9'), ('S1-10')
            ) AS t(entity_id)
        """)

    def _build_split(self, cutoff=80):
        return self.con.execute(f"""
            SELECT entity_id, CASE WHEN abs(hash(entity_id)) % 100 < {cutoff} THEN 'train' ELSE 'validation' END AS split
            FROM fake_s1
        """).fetchdf()

    def test_split_is_deterministic_across_calls(self):
        a = self._build_split()
        b = self._build_split()
        self.assertEqual(a["split"].tolist(), b["split"].tolist())

    def test_every_entity_assigned_exactly_one_split(self):
        df = self._build_split()
        self.assertEqual(set(df["split"].unique()) - {"train", "validation"}, set())
        self.assertEqual(len(df), 10)

    def test_default_split_fraction_is_80_20(self):
        self.assertEqual(SPLIT_TRAIN_FRAC, 0.8)


class TestFeatureLogicOnSyntheticPair(unittest.TestCase):
    """Sanity-check the actual SQL expressions used in build_features by
    running them against a couple of hand-built rows, independent of the
    real Parquet files."""

    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute("""
            CREATE TEMP TABLE pair AS SELECT * FROM (VALUES
                ('acme foundry', 'acme foundry', ['acme','foundry'], ['acme','foundry'],
                 'acme foundry',  'acme foundry', ['acme','foundry']),
                ('acme foundry', 'acme fondry',  ['acme','foundry'], ['acme','fondry'],
                 'acme foundry',  'acme fondry',  ['acme','fondry'])
            ) AS t(s1_norm, t_norm, s1_tok, t_tok, s1_core, t_core, t_core_tok)
        """)

    def test_exact_match_row_has_jaccard_one(self):
        row = self.con.execute("""
            SELECT (s1_norm = t_norm) AS exact,
                   len(list_intersect(s1_tok, t_tok))*1.0 / greatest(len(list_distinct(list_concat(s1_tok, t_tok))), 1) AS jac
            FROM pair LIMIT 1
        """).fetchone()
        self.assertTrue(row[0])
        self.assertEqual(row[1], 1.0)

    def test_typo_row_has_small_edit_distance_and_partial_jaccard(self):
        row = self.con.execute("""
            SELECT levenshtein(s1_core, t_core) AS dist,
                   (s1_norm = t_norm) AS exact,
                   len(list_intersect(s1_tok, t_tok))*1.0 / greatest(len(list_distinct(list_concat(s1_tok, t_tok))), 1) AS jac
            FROM pair OFFSET 1 LIMIT 1
        """).fetchone()
        self.assertLessEqual(row[0], 2)
        self.assertFalse(row[1])
        self.assertLess(row[2], 1.0)


if __name__ == "__main__":
    unittest.main()
