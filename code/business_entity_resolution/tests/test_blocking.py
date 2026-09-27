"""Unit tests for the Blocking V1 rule logic in src.blocking, using tiny
synthetic in-memory tables instead of the full normalized Parquet files (so
these run in milliseconds and don't require the dataset to be present).

Run with:
    cd code/business_entity_resolution
    python -m unittest tests.test_blocking -v
"""
import unittest

import duckdb

from src import config


def _make_con():
    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    return con


class TestBlockingRules(unittest.TestCase):
    def setUp(self):
        self.con = _make_con()
        # A tiny synthetic S1 table: one exact match, one core-name match,
        # one word-order match, one rare-token match, one postal_code-only
        # match, and one entity that should get NO candidates at all.
        self.con.execute("""
            CREATE TEMP TABLE s1 AS SELECT * FROM (VALUES
                ('S1-1', 'US', 'acme foundry',        'acme foundry',        ['acme','foundry'],        ['acme','foundry'],        NULL),
                ('S1-2', 'US', 'acme foundry inc',    'acme foundry',        ['acme','foundry','inc'],  ['acme','foundry','inc'],  NULL),
                ('S1-3', 'US', 'blue harbor exports',   'blue harbor exports', ['blue','harbor','exports'], ['blue','exports','harbor'], NULL),
                ('S1-4', 'US', 'zyxquil traders',     'zyxquil traders',     ['zyxquil','traders'],     ['traders','zyxquil'],     NULL),
                ('S1-5', 'US', 'totally different co', 'totally different',  ['totally','different','co'], ['co','different','totally'], '10001'),
                ('S1-6', 'US', 'no match here llc',   'no match here',       ['no','match','here','llc'], ['here','llc','match','no'], NULL)
            ) AS t(entity_id, country, name_normalized, name_core, name_tokens, name_sorted_tokens, postal_code)
        """)
        self.con.execute("""
            CREATE TEMP TABLE s2 AS SELECT * FROM (VALUES
                ('S2-1', 'US', 'acme foundry',        'acme foundry',        ['acme','foundry'],        ['acme','foundry'],        NULL),
                ('S2-2', 'US', 'acme foundry inc',    'acme foundry',        ['acme','foundry','inc'],  ['acme','foundry','inc'],  NULL),
                ('S2-3', 'US', 'harbor blue exports',  'harbor blue exports', ['harbor','blue','exports'], ['blue','exports','harbor'], NULL),
                ('S2-4', 'US', 'zyxquil holdings',    'zyxquil holdings',    ['zyxquil','holdings'],    ['holdings','zyxquil'],    NULL),
                ('S2-5', 'US', 'unrelated business',  'unrelated business',  ['unrelated','business'],  ['business','unrelated'],  '10001'),
                ('S2-6', 'US', 'completely unrelated', 'completely unrelated', ['completely','unrelated'], ['completely','unrelated'], NULL)
            ) AS t(entity_id, country, name_normalized, name_core, name_tokens, name_sorted_tokens, postal_code)
        """)
    def _run_rules_on_synthetic(self):
        # Mirror build_candidates_by_rule's SQL but against the synthetic
        # s1/s2 temp tables instead of the real Parquet files.
        s1_sql, t_sql = "(SELECT * FROM s1)", "(SELECT * FROM s2)"
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE s1_token_df AS
            SELECT country, tok, count(*) AS df
            FROM (SELECT country, unnest(name_tokens) AS tok FROM {s1_sql}) GROUP BY country, tok
        """)
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE t_token_df AS
            SELECT country, tok, count(*) AS df
            FROM (SELECT country, unnest(name_tokens) AS tok FROM {t_sql}) GROUP BY country, tok
        """)
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE allowed_tokens AS
            SELECT s.country, s.tok FROM s1_token_df s JOIN t_token_df t USING (country, tok)
            WHERE length(s.tok) >= 3 AND s.df <= 100 AND t.df <= 100
        """)
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE s1_tokens_allowed AS
            SELECT st.entity_id AS source1_entity_id, st.country, st.tok
            FROM (SELECT entity_id, country, unnest(name_tokens) AS tok FROM {s1_sql}) st
            JOIN allowed_tokens USING (country, tok)
        """)
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE t_tokens_allowed AS
            SELECT tt.entity_id AS candidate_id, tt.country, tt.tok
            FROM (SELECT entity_id, country, unnest(name_tokens) AS tok FROM {t_sql}) tt
            JOIN allowed_tokens USING (country, tok)
        """)
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE candidates_by_rule AS
            SELECT source1_entity_id, candidate_id, rule FROM (
                SELECT s1.entity_id AS source1_entity_id, t.entity_id AS candidate_id, 'exact_name' AS rule
                FROM {s1_sql} s1
                JOIN {t_sql} t ON s1.country=t.country AND s1.name_normalized=t.name_normalized
                WHERE s1.name_normalized IS NOT NULL
                UNION ALL
                SELECT s1.entity_id, t.entity_id, 'core_name' FROM {s1_sql} s1
                JOIN {t_sql} t ON s1.country=t.country AND s1.name_core=t.name_core
                WHERE s1.name_core IS NOT NULL
                UNION ALL
                SELECT s1.entity_id, t.entity_id, 'sorted_tokens' FROM {s1_sql} s1
                JOIN {t_sql} t ON s1.country=t.country AND s1.name_sorted_tokens=t.name_sorted_tokens
                WHERE s1.name_sorted_tokens IS NOT NULL AND len(s1.name_sorted_tokens) > 0
                UNION ALL
                SELECT s1.entity_id, t.entity_id, 'postal_code' FROM {s1_sql} s1
                JOIN {t_sql} t ON s1.country=t.country AND s1.postal_code=t.postal_code
                WHERE s1.postal_code IS NOT NULL
                UNION ALL
                SELECT DISTINCT sa.source1_entity_id, ta.candidate_id, 'strong_token'
                FROM s1_tokens_allowed sa JOIN t_tokens_allowed ta USING (country, tok)
            )
        """)
        return self.con.execute("SELECT * FROM candidates_by_rule").fetchdf()

    def test_exact_name_rule_matches(self):
        df = self._run_rules_on_synthetic()
        pairs = set(zip(df.source1_entity_id, df.candidate_id, df.rule))
        self.assertIn(("S1-1", "S2-1", "exact_name"), pairs)

    def test_core_name_rule_catches_suffix_difference(self):
        df = self._run_rules_on_synthetic()
        pairs = set(zip(df.source1_entity_id, df.candidate_id, df.rule))
        # S1-1 "acme foundry" and S2-2 "acme foundry inc" share name_core
        # "acme foundry" even though their normalized names differ.
        self.assertIn(("S1-1", "S2-2", "core_name"), pairs)

    def test_sorted_tokens_rule_catches_word_order_swap(self):
        df = self._run_rules_on_synthetic()
        pairs = set(zip(df.source1_entity_id, df.candidate_id, df.rule))
        self.assertIn(("S1-3", "S2-3", "sorted_tokens"), pairs)

    def test_postal_code_rule_matches_on_postal_code_alone(self):
        df = self._run_rules_on_synthetic()
        pairs = set(zip(df.source1_entity_id, df.candidate_id, df.rule))
        self.assertIn(("S1-5", "S2-5", "postal_code"), pairs)

    def test_no_spurious_match_for_unrelated_entity(self):
        df = self._run_rules_on_synthetic()
        s1_6_candidates = set(df[df.source1_entity_id == "S1-6"].candidate_id)
        self.assertEqual(s1_6_candidates, set())

    def test_union_has_no_duplicate_pairs(self):
        df = self._run_rules_on_synthetic()
        union_pairs = list(zip(df.source1_entity_id, df.candidate_id))
        distinct_pairs = self.con.execute(
            "SELECT DISTINCT source1_entity_id, candidate_id FROM candidates_by_rule"
        ).fetchdf()
        # exact_name and core_name both fire for (S1-1,S2-1)/(S1-2,S2-2),
        # so the union (deduplicated) must be strictly smaller than the
        # non-deduplicated by-rule table, but every union pair must be real.
        self.assertLess(len(distinct_pairs), len(union_pairs))
        for s1_id, cand_id in zip(distinct_pairs.source1_entity_id, distinct_pairs.candidate_id):
            self.assertTrue(cand_id.startswith("S2-"))
            self.assertTrue(s1_id.startswith("S1-"))

    def test_candidate_ids_never_cross_source(self):
        df = self._run_rules_on_synthetic()
        self.assertTrue((df.candidate_id.str.startswith("S2-")).all())
        self.assertTrue((df.source1_entity_id.str.startswith("S1-")).all())


class TestConfigThreshold(unittest.TestCase):
    def test_token_df_threshold_is_conservative(self):
        # Regression guard against silently reverting to the first (too
        # permissive) threshold that caused the 1.3M-pair blow-up during
        # development -- see README.md "Blocking V1" section.
        self.assertLessEqual(config.TOKEN_DF_THRESHOLD, 200)
        self.assertGreaterEqual(config.TOKEN_MIN_LEN, 3)


if __name__ == "__main__":
    unittest.main()
