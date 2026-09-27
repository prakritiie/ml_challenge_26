"""Blocking V1: candidate-pair generation for Source1 -> Source2 and
Source1 -> Source3, kept as two separate indexes (never combined).

This module implements ONLY blocking / candidate generation -- no
similarity features, embeddings, or ML matching. See README.md for the
pipeline roadmap.

Design
------
All heavy lifting is pure SQL executed by DuckDB directly against the
normalized Parquet files (see normalization.py): joins, not Python loops,
so nothing here does an all-pairs comparison and nothing materializes a
full source table in Python. Every rule is an equi-join on a blocking key
(never a fuzzy comparison), which is what keeps each rule's cost close to
"number of real collisions" rather than |S1| x |target|.

Five rules, applied independently and then unioned:
  1. exact_name    : country + name_normalized
  2. core_name     : country + name_core
  3. sorted_tokens : country + name_sorted_tokens (word-order invariant)
  4. strong_token  : country + a single name token, restricted to tokens
                     that are NOT overly common (see TOKEN_DF_THRESHOLD)
  5. postal_code   : country + postal_code (only ~1% of rows have one)

Country is treated as an open string everywhere (US/India in train,
France additionally in test) -- never hardcoded or filtered to a fixed set.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

from . import config, io_utils

S1_KEY = "train_source1"


def _sampled_sql(base_sql: str, n: Optional[int]) -> str:
    if n is None:
        return f"(SELECT * FROM {base_sql})"
    return f"(SELECT * FROM {base_sql} USING SAMPLE {n} (reservoir, 42))"


def _s1_sql(n: Optional[int] = None) -> str:
    return _sampled_sql(io_utils.parquet_sql(config.NORMALIZED_FILES[S1_KEY]), n)


def _target_sql(target_key: str) -> str:
    return f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[target_key])})"


def build_candidates_by_rule(con, target_key: str, s1_sample: Optional[int] = None) -> None:
    """Create a TEMP TABLE `candidates_by_rule` with columns
    (source1_entity_id, candidate_id, rule), one row per (pair, rule) hit
    (NOT deduplicated across rules -- that's what lets blocking_eval.py
    report per-rule metrics as well as the union)."""
    s1_sql = _s1_sql(s1_sample)
    t_sql = _target_sql(target_key)

    # --- strong-token rule setup: symmetric document-frequency filter ------
    # A token is used as a blocking key only if it is not overly common on
    # EITHER side (S1's own frequency AND the target's frequency), which
    # bounds the worst-case pair blow-up from a single token to roughly
    # TOKEN_DF_THRESHOLD^2 instead of an unbounded product.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_token_df AS
        SELECT country, tok, count(*) AS df
        FROM (SELECT country, unnest(name_tokens) AS tok FROM {s1_sql})
        GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE t_token_df AS
        SELECT country, tok, count(*) AS df
        FROM (SELECT country, unnest(name_tokens) AS tok FROM {t_sql})
        GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE allowed_tokens AS
        SELECT s.country, s.tok
        FROM s1_token_df s
        JOIN t_token_df t USING (country, tok)
        WHERE length(s.tok) >= {config.TOKEN_MIN_LEN}
          AND s.df <= {config.TOKEN_DF_THRESHOLD}
          AND t.df <= {config.TOKEN_DF_THRESHOLD}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_tokens_allowed AS
        SELECT st.entity_id AS source1_entity_id, st.country, st.tok
        FROM (SELECT entity_id, country, unnest(name_tokens) AS tok FROM {s1_sql}) st
        JOIN allowed_tokens USING (country, tok)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE t_tokens_allowed AS
        SELECT tt.entity_id AS candidate_id, tt.country, tt.tok
        FROM (SELECT entity_id, country, unnest(name_tokens) AS tok FROM {t_sql}) tt
        JOIN allowed_tokens USING (country, tok)
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE candidates_by_rule AS
        SELECT source1_entity_id, candidate_id, rule FROM (
            SELECT s1.entity_id AS source1_entity_id, t.entity_id AS candidate_id,
                   'exact_name' AS rule
            FROM {s1_sql} s1
            JOIN {t_sql} t ON s1.country = t.country AND s1.name_normalized = t.name_normalized
            WHERE s1.name_normalized IS NOT NULL

            UNION ALL

            SELECT s1.entity_id, t.entity_id, 'core_name'
            FROM {s1_sql} s1
            JOIN {t_sql} t ON s1.country = t.country AND s1.name_core = t.name_core
            WHERE s1.name_core IS NOT NULL

            UNION ALL

            SELECT s1.entity_id, t.entity_id, 'sorted_tokens'
            FROM {s1_sql} s1
            JOIN {t_sql} t ON s1.country = t.country AND s1.name_sorted_tokens = t.name_sorted_tokens
            WHERE s1.name_sorted_tokens IS NOT NULL AND len(s1.name_sorted_tokens) > 0

            UNION ALL

            SELECT s1.entity_id, t.entity_id, 'postal_code'
            FROM {s1_sql} s1
            JOIN {t_sql} t ON s1.country = t.country AND s1.postal_code = t.postal_code
            WHERE s1.postal_code IS NOT NULL

            UNION ALL

            SELECT DISTINCT sa.source1_entity_id, ta.candidate_id, 'strong_token'
            FROM s1_tokens_allowed sa
            JOIN t_tokens_allowed ta USING (country, tok)
        )
    """)


def generate_candidates(
    con,
    target_key: str,
    s1_sample: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> dict:
    """Run all V1 rules for S1 -> target_key and write two Parquet files:
    - candidates_by_rule_<target_key>.parquet : (source1_entity_id, candidate_id, rule)
    - candidates_<target_key>.parquet         : deduplicated union, the set
      actually meant to be handed to a matching model.

    Returns a small dict of row counts for logging."""
    out_dir = out_dir or config.BLOCKING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    by_rule_path = out_dir / f"candidates_by_rule_{target_key}.parquet"
    union_path = out_dir / f"candidates_{target_key}.parquet"

    t0 = time.time()
    build_candidates_by_rule(con, target_key, s1_sample=s1_sample)
    build_elapsed = time.time() - t0

    con.execute(f"COPY candidates_by_rule TO '{by_rule_path}' (FORMAT PARQUET)")
    con.execute(f"""
        COPY (SELECT DISTINCT source1_entity_id, candidate_id FROM candidates_by_rule)
        TO '{union_path}' (FORMAT PARQUET)
    """)
    write_elapsed = time.time() - t0 - build_elapsed

    n_by_rule = con.execute("SELECT count(*) FROM candidates_by_rule").fetchone()[0]
    n_union = con.execute(f"SELECT count(*) FROM read_parquet('{union_path}')").fetchone()[0]

    return {
        "target_key": target_key,
        "by_rule_path": str(by_rule_path),
        "union_path": str(union_path),
        "n_by_rule_rows": n_by_rule,
        "n_union_pairs": n_union,
        "build_seconds": round(build_elapsed, 1),
        "write_seconds": round(write_elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Blocking V1 candidate generation")
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    parser.add_argument("--s1-sample", type=int, default=None,
                         help="Restrict S1 to a reservoir sample of N entities (sanity-test mode). "
                              "Target source is always used in full.")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    out_dir = Path(args.out_dir) if args.out_dir else None
    result = generate_candidates(con, args.target, s1_sample=args.s1_sample, out_dir=out_dir)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
