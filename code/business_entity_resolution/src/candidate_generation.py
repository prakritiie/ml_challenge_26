"""Production candidate generation: the ADOPTED blocking rule set for the
matching stage, combining V1's five base rules with the V2 and V5
additions validated by experimentation (see README.md "Adopted candidate
generation").

Adopted rules (10 total):
  V1 base (reused from derived/blocking/candidates_by_rule_train_source*.parquet,
  already computed at full scale -- NOT recomputed here):
    - exact_name, core_name, sorted_tokens, strong_token, postal_code
  V2 additions:
    - ngram            (character 4-gram of name_core, df<=50 symmetric)
    - guarded_city_street (city_candidate + street_number, guarded)
  V5 additions:
    - rare_address_token  (address token, df<=25 symmetric, full target population)
    - rare_address_ngram  (character 4-gram of address_normalized, df<=25 symmetric)
    - name_prefix         (first 6 chars of name_core, df<=100 symmetric)

This module builds the 5 NEW rules at full S1 scale (reusing the same
rule-building functions already validated in blocking_v2_experiments.py /
blocking_v5_experiments.py -- run here against the FULL train_source1 table
instead of the 20K sample), unions them with the existing V1 output, and
writes the combined by-rule and deduplicated candidate sets to
derived/matching/.
"""

import argparse
import time
from pathlib import Path

from . import config, io_utils
from .blocking_v2_experiments import build_ngram_rule, build_guarded_address_rule
from .blocking_v5_experiments import build_addr_token_shared, build_addr_ngram_shared, build_prefix_shared

MATCHING_DIR = config.DERIVED_ROOT / "matching"

ADOPTED_THRESHOLDS = {
    "ngram_df": 50,
    "rare_address_token_df": 25,
    "rare_address_ngram_df": 25,
    "name_prefix_len": 6,
    "name_prefix_df": 100,
}

V1_RULES = ["exact_name", "core_name", "sorted_tokens", "strong_token", "postal_code"]
NEW_RULES = ["ngram", "guarded_city_street", "rare_address_token", "rare_address_ngram", "name_prefix"]
ALL_RULES = V1_RULES + NEW_RULES


def build_s1_full(con) -> None:
    """Populate the s1_sample_rows temp table (referenced by the reused
    rule-builder functions) with the FULL train_source1 table, not a
    sample -- this is production candidate generation."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_sample_rows AS
        SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES['train_source1'])}
    """)


def build_new_rules(con, target_key: str) -> str:
    """Build the 5 NEW rules (beyond V1's existing 5) at full S1 scale,
    tagged with their rule name, as one (source1_entity_id, candidate_id,
    rule) temp table. Returns that table's name."""
    build_s1_full(con)

    t = ADOPTED_THRESHOLDS
    ngram_table = build_ngram_rule(con, target_key, s1_sample=1, n=4, df_threshold=t["ngram_df"])
    addr_table = build_guarded_address_rule(con, target_key, s1_sample=1)
    tok_shared = build_addr_token_shared(con, target_key, s1_sample=1, loose_thresh=t["rare_address_token_df"])
    ngram_addr_shared = build_addr_ngram_shared(con, target_key, s1_sample=1, n=4, loose_thresh=t["rare_address_ngram_df"])
    prefix_shared = build_prefix_shared(con, target_key, s1_sample=1, prefix_len=t["name_prefix_len"],
                                         loose_thresh=t["name_prefix_df"])

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE new_rules AS
        SELECT source1_entity_id, candidate_id, 'ngram' AS rule FROM {ngram_table}
        UNION ALL
        SELECT source1_entity_id, candidate_id, 'guarded_city_street' AS rule FROM {addr_table}
        UNION ALL
        SELECT DISTINCT source1_entity_id, candidate_id, 'rare_address_token' AS rule
        FROM {tok_shared} WHERE needed_thresh <= {t['rare_address_token_df']}
        UNION ALL
        SELECT DISTINCT source1_entity_id, candidate_id, 'rare_address_ngram' AS rule
        FROM {ngram_addr_shared} WHERE needed_thresh <= {t['rare_address_ngram_df']}
        UNION ALL
        SELECT DISTINCT source1_entity_id, candidate_id, 'name_prefix' AS rule
        FROM {prefix_shared} WHERE needed_thresh <= {t['name_prefix_df']}
    """)
    return "new_rules"


def generate_adopted_candidates(con, target_key: str, out_dir: Path = None) -> dict:
    """Build the full 10-rule candidate set for S1 -> target_key and write:
    - candidates_by_rule_<target>.parquet : (source1_entity_id, candidate_id, rule), all 10 rules
    - candidates_<target>.parquet         : deduplicated union (the actual candidate set)
    """
    out_dir = out_dir or MATCHING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    by_rule_path = out_dir / f"candidates_by_rule_{target_key}.parquet"
    union_path = out_dir / f"candidates_{target_key}.parquet"

    v1_by_rule_path = config.BLOCKING_DIR / f"candidates_by_rule_{target_key}.parquet"

    t0 = time.time()
    build_new_rules(con, target_key)
    new_rules_elapsed = time.time() - t0

    t1 = time.time()
    con.execute(f"""
        COPY (
            SELECT source1_entity_id, candidate_id, rule FROM read_parquet('{v1_by_rule_path}')
            UNION ALL
            SELECT source1_entity_id, candidate_id, rule FROM new_rules
        ) TO '{by_rule_path}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT DISTINCT source1_entity_id, candidate_id FROM read_parquet('{by_rule_path}')
        ) TO '{union_path}' (FORMAT PARQUET)
    """)
    write_elapsed = time.time() - t1

    n_by_rule = con.execute(f"SELECT count(*) FROM read_parquet('{by_rule_path}')").fetchone()[0]
    n_union = con.execute(f"SELECT count(*) FROM read_parquet('{union_path}')").fetchone()[0]

    return {
        "target_key": target_key,
        "by_rule_path": str(by_rule_path),
        "union_path": str(union_path),
        "n_by_rule_rows": n_by_rule,
        "n_union_pairs": n_union,
        "new_rules_build_seconds": round(new_rules_elapsed, 1),
        "write_seconds": round(write_elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adopted (V2+V5) candidate generation")
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    result = generate_adopted_candidates(con, args.target)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
