"""Blocking V2 experiments: independently evaluate candidate new blocking
rules against the V1 baseline, BEFORE deciding whether to fold any of them
into the real pipeline (see README.md "Blocking V2 experiments").

Every experiment here is measured on a fixed reservoir sample of Source1
(--s1-sample, default 20,000 entities) against the FULL target table, so
recall/candidate-volume comparisons are apples-to-apples across rules
without re-running the expensive full 2.2M-row V1 build multiple times.
Nothing here overwrites the real V1 candidate files in derived/blocking/.

Experiments:
  A. character n-gram blocking (typo/OCR tolerance)
  B. address-assisted blocking (postal_code / street_number / city_candidate
     combinations), for DBA/alias cases that share an address but not a name
  (Cross-script transliteration is analysis-only -- see blocking_v2_report.py
  / README.md; it is not a candidate-generation rule here.)
"""

import argparse
import time
from typing import Optional

from . import config, io_utils

S1_KEY = "train_source1"


def _sampled_sql(base_sql: str, n: Optional[int]) -> str:
    if n is None:
        return f"(SELECT * FROM {base_sql})"
    return f"(SELECT * FROM {base_sql} USING SAMPLE {n} (reservoir, 99))"


def _s1_sql(n: Optional[int]) -> str:
    """Reference the ONE materialized sample (see build_s1_sample) rather
    than re-sampling -- re-running `USING SAMPLE` per call, even with a
    fixed seed, is an easy way to accidentally end up with subtly different
    rows between the V1/new-rule/ground-truth-restriction queries."""
    if n is None:
        return f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[S1_KEY])})"
    return "(SELECT * FROM s1_sample_rows)"


def _target_sql(target_key: str) -> str:
    return f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[target_key])})"


def build_s1_sample(con, n: int) -> None:
    base = io_utils.parquet_sql(config.NORMALIZED_FILES[S1_KEY])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_sample_rows AS
        SELECT * FROM {base} USING SAMPLE {n} (reservoir, 99)
    """)


def build_gt_table(con, restrict_to_s1_sample: bool = True) -> None:
    """IMPORTANT: when evaluating on a sample of S1, ground truth must be
    restricted to that sample's source1_entity_ids. Otherwise recall is
    computed against ALL ~3.7-3.9M true matches in the full training set
    while candidates only exist for a handful of sampled entities, making
    every rule look like it has near-zero recall (a real bug caught while
    building this harness)."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_exploded_full AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{config.TRAIN_GROUND_TRUTH}', delim='\t', header=true, quote='', escape='', all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) != ''
    """)
    if restrict_to_s1_sample:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE gt_exploded AS
            SELECT g.* FROM gt_exploded_full g
            JOIN s1_sample_rows s ON s.entity_id = g.source1_entity_id
        """)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE gt_exploded AS SELECT * FROM gt_exploded_full")


def build_v1_candidates(con, target_key: str, s1_sample: int) -> str:
    """Recreate V1's five rules for the sample (mirrors blocking.py). Returns
    the temp table name holding the deduplicated V1 union for this sample."""
    s1_sql = _s1_sql(s1_sample)
    t_sql = _target_sql(target_key)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE v1_s1_token_df AS
        SELECT country, tok, count(*) AS df
        FROM (SELECT country, unnest(name_tokens) AS tok FROM {s1_sql}) GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE v1_t_token_df AS
        SELECT country, tok, count(*) AS df
        FROM (SELECT country, unnest(name_tokens) AS tok FROM {t_sql}) GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE v1_allowed_tokens AS
        SELECT s.country, s.tok FROM v1_s1_token_df s JOIN v1_t_token_df t USING (country, tok)
        WHERE length(s.tok) >= {config.TOKEN_MIN_LEN}
          AND s.df <= {config.TOKEN_DF_THRESHOLD} AND t.df <= {config.TOKEN_DF_THRESHOLD}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE v1_s1_tokens_allowed AS
        SELECT st.entity_id AS source1_entity_id, st.country, st.tok
        FROM (SELECT entity_id, country, unnest(name_tokens) AS tok FROM {s1_sql}) st
        JOIN v1_allowed_tokens USING (country, tok)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE v1_t_tokens_allowed AS
        SELECT tt.entity_id AS candidate_id, tt.country, tt.tok
        FROM (SELECT entity_id, country, unnest(name_tokens) AS tok FROM {t_sql}) tt
        JOIN v1_allowed_tokens USING (country, tok)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE v1_candidates AS
        SELECT DISTINCT source1_entity_id, candidate_id FROM (
            SELECT s1.entity_id AS source1_entity_id, t.entity_id AS candidate_id
            FROM {s1_sql} s1 JOIN {t_sql} t ON s1.country=t.country AND s1.name_normalized=t.name_normalized
            WHERE s1.name_normalized IS NOT NULL
            UNION ALL
            SELECT s1.entity_id, t.entity_id FROM {s1_sql} s1
            JOIN {t_sql} t ON s1.country=t.country AND s1.name_core=t.name_core
            WHERE s1.name_core IS NOT NULL
            UNION ALL
            SELECT s1.entity_id, t.entity_id FROM {s1_sql} s1
            JOIN {t_sql} t ON s1.country=t.country AND s1.name_sorted_tokens=t.name_sorted_tokens
            WHERE s1.name_sorted_tokens IS NOT NULL AND len(s1.name_sorted_tokens) > 0
            UNION ALL
            SELECT s1.entity_id, t.entity_id FROM {s1_sql} s1
            JOIN {t_sql} t ON s1.country=t.country AND s1.postal_code=t.postal_code
            WHERE s1.postal_code IS NOT NULL
            UNION ALL
            SELECT sa.source1_entity_id, ta.candidate_id
            FROM v1_s1_tokens_allowed sa JOIN v1_t_tokens_allowed ta USING (country, tok)
        )
    """)
    return "v1_candidates"


# ---------------------------------------------------------------------------
# Experiment A: character n-gram blocking
# ---------------------------------------------------------------------------
def build_ngram_rule(con, target_key: str, s1_sample: int, n: int = 4, df_threshold: int = 50) -> str:
    """Character n-grams of name_core, symmetric document-frequency filter
    (same design as V1's strong_token rule, applied to n-grams instead of
    word tokens). Returns the temp table name of (source1_entity_id,
    candidate_id) pairs for this rule alone."""
    s1_sql = _s1_sql(s1_sample)
    t_sql = _target_sql(target_key)

    def ngram_expr(col: str) -> str:
        return (f"unnest([substr({col}, i, {n}) FOR i IN "
                f"generate_series(1, greatest(length({col})-{n}+1, 0))])")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ng_s1_df AS
        SELECT country, tok, count(*) AS df FROM (
          SELECT country, {ngram_expr('name_core')} AS tok FROM {s1_sql}
          WHERE name_core IS NOT NULL AND length(name_core) >= {n}
        ) GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ng_t_df AS
        SELECT country, tok, count(*) AS df FROM (
          SELECT country, {ngram_expr('name_core')} AS tok FROM {t_sql}
          WHERE name_core IS NOT NULL AND length(name_core) >= {n}
        ) GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ng_allowed AS
        SELECT s.country, s.tok FROM ng_s1_df s JOIN ng_t_df t USING (country, tok)
        WHERE s.df <= {df_threshold} AND t.df <= {df_threshold}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ng_s1_allowed AS
        SELECT st.entity_id AS source1_entity_id, st.country, st.tok FROM (
          SELECT entity_id, country, {ngram_expr('name_core')} AS tok FROM {s1_sql}
          WHERE name_core IS NOT NULL AND length(name_core) >= {n}
        ) st
        JOIN ng_allowed USING (country, tok)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ng_t_allowed AS
        SELECT tt.entity_id AS candidate_id, tt.country, tt.tok FROM (
          SELECT entity_id, country, {ngram_expr('name_core')} AS tok FROM {t_sql}
          WHERE name_core IS NOT NULL AND length(name_core) >= {n}
        ) tt
        JOIN ng_allowed USING (country, tok)
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE ngram_candidates AS
        SELECT DISTINCT sa.source1_entity_id, ta.candidate_id
        FROM ng_s1_allowed sa JOIN ng_t_allowed ta USING (country, tok)
    """)
    return "ngram_candidates"


# ---------------------------------------------------------------------------
# Experiment B: address-assisted blocking
# ---------------------------------------------------------------------------
ADDRESS_RULES = {
    "postal_only": ["postal_code"],
    "postal_streetnum": ["postal_code", "street_number"],
    "city_streetnum": ["city_candidate", "street_number"],
}


def build_address_rule(con, target_key: str, s1_sample: int, fields: list) -> str:
    s1_sql = _s1_sql(s1_sample)
    t_sql = _target_sql(target_key)
    conds = " AND ".join(f"s1.{f} = t.{f}" for f in fields)
    not_null = " AND ".join(f"s1.{f} IS NOT NULL" for f in fields)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE addr_candidates AS
        SELECT DISTINCT s1.entity_id AS source1_entity_id, t.entity_id AS candidate_id
        FROM {s1_sql} s1 JOIN {t_sql} t ON s1.country = t.country AND {conds}
        WHERE {not_null}
    """)
    return "addr_candidates"


ADDRESS_GUARD_MIN_STREET_NUMBER_LEN = 3
ADDRESS_GUARD_COMPOSITE_DF_THRESHOLD = 50
# Chosen by measuring recall-lost vs. pairs-removed on the 20K sample
# (S1->S2): min_len=3, threshold=50 removes 93.5% of the raw rule's pairs
# (517,822 -> 33,834) while keeping 77% of its standalone recall (8,320 of
# 10,815 matches) and capping the max candidates for one entity at 50
# (down from 2,030 unguarded). min_len=4 was tested and found too
# aggressive (loses more than half the rule's recall for little extra
# pair reduction beyond min_len=3). See README.md "Blocking V2 combined".


def build_guarded_address_rule(
    con, target_key: str, s1_sample: int,
    min_street_number_len: int = ADDRESS_GUARD_MIN_STREET_NUMBER_LEN,
    composite_df_threshold: int = ADDRESS_GUARD_COMPOSITE_DF_THRESHOLD,
) -> str:
    """country + city_candidate + street_number, guarded against the
    pathological blow-up found in the raw version (e.g. "mumbai"+"1" alone
    produced 2,030 candidates for a single S1 entity). Both guards are
    data-driven, computed from the actual frequency distributions -- neither
    references any specific city or country:

    1. `min_street_number_len`: street_number length vs. average composite-key
       frequency (measured on train_source2/3) drops off sharply with length
       (~8,700 avg df at length 1, ~1,200 at length 2, ~280 at length 3, ~40
       at length 4) -- short numeric street numbers are structurally
       non-discriminating regardless of which city they're paired with.
    2. `composite_df_threshold`: the same symmetric document-frequency cap
       used for V1's strong_token rule, applied to the (country,
       city_candidate, street_number) composite key on both the S1 side and
       the target side.
    """
    s1_sql = _s1_sql(s1_sample)
    t_sql = _target_sql(target_key)
    len_filter = f"length(street_number) >= {min_street_number_len}"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE addr_s1_df AS
        SELECT country, city_candidate, street_number, count(*) AS df
        FROM {s1_sql}
        WHERE city_candidate IS NOT NULL AND street_number IS NOT NULL AND {len_filter}
        GROUP BY country, city_candidate, street_number
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE addr_t_df AS
        SELECT country, city_candidate, street_number, count(*) AS df
        FROM {t_sql}
        WHERE city_candidate IS NOT NULL AND street_number IS NOT NULL AND {len_filter}
        GROUP BY country, city_candidate, street_number
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE addr_allowed_keys AS
        SELECT s.country, s.city_candidate, s.street_number
        FROM addr_s1_df s JOIN addr_t_df t USING (country, city_candidate, street_number)
        WHERE s.df <= {composite_df_threshold} AND t.df <= {composite_df_threshold}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE addr_candidates AS
        SELECT DISTINCT s1.entity_id AS source1_entity_id, t.entity_id AS candidate_id
        FROM {s1_sql} s1
        JOIN addr_allowed_keys a ON s1.country=a.country AND s1.city_candidate=a.city_candidate AND s1.street_number=a.street_number
        JOIN {t_sql} t ON t.country=a.country AND t.city_candidate=a.city_candidate AND t.street_number=a.street_number
    """)
    return "addr_candidates"


# ---------------------------------------------------------------------------
# Shared evaluation harness
# ---------------------------------------------------------------------------
def _pairs_count(con, table: str) -> int:
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _distribution(con, table: str, n_s1_total: int) -> dict:
    row = con.execute(f"""
        SELECT avg(n), quantile_cont(n,0.5), quantile_cont(n,0.95), max(n)
        FROM (SELECT source1_entity_id, count(*) AS n FROM {table} GROUP BY source1_entity_id)
    """).fetchone()
    return {"avg_per_entity_with_cands": round(row[0], 2) if row[0] else 0,
            "median_per_entity_with_cands": row[1], "p95_per_entity_with_cands": row[2],
            "max_per_entity": row[3]}


def _recall(con, table: str, prefix: str) -> dict:
    row = con.execute(f"""
        WITH gt AS (SELECT * FROM gt_exploded WHERE matched_id LIKE '{prefix}%')
        SELECT (SELECT count(*) FROM gt),
               (SELECT count(*) FROM gt g JOIN {table} c
                  ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id)
    """).fetchone()
    total, found = row[0], row[1]
    return {"total_true_matches": total, "found": found,
            "recall": round(found/total, 4) if total else None}


def evaluate_addition(con, target_key: str, s1_sample: int, new_rule_table: str,
                       rule_label: str, example_limit: int = 5) -> dict:
    prefix = "S2-" if target_key == "train_source2" else "S3-"
    n_s1_total = s1_sample

    v1_table = build_v1_candidates(con, target_key, s1_sample)
    v1_recall = _recall(con, v1_table, prefix)
    v1_pairs = _pairs_count(con, v1_table)

    new_recall = _recall(con, new_rule_table, prefix)
    new_pairs = _pairs_count(con, new_rule_table)
    new_dist = _distribution(con, new_rule_table, n_s1_total)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE union_table AS
        SELECT source1_entity_id, candidate_id FROM {v1_table}
        UNION
        SELECT source1_entity_id, candidate_id FROM {new_rule_table}
    """)
    union_recall = _recall(con, "union_table", prefix)
    union_pairs = _pairs_count(con, "union_table")

    additional_candidates = union_pairs - v1_pairs
    additional_matches = union_recall["found"] - v1_recall["found"]
    recall_per_million = (additional_matches / (additional_candidates / 1e6)) if additional_candidates > 0 else None

    newly_recovered = con.execute(f"""
        WITH gt AS (SELECT * FROM gt_exploded WHERE matched_id LIKE '{prefix}%')
        SELECT g.source1_entity_id, s1.business_name AS s1_name, g.matched_id, t.business_name AS match_name
        FROM gt g
        JOIN {new_rule_table} nr ON g.source1_entity_id=nr.source1_entity_id AND g.matched_id=nr.candidate_id
        LEFT JOIN {v1_table} v1 ON g.source1_entity_id=v1.source1_entity_id AND g.matched_id=v1.candidate_id
        JOIN read_parquet('{config.NORMALIZED_FILES["train_source1"]}') s1 ON s1.entity_id=g.source1_entity_id
        JOIN read_parquet('{config.NORMALIZED_FILES[target_key]}') t ON t.entity_id=g.matched_id
        WHERE v1.source1_entity_id IS NULL
        LIMIT {example_limit}
    """).fetchdf()

    still_missed = con.execute(f"""
        WITH gt AS (SELECT * FROM gt_exploded WHERE matched_id LIKE '{prefix}%')
        SELECT g.source1_entity_id, s1.business_name AS s1_name, g.matched_id, t.business_name AS match_name
        FROM gt g
        LEFT JOIN union_table u ON g.source1_entity_id=u.source1_entity_id AND g.matched_id=u.candidate_id
        JOIN read_parquet('{config.NORMALIZED_FILES["train_source1"]}') s1 ON s1.entity_id=g.source1_entity_id
        JOIN read_parquet('{config.NORMALIZED_FILES[target_key]}') t ON t.entity_id=g.matched_id
        WHERE u.source1_entity_id IS NULL
        USING SAMPLE {example_limit} (reservoir, 7)
    """).fetchdf()

    result = {
        "target": target_key, "rule": rule_label, "s1_sample": s1_sample,
        "v1_baseline": {**v1_recall, "pairs": v1_pairs},
        "new_rule_alone": {**new_recall, "pairs": new_pairs, **new_dist},
        "v1_plus_new": {**union_recall, "pairs": union_pairs},
        "additional_candidates": additional_candidates,
        "additional_true_matches_recovered": additional_matches,
        "recall_added_per_million_additional_candidates": round(recall_per_million, 2) if recall_per_million else None,
    }
    print(f"\n{'='*90}\nEXPERIMENT: {rule_label} | target={target_key} | s1_sample={s1_sample}\n{'='*90}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"  newly recovered examples:\n{newly_recovered.to_string(index=False) if not newly_recovered.empty else '    (none)'}")
    print(f"  still-missed examples:\n{still_missed.to_string(index=False) if not still_missed.empty else '    (none!)'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Blocking V2 experiments (sample-based)")
    parser.add_argument("--experiment", required=True, choices=["ngram", "address_postal_only",
                                                                  "address_postal_streetnum", "address_city_streetnum"])
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    parser.add_argument("--s1-sample", type=int, default=20000)
    parser.add_argument("--ngram-n", type=int, default=4)
    parser.add_argument("--ngram-threshold", type=int, default=50)
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    t0 = time.time()
    build_s1_sample(con, args.s1_sample)
    build_gt_table(con)

    if args.experiment == "ngram":
        table = build_ngram_rule(con, args.target, args.s1_sample, n=args.ngram_n, df_threshold=args.ngram_threshold)
        label = f"ngram(n={args.ngram_n},thresh={args.ngram_threshold})"
    else:
        fields = ADDRESS_RULES[args.experiment.replace("address_", "")]
        table = build_address_rule(con, args.target, args.s1_sample, fields)
        label = f"address({'+'.join(fields)})"

    evaluate_addition(con, args.target, args.s1_sample, table, label)
    print(f"\n[total time: {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
