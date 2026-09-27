"""Evaluate Blocking V1 candidate sets against the training ground truth.

Reports, per rule and for the union of all rules, per target source
(train_source2 / train_source3 kept separate) and overall:
- coverage (entities with >=1 candidate)
- candidate-count distribution per S1 entity (avg/median/p95/max)
- candidate reduction ratio
- true-match recall against train_ground_truth.tsv
- missed-match examples

This module only reads existing Parquet/TSV files and writes nothing except
a printed report -- no candidate generation happens here (see blocking.py).
"""

import argparse
import time

from . import config, io_utils

RULES = ["exact_name", "core_name", "sorted_tokens", "strong_token", "postal_code"]
TARGETS = {"train_source2": "S2-", "train_source3": "S3-"}


def _n_rows(con, parquet_path) -> int:
    return con.execute(f"SELECT count(*) FROM read_parquet('{parquet_path}')").fetchone()[0]


def _candidate_distribution(con, pairs_sql: str, n_s1_total: int) -> dict:
    """pairs_sql must select (source1_entity_id, candidate_id), already
    deduplicated. Returns per-entity candidate-count stats computed over
    ALL n_s1_total S1 entities (entities with 0 candidates count as 0)."""
    row = con.execute(f"""
        WITH per_entity AS (
            SELECT source1_entity_id, count(*) AS n
            FROM ({pairs_sql})
            GROUP BY source1_entity_id
        ),
        filled AS (
            SELECT coalesce(pe.n, 0) AS n
            FROM (SELECT entity_id FROM read_parquet('{config.NORMALIZED_FILES["train_source1"]}')) s1
            LEFT JOIN per_entity pe ON s1.entity_id = pe.source1_entity_id
        )
        SELECT
            sum(CASE WHEN n > 0 THEN 1 ELSE 0 END) AS entities_with_candidates,
            sum(n) AS total_pairs,
            avg(n) AS avg_n,
            quantile_cont(n, 0.5) AS median_n,
            quantile_cont(n, 0.95) AS p95_n,
            max(n) AS max_n
        FROM filled
    """).fetchone()
    return {
        "entities_with_candidates": row[0],
        "total_candidate_pairs": row[1] or 0,
        "avg_per_entity": round(row[2], 3) if row[2] is not None else 0.0,
        "median_per_entity": row[3],
        "p95_per_entity": row[4],
        "max_per_entity": row[5],
    }


def _recall(con, pairs_sql: str, gt_table: str, prefix: str) -> dict:
    row = con.execute(f"""
        WITH gt AS (SELECT * FROM {gt_table} WHERE matched_id LIKE '{prefix}%'),
        found AS (
            SELECT g.source1_entity_id, g.matched_id
            FROM gt g
            JOIN ({pairs_sql}) c
              ON g.source1_entity_id = c.source1_entity_id AND g.matched_id = c.candidate_id
        )
        SELECT (SELECT count(*) FROM gt) AS total_gt,
               (SELECT count(*) FROM found) AS found
    """).fetchone()
    total_gt, found = row[0], row[1]
    return {
        "total_true_matches": total_gt,
        "found": found,
        "missed": total_gt - found,
        "recall": round(found / total_gt, 4) if total_gt else None,
    }


def _missed_examples(con, pairs_sql: str, gt_table: str, prefix: str, target_key: str, limit: int = 5):
    target_path = config.NORMALIZED_FILES[target_key]
    s1_path = config.NORMALIZED_FILES["train_source1"]
    df = con.execute(f"""
        WITH gt AS (SELECT * FROM {gt_table} WHERE matched_id LIKE '{prefix}%'),
        missed AS (
            SELECT g.source1_entity_id, g.matched_id
            FROM gt g
            LEFT JOIN ({pairs_sql}) c
              ON g.source1_entity_id = c.source1_entity_id AND g.matched_id = c.candidate_id
            WHERE c.source1_entity_id IS NULL
        )
        SELECT m.source1_entity_id, s1.business_name AS s1_name, s1.business_address AS s1_address,
               m.matched_id, t.business_name AS match_name, t.business_address AS match_address
        FROM missed m
        JOIN read_parquet('{s1_path}') s1 ON s1.entity_id = m.source1_entity_id
        JOIN read_parquet('{target_path}') t ON t.entity_id = m.matched_id
        USING SAMPLE {limit} (reservoir, 13)
    """).fetchdf()
    return df


def build_gt_table(con) -> None:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_exploded AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{config.TRAIN_GROUND_TRUTH}', delim='\t', header=true, quote='', escape='', all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) != ''
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE gt_entity_category AS
        SELECT source1_entity_id,
               max(CASE WHEN matched_id LIKE 'S2-%' THEN 1 ELSE 0 END) = 1 AS has_s2,
               max(CASE WHEN matched_id LIKE 'S3-%' THEN 1 ELSE 0 END) = 1 AS has_s3
        FROM gt_exploded
        GROUP BY source1_entity_id
    """)


def evaluate_target(con, target_key: str, n_s1_total: int, verbose: bool = True) -> dict:
    prefix = TARGETS[target_key]
    by_rule_path = config.BLOCKING_DIR / f"candidates_by_rule_{target_key}.parquet"
    union_path = config.BLOCKING_DIR / f"candidates_{target_key}.parquet"
    n_target_total = _n_rows(con, config.NORMALIZED_FILES[target_key])

    results = {"target": target_key, "n_target_total": n_target_total, "rules": {}}

    for rule in RULES:
        pairs_sql = f"SELECT DISTINCT source1_entity_id, candidate_id FROM read_parquet('{by_rule_path}') WHERE rule = '{rule}'"
        dist = _candidate_distribution(con, pairs_sql, n_s1_total)
        rec = _recall(con, pairs_sql, "gt_exploded", prefix)
        reduction = 1 - (dist["total_candidate_pairs"] / (n_s1_total * n_target_total))
        results["rules"][rule] = {**dist, **rec, "reduction_ratio": reduction}
        if verbose:
            print(f"\n--- {target_key} / rule={rule} ---")
            for k, v in results["rules"][rule].items():
                print(f"  {k}: {v}")

    union_sql = f"SELECT source1_entity_id, candidate_id FROM read_parquet('{union_path}')"
    dist = _candidate_distribution(con, union_sql, n_s1_total)
    rec = _recall(con, union_sql, "gt_exploded", prefix)
    reduction = 1 - (dist["total_candidate_pairs"] / (n_s1_total * n_target_total))
    results["union"] = {**dist, **rec, "reduction_ratio": reduction}
    if verbose:
        print(f"\n=== {target_key} / UNION of all rules ===")
        for k, v in results["union"].items():
            print(f"  {k}: {v}")
        print(f"\n--- {target_key} missed-match examples (union) ---")
        examples = _missed_examples(con, union_sql, "gt_exploded", prefix, target_key)
        print(examples.to_string(index=False) if not examples.empty else "  (no missed matches!)")

    return results


def evaluate_overall(con, n_s1_total: int) -> dict:
    s2_union_sql = f"SELECT source1_entity_id, candidate_id FROM read_parquet('{config.BLOCKING_DIR / 'candidates_train_source2.parquet'}')"
    s3_union_sql = f"SELECT source1_entity_id, candidate_id FROM read_parquet('{config.BLOCKING_DIR / 'candidates_train_source3.parquet'}')"

    total_gt = con.execute("SELECT count(*) FROM gt_exploded").fetchone()[0]
    s2_rec = _recall(con, s2_union_sql, "gt_exploded", "S2-")
    s3_rec = _recall(con, s3_union_sql, "gt_exploded", "S3-")
    overall_found = s2_rec["found"] + s3_rec["found"]
    overall_recall = round(overall_found / total_gt, 4)

    def segment_recall(condition: str) -> dict:
        # condition filters gt_entity_category; recall computed match-level
        # (found matches / total matches) within that entity segment, across
        # both S2 and S3 candidate sets as appropriate.
        row = con.execute(f"""
            WITH seg AS (SELECT source1_entity_id FROM gt_entity_category WHERE {condition}),
            gt_seg AS (SELECT g.* FROM gt_exploded g JOIN seg s USING (source1_entity_id)),
            found AS (
                SELECT g.source1_entity_id, g.matched_id
                FROM gt_seg g
                JOIN (
                    SELECT * FROM ({s2_union_sql})
                    UNION ALL
                    SELECT * FROM ({s3_union_sql})
                ) c ON g.source1_entity_id = c.source1_entity_id AND g.matched_id = c.candidate_id
            )
            SELECT (SELECT count(*) FROM gt_seg), (SELECT count(*) FROM found)
        """).fetchone()
        total, found = row[0], row[1]
        return {"total_true_matches": total, "found": found,
                "recall": round(found / total, 4) if total else None}

    only_s2 = segment_recall("has_s2 AND NOT has_s3")
    only_s3 = segment_recall("has_s3 AND NOT has_s2")
    both = segment_recall("has_s2 AND has_s3")

    # Singleton behavior: S1 entities with zero true matches. Report how many
    # still receive candidates (which the matching model must learn to reject)
    # and how large those candidate sets are.
    singleton_row = con.execute(f"""
        WITH singletons AS (
            SELECT entity_id AS source1_entity_id
            FROM read_parquet('{config.NORMALIZED_FILES["train_source1"]}')
            WHERE entity_id NOT IN (SELECT source1_entity_id FROM gt_exploded)
        ),
        combined AS (
            SELECT * FROM ({s2_union_sql})
            UNION ALL
            SELECT * FROM ({s3_union_sql})
        ),
        per_singleton AS (
            SELECT s.source1_entity_id, count(c.candidate_id) AS n
            FROM singletons s
            LEFT JOIN combined c ON s.source1_entity_id = c.source1_entity_id
            GROUP BY s.source1_entity_id
        )
        SELECT count(*) AS n_singletons,
               sum(CASE WHEN n > 0 THEN 1 ELSE 0 END) AS n_singletons_with_candidates,
               avg(n) AS avg_candidates,
               quantile_cont(n, 0.5) AS median_candidates,
               max(n) AS max_candidates
        FROM per_singleton
    """).fetchone()

    result = {
        "total_gt_matches": total_gt,
        "s2_recall": s2_rec,
        "s3_recall": s3_rec,
        "overall_blocking_recall": {"found": overall_found, "total": total_gt, "recall": overall_recall},
        "only_s2_entities_recall": only_s2,
        "only_s3_entities_recall": only_s3,
        "both_entities_recall": both,
        "singleton_behavior": {
            "n_singletons": singleton_row[0],
            "n_with_candidates": singleton_row[1],
            "pct_with_candidates": round(100 * singleton_row[1] / singleton_row[0], 2),
            "avg_candidates": round(singleton_row[2], 2),
            "median_candidates": singleton_row[3],
            "max_candidates": singleton_row[4],
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Blocking V1 recall against train ground truth")
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    t0 = time.time()
    build_gt_table(con)
    n_s1_total = _n_rows(con, config.NORMALIZED_FILES["train_source1"])
    print(f"n_s1_total: {n_s1_total:,}")

    for target_key in args.targets:
        evaluate_target(con, target_key, n_s1_total)

    if set(args.targets) == set(TARGETS):
        print("\n" + "=" * 100)
        print("OVERALL (S1 -> S2 union S3)")
        print("=" * 100)
        overall = evaluate_overall(con, n_s1_total)
        for k, v in overall.items():
            print(f"{k}: {v}")

    print(f"\n[blocking_eval total time: {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
