"""Blocking V2 combined evaluation: A=V1, B=V1+ngram, C=V1+guarded address,
D=V1+ngram+guarded address, plus overlap analysis between the ngram and
address additions and a heuristic categorization of remaining misses.

Everything here reuses the fixed 20K S1 reservoir sample from
blocking_v2_experiments.py against the full S2/S3 tables -- no production
V1 files are regenerated, and S2/S3 are always evaluated in separate
process invocations (see README.md / --target).
"""

import argparse
import time

from . import config, io_utils
from .blocking_v2_experiments import (
    build_s1_sample, build_gt_table, build_v1_candidates,
    build_ngram_rule, build_guarded_address_rule,
)

PREFIX = {"train_source2": "S2-", "train_source3": "S3-"}


def _pairs(con, table: str) -> int:
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _distribution_filled(con, table: str) -> dict:
    row = con.execute(f"""
        WITH per_entity AS (SELECT source1_entity_id, count(*) AS n FROM {table} GROUP BY source1_entity_id),
        filled AS (
            SELECT coalesce(pe.n, 0) AS n FROM s1_sample_rows s
            LEFT JOIN per_entity pe ON s.entity_id = pe.source1_entity_id
        )
        SELECT avg(n), quantile_cont(n,0.5), quantile_cont(n,0.95), max(n) FROM filled
    """).fetchone()
    return {"avg_per_entity": round(row[0], 3) if row[0] is not None else 0.0,
            "median_per_entity": row[1], "p95_per_entity": row[2], "max_per_entity": row[3]}


def _recall(con, table: str, prefix: str) -> dict:
    row = con.execute(f"""
        WITH gt AS (SELECT * FROM gt_exploded WHERE matched_id LIKE '{prefix}%')
        SELECT (SELECT count(*) FROM gt),
               (SELECT count(*) FROM gt g JOIN {table} c
                  ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id)
    """).fetchone()
    total, found = row
    return {"total_true_matches": total, "found": found, "recall": round(found/total, 4) if total else None}


def build_union(con, name: str, tables: list) -> None:
    parts = " UNION ".join(f"SELECT source1_entity_id, candidate_id FROM {t}" for t in tables)
    con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {parts}")


def evaluate_target(con, target_key: str, s1_sample: int) -> dict:
    prefix = PREFIX[target_key]
    build_s1_sample(con, s1_sample)
    build_gt_table(con)

    v1 = build_v1_candidates(con, target_key, s1_sample)
    ngram = build_ngram_rule(con, target_key, s1_sample, n=4, df_threshold=50)
    addr = build_guarded_address_rule(con, target_key, s1_sample)

    build_union(con, "cfg_A", [v1])
    build_union(con, "cfg_B", [v1, ngram])
    build_union(con, "cfg_C", [v1, addr])
    build_union(con, "cfg_D", [v1, ngram, addr])

    configs = {}
    v1_pairs = _pairs(con, "cfg_A")
    v1_recall = _recall(con, "cfg_A", prefix)
    for label, table in [("A_v1_baseline", "cfg_A"), ("B_v1_plus_ngram", "cfg_B"),
                          ("C_v1_plus_guarded_address", "cfg_C"), ("D_v1_plus_ngram_plus_address", "cfg_D")]:
        pairs = _pairs(con, table)
        recall = _recall(con, table, prefix)
        dist = _distribution_filled(con, table)
        additional_pairs = pairs - v1_pairs
        additional_matches = recall["found"] - v1_recall["found"]
        recall_per_million = round(additional_matches / (additional_pairs/1e6), 2) if additional_pairs > 0 else None
        configs[label] = {**recall, "total_pairs": pairs, "additional_pairs_over_v1": additional_pairs,
                           **dist, "true_matches_recovered_over_v1": additional_matches,
                           "recall_gained_per_million_additional_candidates": recall_per_million}

    # --- Step 3: overlap between ngram-only-recovered and addr-only-recovered ---
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_target AS
        SELECT * FROM gt_exploded WHERE matched_id LIKE '{prefix}%'
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE found_v1 AS
        SELECT g.source1_entity_id, g.matched_id FROM gt_target g
        JOIN cfg_A c ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE new_by_ngram AS
        (SELECT g.source1_entity_id, g.matched_id FROM gt_target g
         JOIN ngram_candidates c ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id)
        EXCEPT SELECT * FROM found_v1
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE new_by_addr AS
        (SELECT g.source1_entity_id, g.matched_id FROM gt_target g
         JOIN addr_candidates c ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id)
        EXCEPT SELECT * FROM found_v1
    """)
    n_ngram_only = con.execute("SELECT count(*) FROM (SELECT * FROM new_by_ngram EXCEPT SELECT * FROM new_by_addr)").fetchone()[0]
    n_addr_only = con.execute("SELECT count(*) FROM (SELECT * FROM new_by_addr EXCEPT SELECT * FROM new_by_ngram)").fetchone()[0]
    n_both = con.execute("SELECT count(*) FROM (SELECT * FROM new_by_ngram INTERSECT SELECT * FROM new_by_addr)").fetchone()[0]
    n_v1_misses = con.execute("SELECT count(*) FROM (SELECT * FROM gt_target EXCEPT SELECT * FROM found_v1)").fetchone()[0]
    n_still_missed = con.execute("""
        SELECT count(*) FROM (
            (SELECT * FROM gt_target EXCEPT SELECT * FROM found_v1)
            EXCEPT SELECT * FROM new_by_ngram
            EXCEPT SELECT * FROM new_by_addr
        )
    """).fetchone()[0]
    overlap = {
        "v1_misses_total": n_v1_misses,
        "recovered_by_ngram_only": n_ngram_only,
        "recovered_by_address_only": n_addr_only,
        "recovered_by_both": n_both,
        "still_missed_after_D": n_still_missed,
    }

    return {"target": target_key, "s1_sample": s1_sample, "configs": configs, "overlap": overlap}


# ---------------------------------------------------------------------------
# Step 4/5: categorize remaining (config D) misses
# ---------------------------------------------------------------------------
def categorize_misses(con, target_key: str, sample_limit: int = 500) -> dict:
    prefix = PREFIX[target_key]
    s1_path = config.NORMALIZED_FILES["train_source1"]
    t_path = config.NORMALIZED_FILES[target_key]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE missed_D AS
        SELECT g.source1_entity_id, g.matched_id FROM gt_target g
        LEFT JOIN cfg_D c ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id
        WHERE c.source1_entity_id IS NULL
    """)
    n_total_missed = con.execute("SELECT count(*) FROM missed_D").fetchone()[0]

    df = con.execute(f"""
        SELECT
          m.source1_entity_id, m.matched_id,
          s1.business_name AS s1_name, s1.name_core AS s1_core, s1.name_tokens AS s1_tok,
          s1.address_normalized AS s1_addr, s1.name_has_non_latin AS s1_nl,
          t.business_name AS t_name, t.name_core AS t_core, t.name_tokens AS t_tok,
          t.address_normalized AS t_addr, t.name_has_non_latin AS t_nl,
          CASE WHEN s1.name_core IS NOT NULL AND t.name_core IS NOT NULL
               THEN levenshtein(s1.name_core, t.name_core) ELSE NULL END AS core_edit_dist,
          CASE WHEN len(list_intersect(s1.name_tokens, t.name_tokens)) IS NOT NULL
               THEN len(list_intersect(s1.name_tokens, t.name_tokens))*1.0 /
                    greatest(len(list_distinct(list_concat(s1.name_tokens, t.name_tokens))), 1)
               ELSE 0 END AS token_jaccard
        FROM missed_D m
        JOIN read_parquet('{s1_path}') s1 ON s1.entity_id = m.source1_entity_id
        JOIN read_parquet('{t_path}') t ON t.entity_id = m.matched_id
        USING SAMPLE {sample_limit} (reservoir, 11)
    """).fetchdf()

    def categorize(row) -> str:
        if row.s1_nl != row.t_nl:
            return "cross_script"
        if row.s1_addr is None or row.t_addr is None:
            return "missing_address"
        if row.core_edit_dist is not None and 1 <= row.core_edit_dist <= 4:
            return "typo_ocr"
        if row.token_jaccard < 0.2:
            return "alias_dba"
        if row.token_jaccard >= 0.5:
            return "address_variation"
        return "other"

    df["category"] = df.apply(categorize, axis=1)
    counts = df["category"].value_counts().to_dict()
    examples = {cat: df[df.category == cat][["s1_name", "t_name"]].head(3).values.tolist()
                for cat in counts}

    return {
        "target": target_key,
        "n_total_missed_config_D": n_total_missed,
        "sample_size": len(df),
        "category_counts_in_sample": counts,
        "category_pct_in_sample": {k: round(100*v/len(df), 1) for k, v in counts.items()},
        "examples": examples,
    }


SCRIPTS = {
    "Devanagari": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF), "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF), "Odia": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF), "Malayalam": (0x0D00, 0x0D7F),
}


def inspect_cross_script_misses(con, target_key: str, limit: int = 10) -> dict:
    t_path = config.NORMALIZED_FILES[target_key]
    s1_path = config.NORMALIZED_FILES["train_source1"]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cross_script_missed AS
        SELECT m.source1_entity_id, m.matched_id, s1.business_name AS s1_name, t.business_name AS t_name
        FROM missed_D m
        JOIN read_parquet('{s1_path}') s1 ON s1.entity_id = m.source1_entity_id
        JOIN read_parquet('{t_path}') t ON t.entity_id = m.matched_id
        WHERE s1.name_has_non_latin != t.name_has_non_latin
    """)
    n_cross_script = con.execute("SELECT count(*) FROM cross_script_missed").fetchone()[0]

    script_counts = {}
    for script, (lo, hi) in SCRIPTS.items():
        pattern = f"[{chr(lo)}-{chr(hi)}]"
        n = con.execute(f"SELECT count(*) FROM cross_script_missed WHERE regexp_matches(t_name, '{pattern}')").fetchone()[0]
        if n > 0:
            script_counts[script] = n

    examples = con.execute("SELECT s1_name, t_name FROM cross_script_missed LIMIT ?", [limit]).fetchdf()
    return {"target": target_key, "n_cross_script_misses_in_D": n_cross_script,
            "scripts_involved": script_counts, "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Blocking V2 combined evaluation (A/B/C/D)")
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    parser.add_argument("--s1-sample", type=int, default=20000)
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    t0 = time.time()

    result = evaluate_target(con, args.target, args.s1_sample)
    print(f"\n{'='*100}\nSTEP 2: A/B/C/D configs -- {args.target}\n{'='*100}")
    for label, m in result["configs"].items():
        print(f"\n--- {label} ---")
        for k, v in m.items():
            print(f"  {k}: {v}")

    print(f"\n{'='*100}\nSTEP 3: overlap analysis -- {args.target}\n{'='*100}")
    for k, v in result["overlap"].items():
        print(f"  {k}: {v}")

    miss_cat = categorize_misses(con, args.target)
    print(f"\n{'='*100}\nSTEP 4: miss categorization (config D) -- {args.target}\n{'='*100}")
    print(f"  total missed under config D: {miss_cat['n_total_missed_config_D']:,} (sample size {miss_cat['sample_size']})")
    for cat, pct in sorted(miss_cat["category_pct_in_sample"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {miss_cat['category_counts_in_sample'][cat]} ({pct}%)")
        for ex in miss_cat["examples"][cat][:2]:
            print(f"      e.g. {ex}")

    cs = inspect_cross_script_misses(con, args.target)
    print(f"\n{'='*100}\nSTEP 5: cross-script misses remaining -- {args.target}\n{'='*100}")
    print(f"  n_cross_script_misses_in_D: {cs['n_cross_script_misses_in_D']:,}")
    print(f"  scripts_involved: {cs['scripts_involved']}")
    print(cs["examples"].to_string(index=False))

    print(f"\n[total time: {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
