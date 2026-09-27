"""V5 controlled blocking experiments: rare address token blocking (A),
rare address 4-gram blocking (B), name-prefix blocking (C), and a combined
incremental test (D) of the best candidate from each family.

Nothing here modifies blocking.py. All experiments use the fixed 20K S1
reservoir sample against full S2/S3 target tables, evaluated as an addition
on top of V2 (V1 + ngram + guarded address). Target-side document frequency
is always computed from the FULL target table (never the 20K sample).

Efficiency note: rather than rebuilding the token/n-gram join once per
threshold (6 thresholds x 2 rule families x 2 targets = 24 rebuilds), each
family is joined ONCE at the loosest requested threshold (a safe pre-filter
so the join itself never touches an unfiltered high-frequency key -- see
profiling in README.md "V5 experiments"), retaining the per-pair minimum
qualifying threshold (`needed_thresh`). Every tighter threshold is then a
cheap `WHERE needed_thresh <= T` re-slice of that same table.
"""

import argparse
import time
from typing import Optional

from . import config, io_utils
from .blocking_v2_experiments import build_s1_sample, build_gt_table, build_v1_candidates, build_ngram_rule, build_guarded_address_rule
from .blocking_v2_combined import build_union, _pairs, _distribution_filled, _recall, PREFIX

LOOSEST_THRESHOLD = 1000
THRESHOLDS = [25, 50, 100, 200, 500, 1000]


def build_v2_baseline(con, target_key: str, s1_sample: int) -> str:
    build_s1_sample(con, s1_sample)
    build_gt_table(con)
    v1 = build_v1_candidates(con, target_key, s1_sample)
    ngram = build_ngram_rule(con, target_key, s1_sample, n=4, df_threshold=50)
    addr = build_guarded_address_rule(con, target_key, s1_sample)
    build_union(con, "v5_v2_baseline", [v1, ngram, addr])
    return "v5_v2_baseline"


# ---------------------------------------------------------------------------
# EXPERIMENT A: rare address token blocking
# ---------------------------------------------------------------------------
def build_addr_token_shared(con, target_key: str, s1_sample: int, loose_thresh: int = LOOSEST_THRESHOLD) -> str:
    s1_sql = "(SELECT * FROM s1_sample_rows)"
    t_sql = f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[target_key])})"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE a_s1_df AS
        SELECT country, tok, count(*) AS df
        FROM (SELECT country, unnest(address_tokens) AS tok FROM {s1_sql})
        GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE a_t_df AS
        SELECT country, tok, count(*) AS df
        FROM (SELECT country, unnest(address_tokens) AS tok FROM {t_sql})
        GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE a_allowed AS
        SELECT s.country, s.tok, greatest(s.df, t.df) AS needed_thresh
        FROM a_s1_df s JOIN a_t_df t USING (country, tok)
        WHERE s.df <= {loose_thresh} AND t.df <= {loose_thresh}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE a_s1_tok AS
        SELECT st.entity_id AS source1_entity_id, st.country, st.tok
        FROM (SELECT entity_id, country, unnest(address_tokens) AS tok FROM {s1_sql}) st
        JOIN a_allowed USING (country, tok)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE a_t_tok AS
        SELECT tt.entity_id AS candidate_id, tt.country, tt.tok
        FROM (SELECT entity_id, country, unnest(address_tokens) AS tok FROM {t_sql}) tt
        JOIN a_allowed USING (country, tok)
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE addr_token_shared AS
        SELECT s.source1_entity_id, t.candidate_id, min(a.needed_thresh) AS needed_thresh
        FROM a_s1_tok s
        JOIN a_t_tok t USING (country, tok)
        JOIN a_allowed a ON a.country = s.country AND a.tok = s.tok
        GROUP BY s.source1_entity_id, t.candidate_id
    """)
    return "addr_token_shared"


# ---------------------------------------------------------------------------
# EXPERIMENT B: rare address 4-gram blocking
# ---------------------------------------------------------------------------
def build_addr_ngram_shared(con, target_key: str, s1_sample: int, n: int = 4,
                             loose_thresh: int = LOOSEST_THRESHOLD) -> str:
    s1_sql = "(SELECT * FROM s1_sample_rows)"
    t_sql = f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[target_key])})"

    def ngram_expr(col):
        return (f"unnest([substr({col}, i, {n}) FOR i IN "
                f"generate_series(1, greatest(length({col})-{n}+1, 0))])")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE b_s1_df AS
        SELECT country, tok, count(*) AS df FROM (
          SELECT country, {ngram_expr('address_normalized')} AS tok FROM {s1_sql}
          WHERE address_normalized IS NOT NULL AND length(address_normalized) >= {n}
        ) GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE b_t_df AS
        SELECT country, tok, count(*) AS df FROM (
          SELECT country, {ngram_expr('address_normalized')} AS tok FROM {t_sql}
          WHERE address_normalized IS NOT NULL AND length(address_normalized) >= {n}
        ) GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE b_allowed AS
        SELECT s.country, s.tok, greatest(s.df, t.df) AS needed_thresh
        FROM b_s1_df s JOIN b_t_df t USING (country, tok)
        WHERE s.df <= {loose_thresh} AND t.df <= {loose_thresh}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE b_s1_ng AS
        SELECT st.entity_id AS source1_entity_id, st.country, st.tok FROM (
          SELECT entity_id, country, {ngram_expr('address_normalized')} AS tok FROM {s1_sql}
          WHERE address_normalized IS NOT NULL AND length(address_normalized) >= {n}
        ) st JOIN b_allowed USING (country, tok)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE b_t_ng AS
        SELECT tt.entity_id AS candidate_id, tt.country, tt.tok FROM (
          SELECT entity_id, country, {ngram_expr('address_normalized')} AS tok FROM {t_sql}
          WHERE address_normalized IS NOT NULL AND length(address_normalized) >= {n}
        ) tt JOIN b_allowed USING (country, tok)
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE addr_ngram_shared AS
        SELECT s.source1_entity_id, t.candidate_id, min(a.needed_thresh) AS needed_thresh
        FROM b_s1_ng s
        JOIN b_t_ng t USING (country, tok)
        JOIN b_allowed a ON a.country = s.country AND a.tok = s.tok
        GROUP BY s.source1_entity_id, t.candidate_id
    """)
    return "addr_ngram_shared"


# ---------------------------------------------------------------------------
# EXPERIMENT C: name-prefix blocking
# ---------------------------------------------------------------------------
def build_prefix_shared(con, target_key: str, s1_sample: int, prefix_len: int,
                         loose_thresh: int = 5000) -> str:
    """Exact match on the first `prefix_len` characters of name_core. Unlike
    A/B (many keys per record), a record has exactly ONE prefix, so no
    per-pair min-threshold reslicing is needed -- we just keep the raw
    (country, prefix, df_s1, df_t) alongside each candidate pair and filter
    per threshold at evaluation time.

    SAFETY: short prefixes (e.g. length 3) can have target-side frequency in
    the hundreds of thousands (very low cardinality -- only 26^3 possible
    3-letter combinations). The entity-level join is therefore pre-filtered
    to prefixes with frequency <= loose_thresh on BOTH sides BEFORE joining
    S1 to the target table, exactly like experiments A/B -- an earlier
    version of this function skipped that pre-filter and hung/exploded on
    prefix_len=3 (see README.md "V5 experiments" for the incident)."""
    s1_sql = "(SELECT * FROM s1_sample_rows)"
    t_sql = f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[target_key])})"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE c_s1_df AS
        SELECT country, substr(name_core, 1, {prefix_len}) AS prefix, count(*) AS df
        FROM {s1_sql} WHERE name_core IS NOT NULL AND length(name_core) >= {prefix_len}
        GROUP BY country, prefix
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE c_t_df AS
        SELECT country, substr(name_core, 1, {prefix_len}) AS prefix, count(*) AS df
        FROM {t_sql} WHERE name_core IS NOT NULL AND length(name_core) >= {prefix_len}
        GROUP BY country, prefix
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE c_allowed AS
        SELECT s.country, s.prefix, s.df AS s1_df, t.df AS t_df, greatest(s.df, t.df) AS needed_thresh
        FROM c_s1_df s JOIN c_t_df t USING (country, prefix)
        WHERE s.df <= {loose_thresh} AND t.df <= {loose_thresh}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE prefix_shared AS
        SELECT s1.entity_id AS source1_entity_id, t.entity_id AS candidate_id, a.needed_thresh
        FROM {s1_sql} s1
        JOIN c_allowed a ON a.country = s1.country AND a.prefix = substr(s1.name_core, 1, {prefix_len})
        JOIN {t_sql} t ON t.country = a.country AND substr(t.name_core, 1, {prefix_len}) = a.prefix
    """)
    return "prefix_shared"


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------
def evaluate_threshold(con, shared_table: str, threshold: Optional[int], v2_table: str, prefix: str,
                        key_count_table: Optional[str] = None) -> dict:
    where = f"WHERE needed_thresh <= {threshold}" if threshold is not None else ""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rule_candidates AS
        SELECT DISTINCT source1_entity_id, candidate_id FROM {shared_table} {where}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE combined AS
        SELECT source1_entity_id, candidate_id FROM {v2_table}
        UNION
        SELECT source1_entity_id, candidate_id FROM rule_candidates
    """)
    v2_pairs = _pairs(con, v2_table)
    v2_recall = _recall(con, v2_table, prefix)
    rule_pairs = _pairs(con, "rule_candidates")
    combined_pairs = _pairs(con, "combined")
    combined_recall = _recall(con, "combined", prefix)
    dist = _distribution_filled(con, "combined")

    additional_pairs = combined_pairs - v2_pairs
    additional_matches = combined_recall["found"] - v2_recall["found"]
    recall_per_million = round(additional_matches / (additional_pairs / 1e6), 2) if additional_pairs > 0 else None
    n_keys = None
    if key_count_table is not None:
        thr_clause = f"WHERE needed_thresh <= {threshold}" if threshold is not None else ""
        n_keys = con.execute(f"SELECT count(*) FROM {key_count_table} {thr_clause}").fetchone()[0]

    return {
        "threshold": threshold,
        "rule_alone_pairs": rule_pairs,
        **combined_recall,
        "total_pairs": combined_pairs,
        "additional_pairs_over_v2": additional_pairs,
        **dist,
        "true_matches_recovered_over_v2": additional_matches,
        "recall_gained_per_million_additional_candidates": recall_per_million,
        "n_keys_used": n_keys,
        "misses_remaining": v2_recall["total_true_matches"] - combined_recall["found"],
    }


def run_experiment_grid(con, target_key: str, s1_sample: int, builder, key_table: str, thresholds=THRESHOLDS) -> list:
    prefix = PREFIX[target_key]
    v2 = build_v2_baseline(con, target_key, s1_sample)
    shared = builder(con, target_key, s1_sample)
    results = []
    for t in thresholds:
        r = evaluate_threshold(con, shared, t, v2, prefix, key_count_table=key_table)
        results.append(r)
        print(f"  threshold={t}: recall={r['recall']:.4f}, total_pairs={r['total_pairs']:,}, "
              f"+pairs={r['additional_pairs_over_v2']:,}, +matches={r['true_matches_recovered_over_v2']}, "
              f"recall/M={r['recall_gained_per_million_additional_candidates']}, "
              f"avg/med/p95/max={r['avg_per_entity']}/{r['median_per_entity']}/{r['p95_per_entity']}/{r['max_per_entity']}, "
              f"n_keys={r['n_keys_used']}, misses_remaining={r['misses_remaining']}")
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT D: combined incremental test of the best candidate per family
# ---------------------------------------------------------------------------
def run_experiment_d(con, target_key: str, s1_sample: int,
                      a_thresh: int = 25, b_thresh: int = 25,
                      c_prefix_len: int = 6, c_thresh: int = 100) -> dict:
    prefix = PREFIX[target_key]
    v2 = build_v2_baseline(con, target_key, s1_sample)
    v2_pairs = _pairs(con, v2)
    v2_recall = _recall(con, v2, prefix)

    a_shared = build_addr_token_shared(con, target_key, s1_sample, loose_thresh=max(a_thresh, LOOSEST_THRESHOLD))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE d_a AS
        SELECT DISTINCT source1_entity_id, candidate_id FROM {a_shared} WHERE needed_thresh <= {a_thresh}
    """)
    b_shared = build_addr_ngram_shared(con, target_key, s1_sample, n=4, loose_thresh=max(b_thresh, LOOSEST_THRESHOLD))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE d_b AS
        SELECT DISTINCT source1_entity_id, candidate_id FROM {b_shared} WHERE needed_thresh <= {b_thresh}
    """)
    c_shared = build_prefix_shared(con, target_key, s1_sample, c_prefix_len, loose_thresh=max(c_thresh, 5000))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE d_c AS
        SELECT DISTINCT source1_entity_id, candidate_id FROM {c_shared} WHERE needed_thresh <= {c_thresh}
    """)

    def eval_combo(name, tables):
        build_union(con, f"d_combo_{name}", [v2] + tables)
        pairs = _pairs(con, f"d_combo_{name}")
        recall = _recall(con, f"d_combo_{name}", prefix)
        dist = _distribution_filled(con, f"d_combo_{name}")
        add_pairs = pairs - v2_pairs
        add_matches = recall["found"] - v2_recall["found"]
        eff = round(add_matches / (add_pairs / 1e6), 2) if add_pairs > 0 else None
        return {**recall, "total_pairs": pairs, "additional_pairs_over_v2": add_pairs, **dist,
                "true_matches_recovered_over_v2": add_matches, "recall_gained_per_million_additional_candidates": eff}

    configs = {
        "V2_only": {**v2_recall, "total_pairs": v2_pairs, "additional_pairs_over_v2": 0, "true_matches_recovered_over_v2": 0},
        "V2+addr_token": eval_combo("a", ["d_a"]),
        "V2+addr_4gram": eval_combo("b", ["d_b"]),
        "V2+prefix": eval_combo("c", ["d_c"]),
        "V2+addr_token+addr_4gram+prefix": eval_combo("all", ["d_a", "d_b", "d_c"]),
    }

    # Overlap: among matches newly recovered (not in V2), which rule(s) found them?
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE d_gt_target AS SELECT * FROM gt_exploded WHERE matched_id LIKE '{prefix}%'
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE d_found_v2 AS
        SELECT g.source1_entity_id, g.matched_id FROM d_gt_target g
        JOIN v5_v2_baseline c ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id
    """)

    def new_by(table):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE d_new_{table} AS
            (SELECT g.source1_entity_id, g.matched_id FROM d_gt_target g
             JOIN {table} c ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id)
            EXCEPT SELECT * FROM d_found_v2
        """)
    new_by("d_a")
    new_by("d_b")
    new_by("d_c")

    def count(sql):
        return con.execute(sql).fetchone()[0]

    only_a = count("SELECT count(*) FROM (SELECT * FROM d_new_d_a EXCEPT SELECT * FROM d_new_d_b EXCEPT SELECT * FROM d_new_d_c)")
    only_b = count("SELECT count(*) FROM (SELECT * FROM d_new_d_b EXCEPT SELECT * FROM d_new_d_a EXCEPT SELECT * FROM d_new_d_c)")
    only_c = count("SELECT count(*) FROM (SELECT * FROM d_new_d_c EXCEPT SELECT * FROM d_new_d_a EXCEPT SELECT * FROM d_new_d_b)")
    multi = count("""
        SELECT count(*) FROM (
            SELECT * FROM d_new_d_a WHERE (source1_entity_id, matched_id) IN (SELECT source1_entity_id, matched_id FROM d_new_d_b)
               OR (source1_entity_id, matched_id) IN (SELECT source1_entity_id, matched_id FROM d_new_d_c)
        )
    """)
    total_new_any = count("""
        SELECT count(*) FROM (
            SELECT * FROM d_new_d_a UNION SELECT * FROM d_new_d_b UNION SELECT * FROM d_new_d_c
        )
    """)

    overlap = {
        "recovered_only_by_addr_token": only_a,
        "recovered_only_by_addr_4gram": only_b,
        "recovered_only_by_prefix": only_c,
        "recovered_by_multiple_rules": multi,
        "total_newly_recovered_by_any": total_new_any,
        "v2_misses_remaining_after_all": v2_recall["total_true_matches"] - v2_recall["found"] - total_new_any,
    }

    print(f"\n=== EXPERIMENT D: combined -- {target_key} ===")
    for label, m in configs.items():
        print(f"\n--- {label} ---")
        for k, v in m.items():
            print(f"  {k}: {v}")
    print("\n--- overlap ---")
    for k, v in overlap.items():
        print(f"  {k}: {v}")

    return {"configs": configs, "overlap": overlap}


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 blocking experiments")
    parser.add_argument("--experiment", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    parser.add_argument("--s1-sample", type=int, default=20000)
    parser.add_argument("--prefix-len", type=int, default=None, help="Required for experiment C")
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    t0 = time.time()

    if args.experiment == "A":
        print(f"\n=== EXPERIMENT A: rare address token blocking -- {args.target} ===")
        run_experiment_grid(con, args.target, args.s1_sample, build_addr_token_shared, key_table="a_allowed")
    elif args.experiment == "B":
        print(f"\n=== EXPERIMENT B: rare address 4-gram blocking -- {args.target} ===")
        run_experiment_grid(con, args.target, args.s1_sample, build_addr_ngram_shared, key_table="b_allowed")
    elif args.experiment == "C":
        if args.prefix_len is None:
            raise SystemExit("--prefix-len is required for experiment C")
        print(f"\n=== EXPERIMENT C: name-prefix(len={args.prefix_len}) blocking -- {args.target} ===")
        prefix = PREFIX[args.target]
        v2 = build_v2_baseline(con, args.target, args.s1_sample)
        shared = build_prefix_shared(con, args.target, args.s1_sample, args.prefix_len)

        # Unguarded (no cap) -- to surface pathological prefixes
        top = con.execute("SELECT prefix, s1_df, t_df FROM c_allowed ORDER BY t_df DESC LIMIT 10").fetchdf()
        print("  top prefixes by target-side frequency (unguarded):")
        print(top.to_string(index=False))
        r_unguarded = evaluate_threshold(con, shared, None, v2, prefix, key_count_table="c_allowed")
        print(f"  UNGUARDED: recall={r_unguarded['recall']:.4f}, total_pairs={r_unguarded['total_pairs']:,}, "
              f"+pairs={r_unguarded['additional_pairs_over_v2']:,}, +matches={r_unguarded['true_matches_recovered_over_v2']}, "
              f"max_per_entity={r_unguarded['max_per_entity']}")

        for t in [100, 500, 1000, 5000]:
            r = evaluate_threshold(con, shared, t, v2, prefix, key_count_table="c_allowed")
            print(f"  threshold={t}: recall={r['recall']:.4f}, total_pairs={r['total_pairs']:,}, "
                  f"+pairs={r['additional_pairs_over_v2']:,}, +matches={r['true_matches_recovered_over_v2']}, "
                  f"recall/M={r['recall_gained_per_million_additional_candidates']}, "
                  f"avg/med/p95/max={r['avg_per_entity']}/{r['median_per_entity']}/{r['p95_per_entity']}/{r['max_per_entity']}")

    elif args.experiment == "D":
        run_experiment_d(con, args.target, args.s1_sample)

    print(f"\n[total time: {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
