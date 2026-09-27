"""V3 experiment: does adding a Devanagari transliteration representation to
blocking recover a meaningful share of the remaining cross-script true
matches, without candidate explosion?

Reuses the same fixed 20K S1 reservoir sample and full target tables as
blocking_v2_experiments.py / blocking_v2_combined.py. Nothing here is wired
into blocking.py; production V1 candidate files are never regenerated.

Configs evaluated (S2 and S3 separately, per the experiment instructions):
  A. V1
  B. V1 + ngram + guarded address          (the "V2 combined" result)
  C. V1 + transliterated exact/core-name blocking
  D. V1 + ngram + guarded address + transliterated blocking
"""

import argparse
import time

import pyarrow as pa

from . import config, io_utils
from .normalization import tokenize, strip_legal_suffix_tokens
from .transliteration import transliterate_devanagari
from .blocking_v2_experiments import (
    build_s1_sample, build_gt_table, build_v1_candidates,
    build_ngram_rule, build_guarded_address_rule,
)
from .blocking_v2_combined import _pairs, _distribution_filled, _recall, build_union

PREFIX = {"train_source2": "S2-", "train_source3": "S3-"}


# ---------------------------------------------------------------------------
# Step 2 (applied): build the transliterated representation for the target
# ---------------------------------------------------------------------------
def build_translit_table(con, target_key: str, df_threshold: int = None) -> str:
    """Pull ONLY the Devanagari-containing rows of the target source (a
    small subset -- ~2-6% of the table), transliterate them in Python, and
    register the result as a DuckDB temp table `translit_<target_key>` with
    columns (entity_id, country, translit_normalized, translit_core). Does
    NOT touch business_name/name_normalized/name_core -- this is a new,
    separate representation only used by this experiment.

    If df_threshold is given, composite keys (country, translit_normalized)
    or (country, translit_core) with target-side frequency above the
    threshold are excluded (Step 6 guard)."""
    t_path = config.NORMALIZED_FILES[target_key]
    devanagari_pattern = "[ऀ-ॿ]"
    df = con.execute(f"""
        SELECT entity_id, country, business_name
        FROM read_parquet('{t_path}')
        WHERE regexp_matches(business_name, '{devanagari_pattern}')
    """).fetchdf()

    translit_normalized = []
    translit_core = []
    for name in df["business_name"]:
        tn = transliterate_devanagari(name)
        translit_normalized.append(tn)
        if tn:
            core_tokens = strip_legal_suffix_tokens(tokenize(tn))
            translit_core.append(" ".join(core_tokens) if core_tokens else tn)
        else:
            translit_core.append(None)

    out = pa.table({
        "entity_id": df["entity_id"].tolist(),
        "country": df["country"].tolist(),
        "translit_normalized": translit_normalized,
        "translit_core": translit_core,
    })
    con.register("translit_arrow", out)
    table_name = f"translit_{target_key}"
    con.execute(f"CREATE OR REPLACE TEMP TABLE {table_name} AS SELECT * FROM translit_arrow")
    con.unregister("translit_arrow")

    # Frequency companion tables are cheap (this table is small, ~2-6% of
    # the full target) and are always built so build_translit_rule can
    # apply a df_threshold guard later even if it wasn't requested here.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table_name}_norm_df AS
        SELECT country, translit_normalized, count(*) AS df FROM {table_name}
        WHERE translit_normalized IS NOT NULL GROUP BY country, translit_normalized
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table_name}_core_df AS
        SELECT country, translit_core, count(*) AS df FROM {table_name}
        WHERE translit_core IS NOT NULL GROUP BY country, translit_core
    """)
    return table_name


def build_translit_rule(con, target_key: str, translit_table: str,
                         df_threshold: int = None) -> str:
    """country + (name_normalized = translit_normalized OR name_core =
    translit_core), i.e. S1's ordinary Latin fields against the target's
    transliterated Devanagari->Latin fields. Assumes build_s1_sample() has
    already populated s1_sample_rows on this connection."""
    s1_sql = "(SELECT * FROM s1_sample_rows)"

    norm_filter = ""
    core_filter = ""
    if df_threshold is not None:
        norm_filter = f"""
            JOIN {translit_table}_norm_df nd ON tt.country=nd.country AND tt.translit_normalized=nd.translit_normalized AND nd.df <= {df_threshold}
        """
        core_filter = f"""
            JOIN {translit_table}_core_df cd ON tt.country=cd.country AND tt.translit_core=cd.translit_core AND cd.df <= {df_threshold}
        """

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE translit_candidates AS
        SELECT DISTINCT source1_entity_id, candidate_id FROM (
            SELECT s1.entity_id AS source1_entity_id, tt.entity_id AS candidate_id
            FROM {s1_sql} s1
            JOIN {translit_table} tt ON s1.country = tt.country AND s1.name_normalized = tt.translit_normalized
            {norm_filter}
            WHERE tt.translit_normalized IS NOT NULL

            UNION ALL

            SELECT s1.entity_id, tt.entity_id
            FROM {s1_sql} s1
            JOIN {translit_table} tt ON s1.country = tt.country AND s1.name_core = tt.translit_core
            {core_filter}
            WHERE tt.translit_core IS NOT NULL
        )
    """)
    return "translit_candidates"


# ---------------------------------------------------------------------------
# Step 3: A/B/C/D evaluation
# ---------------------------------------------------------------------------
def evaluate_target(con, target_key: str, s1_sample: int, translit_df_threshold: int = None) -> dict:
    prefix = PREFIX[target_key]
    build_s1_sample(con, s1_sample)
    build_gt_table(con)

    v1 = build_v1_candidates(con, target_key, s1_sample)
    ngram = build_ngram_rule(con, target_key, s1_sample, n=4, df_threshold=50)
    addr = build_guarded_address_rule(con, target_key, s1_sample)
    translit_table = build_translit_table(con, target_key, df_threshold=translit_df_threshold)
    translit = build_translit_rule(con, target_key, translit_table, df_threshold=translit_df_threshold)

    build_union(con, "v3_cfg_A", [v1])
    build_union(con, "v3_cfg_B", [v1, ngram, addr])
    build_union(con, "v3_cfg_C", [v1, translit])
    build_union(con, "v3_cfg_D", [v1, ngram, addr, translit])

    v1_pairs = _pairs(con, "v3_cfg_A")
    v1_recall = _recall(con, "v3_cfg_A", prefix)

    configs = {}
    for label, table in [("A_v1", "v3_cfg_A"), ("B_v1_ngram_address", "v3_cfg_B"),
                          ("C_v1_translit", "v3_cfg_C"), ("D_v1_ngram_address_translit", "v3_cfg_D")]:
        pairs = _pairs(con, table)
        recall = _recall(con, table, prefix)
        dist = _distribution_filled(con, table)
        additional_pairs = pairs - v1_pairs
        additional_matches = recall["found"] - v1_recall["found"]
        recall_per_million = round(additional_matches / (additional_pairs/1e6), 2) if additional_pairs > 0 else None
        configs[label] = {**recall, "total_pairs": pairs, "additional_pairs_over_v1": additional_pairs,
                           **dist, "true_matches_recovered_over_v1": additional_matches,
                           "recall_gained_per_million_additional_candidates": recall_per_million}

    return {"target": target_key, "configs": configs, "translit_table": translit_table, "translit_rule": translit}


# ---------------------------------------------------------------------------
# Step 4: cross-script-specific recall
# ---------------------------------------------------------------------------
def cross_script_recall(con, target_key: str) -> dict:
    prefix = PREFIX[target_key]
    t_path = config.NORMALIZED_FILES[target_key]
    s1_path = config.NORMALIZED_FILES["train_source1"]
    indic_pattern = "[ऀ-ൿ]"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cross_script_gt AS
        SELECT g.source1_entity_id, g.matched_id
        FROM gt_exploded g
        JOIN {io_utils.parquet_sql(t_path)} t ON t.entity_id = g.matched_id
        WHERE g.matched_id LIKE '{prefix}%' AND regexp_matches(t.business_name, '{indic_pattern}')
    """)
    total = con.execute("SELECT count(*) FROM cross_script_gt").fetchone()[0]

    def captured_by(table):
        return con.execute(f"""
            SELECT count(*) FROM cross_script_gt g
            JOIN {table} c ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id
        """).fetchone()[0]

    by_v1_b = captured_by("v3_cfg_B")
    by_translit_alone = con.execute("""
        SELECT count(*) FROM cross_script_gt g
        JOIN translit_candidates c ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id
    """).fetchone()[0]
    by_d = captured_by("v3_cfg_D")

    return {
        "target": target_key,
        "total_cross_script_true_matches": total,
        "captured_by_v2_baseline_B": by_v1_b,
        "captured_by_translit_alone": by_translit_alone,
        "captured_by_D_combined": by_d,
        "still_missed_after_D": total - by_d,
    }


# ---------------------------------------------------------------------------
# Step 5: quality samples
# ---------------------------------------------------------------------------
def quality_samples(con, target_key: str, limit: int = 15):
    prefix = PREFIX[target_key]
    t_path = config.NORMALIZED_FILES[target_key]
    s1_path = config.NORMALIZED_FILES["train_source1"]
    translit_table = f"translit_{target_key}"

    df = con.execute(f"""
        SELECT g.source1_entity_id, s1.business_name AS s1_name,
               t.business_name AS t_name_original, tt.translit_normalized,
               CASE WHEN c.source1_entity_id IS NOT NULL THEN true ELSE false END AS captured
        FROM gt_exploded g
        JOIN {io_utils.parquet_sql(s1_path)} s1 ON s1.entity_id = g.source1_entity_id
        JOIN {io_utils.parquet_sql(t_path)} t ON t.entity_id = g.matched_id
        JOIN {translit_table} tt ON tt.entity_id = g.matched_id
        LEFT JOIN v3_cfg_D c ON g.source1_entity_id=c.source1_entity_id AND g.matched_id=c.candidate_id
        WHERE g.matched_id LIKE '{prefix}%'
        USING SAMPLE {limit} (reservoir, 21)
    """).fetchdf()
    return df


# ---------------------------------------------------------------------------
# Step 6: candidate explosion analysis on transliterated keys
# ---------------------------------------------------------------------------
def explosion_analysis(con, target_key: str, threshold: int = 50) -> dict:
    translit_table = f"translit_{target_key}"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ex_norm_df AS
        SELECT country, translit_normalized, count(*) AS df FROM {translit_table}
        WHERE translit_normalized IS NOT NULL GROUP BY country, translit_normalized
    """)
    n_keys = con.execute("SELECT count(*) FROM ex_norm_df").fetchone()[0]
    top = con.execute("SELECT translit_normalized, df FROM ex_norm_df ORDER BY df DESC LIMIT 5").fetchdf()
    n_over = con.execute(f"SELECT count(*), sum(df) FROM ex_norm_df WHERE df > {threshold}").fetchone()

    unguarded = build_translit_rule(con, target_key, translit_table, df_threshold=None)
    unguarded_pairs = _pairs(con, unguarded)
    unguarded_recall = _recall(con, unguarded, PREFIX[target_key])
    guarded = build_translit_rule(con, target_key, translit_table, df_threshold=threshold)
    guarded_pairs = _pairs(con, guarded)
    guarded_recall = _recall(con, guarded, PREFIX[target_key])
    guard_dist = con.execute(f"""
        SELECT max(n) FROM (SELECT source1_entity_id, count(*) AS n FROM {guarded} GROUP BY source1_entity_id)
    """).fetchone()[0]
    return {
        "target": target_key, "distinct_translit_normalized_keys": n_keys,
        "top_keys_by_frequency": top, "keys_over_threshold": n_over[0], "rows_covered_by_those_keys": n_over[1],
        "unguarded_pairs": unguarded_pairs, "unguarded_recall": unguarded_recall,
        "guarded_pairs": guarded_pairs, "guarded_recall": guarded_recall,
        "pairs_removed_by_guard": unguarded_pairs - guarded_pairs,
        "true_matches_lost_by_guard": unguarded_recall["found"] - guarded_recall["found"],
        "max_candidates_after_guard": guard_dist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 transliteration experiment")
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    parser.add_argument("--s1-sample", type=int, default=20000)
    parser.add_argument("--translit-df-threshold", type=int, default=None)
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    t0 = time.time()

    result = evaluate_target(con, args.target, args.s1_sample, translit_df_threshold=args.translit_df_threshold)
    print(f"\n{'='*100}\nSTEP 3: A/B/C/D -- {args.target}\n{'='*100}")
    for label, m in result["configs"].items():
        print(f"\n--- {label} ---")
        for k, v in m.items():
            print(f"  {k}: {v}")

    cs = cross_script_recall(con, args.target)
    print(f"\n{'='*100}\nSTEP 4: cross-script-specific recall -- {args.target}\n{'='*100}")
    for k, v in cs.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*100}\nSTEP 5: transliteration quality samples -- {args.target}\n{'='*100}")
    samples = quality_samples(con, args.target, limit=35)
    print(samples.to_string(index=False))

    print(f"\n{'='*100}\nSTEP 6: candidate explosion analysis -- {args.target}\n{'='*100}")
    exp = explosion_analysis(con, args.target, threshold=50)
    for k, v in exp.items():
        if k == "top_keys_by_frequency":
            print(f"  {k}:\n{v.to_string(index=False)}")
        else:
            print(f"  {k}: {v}")

    print(f"\n[total time: {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
