"""Training-pair construction for the matching stage: ground-truth
labeling, entity-level train/validation split, and feature engineering on
the ADOPTED candidate set (see candidate_generation.py). No model is
trained here.

Everything is implemented as DuckDB SQL against Parquet files -- the full
candidate set is never loaded into pandas. See README.md "Training-pair
construction" for the full report, including why negatives are downsampled
before feature computation (candidate volume is too large -- ~90-100M rows
per target -- to compute per-pair string-similarity features on every
negative when only a stratified sample is needed for training).
"""

import argparse
import time
from pathlib import Path

from . import config, io_utils
from .candidate_generation import MATCHING_DIR, ALL_RULES

SPLIT_TRAIN_FRAC = 0.8
NEGATIVE_SAMPLE_RATIO = 10  # negatives per positive, after entity-level split
RANDOM_SEED = 42

PREFIX = {"train_source2": "S2-", "train_source3": "S3-"}


# ---------------------------------------------------------------------------
# STEP 3: ground-truth labeling
# ---------------------------------------------------------------------------
def build_gt_full(con) -> None:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_exploded_full AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{config.TRAIN_GROUND_TRUTH}', delim='\t', header=true, quote='', escape='', all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) != ''
    """)


def label_candidates(con, target_key: str) -> dict:
    """Label every candidate pair for target_key: 1 if it is a known
    ground-truth match, 0 otherwise. Reports blocking recall (captured vs.
    missed) using the REAL full-scale candidate set, not a sample."""
    prefix = PREFIX[target_key]
    union_path = MATCHING_DIR / f"candidates_{target_key}.parquet"

    build_gt_full(con)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_target AS
        SELECT * FROM gt_exploded_full WHERE matched_id LIKE '{prefix}%'
    """)
    total_gt = con.execute("SELECT count(*) FROM gt_target").fetchone()[0]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE labeled_{target_key} AS
        SELECT c.source1_entity_id, c.candidate_id,
               CASE WHEN g.matched_id IS NOT NULL THEN 1 ELSE 0 END AS label
        FROM read_parquet('{union_path}') c
        LEFT JOIN gt_target g ON c.source1_entity_id = g.source1_entity_id AND c.candidate_id = g.matched_id
    """)

    n_pos = con.execute(f"SELECT count(*) FROM labeled_{target_key} WHERE label=1").fetchone()[0]
    n_neg = con.execute(f"SELECT count(*) FROM labeled_{target_key} WHERE label=0").fetchone()[0]
    captured = n_pos  # every positive IN the candidate set is, by construction, a captured true match
    missed = total_gt - captured

    return {
        "target": target_key,
        "total_gt_matches": total_gt,
        "captured_by_blocking": captured,
        "missed_by_blocking": missed,
        "blocking_recall": round(captured / total_gt, 4) if total_gt else None,
        "n_positive_candidates": n_pos,
        "n_negative_candidates": n_neg,
        "positive_rate": round(n_pos / (n_pos + n_neg), 6) if (n_pos + n_neg) else None,
    }


# ---------------------------------------------------------------------------
# STEP 4: entity-level train/validation split
# ---------------------------------------------------------------------------
def build_entity_split(con, train_frac: float = SPLIT_TRAIN_FRAC) -> dict:
    """Split S1 ENTITIES (not candidate rows) into train/validation via a
    deterministic hash, so every candidate pair belonging to one S1 entity
    stays entirely in one split -- no entity ever appears in both."""
    s1_path = config.NORMALIZED_FILES["train_source1"]
    cutoff = int(train_frac * 100)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_split AS
        SELECT entity_id AS source1_entity_id,
               CASE WHEN abs(hash(entity_id)) % 100 < {cutoff} THEN 'train' ELSE 'validation' END AS split
        FROM {io_utils.parquet_sql(s1_path)}
    """)
    counts = con.execute("SELECT split, count(*) AS n FROM s1_split GROUP BY split").fetchdf()
    return dict(zip(counts["split"], counts["n"]))


# ---------------------------------------------------------------------------
# STEP 5: feature engineering
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # name features
    "exact_normalized_name", "exact_core_name", "exact_sorted_tokens",
    "name_token_jaccard", "name_token_overlap_count",
    "name_char_ngram_jaccard", "name_edit_distance", "name_edit_similarity",
    "name_length_diff", "name_token_count_diff",
    # address features
    "exact_normalized_address", "address_token_jaccard", "address_token_overlap_count",
    "postal_code_equal", "street_number_equal", "city_equal", "state_equal",
    "address_length_diff",
    # metadata
    "country_equal", "target_name_has_non_latin", "target_address_has_non_latin",
    # blocking provenance (10 rules)
    "blocked_exact_name", "blocked_core_name", "blocked_sorted_tokens", "blocked_strong_token",
    "blocked_ngram", "blocked_postal", "blocked_city_street",
    "blocked_rare_address_token", "blocked_address_ngram", "blocked_name_prefix",
]

# ---------------------------------------------------------------------------
# STEP 6 -> model feature cleanup (found during feature validation, see
# README.md "Training-pair construction" / "Model training stage"). Applied
# by SELECTING columns at read time -- the underlying feature Parquet files
# are NOT regenerated, since every removed column is either provably
# constant or a provable exact duplicate of a kept column.
# ---------------------------------------------------------------------------
REMOVED_FEATURES = {
    "country_equal": (
        "Constant (=True for 100% of rows on both S1->S2 and S1->S3). Every "
        "one of the 10 blocking rules joins ON s1.country = t.country, so no "
        "candidate pair can ever have mismatched countries -- zero predictive "
        "value, verified via feature_validation (n_distinct=1 on both targets)."
    ),
    "blocked_exact_name": (
        "Exact duplicate of exact_normalized_name -- the 'exact_name' "
        "blocking rule's join condition IS `s1.name_normalized = "
        "t.name_normalized`, the same boolean the feature recomputes "
        "independently. Confirmed identical positive/negative class means "
        "in feature_validation."
    ),
    "blocked_core_name": (
        "Exact duplicate of exact_core_name, same reasoning (the "
        "'core_name' rule's join condition is `s1.name_core = t.name_core`)."
    ),
    "blocked_sorted_tokens": (
        "Exact duplicate of exact_sorted_tokens, same reasoning (the "
        "'sorted_tokens' rule's join condition is `s1.name_sorted_tokens = "
        "t.name_sorted_tokens`)."
    ),
}

# The cleaned, explicit feature list actually used for model training.
# 31 raw features - 4 removed (1 constant + 3 duplicates) = 27.
MODEL_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in REMOVED_FEATURES]

ID_COLUMNS = ["source1_entity_id", "candidate_id"]
LABEL_COLUMN = "label"
SPLIT_COLUMN = "split"


def _ngram_expr(col: str, n: int = 4) -> str:
    return f"[substr({col}, i, {n}) FOR i IN generate_series(1, greatest(length({col})-{n}+1, 0))]"


def sample_negatives(con, target_key: str, ratio: int = NEGATIVE_SAMPLE_RATIO, seed: int = RANDOM_SEED) -> str:
    """Keep ALL positives and a random sample of negatives (ratio per
    positive), so downstream per-pair feature computation (string
    similarity, edit distance, n-gram Jaccard) runs on a training-sized
    dataset instead of every one of the ~90-100M raw candidate pairs."""
    n_pos = con.execute(f"SELECT count(*) FROM labeled_{target_key} WHERE label=1").fetchone()[0]
    n_neg_target = n_pos * ratio
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sampled_{target_key} AS
        SELECT * FROM labeled_{target_key} WHERE label = 1
        UNION ALL
        SELECT * FROM (
            SELECT * FROM labeled_{target_key} WHERE label = 0
            USING SAMPLE {n_neg_target} (reservoir, {seed})
        )
    """)
    return f"sampled_{target_key}"


def feature_select_expressions(s1_alias: str = "s1", t_alias: str = "t", prov_alias: str = "p") -> str:
    """The 31 raw feature expressions (SELECT-list fragment, no leading/
    trailing comma), parametrized by table alias so it can be reused
    against different FROM/JOIN shapes -- e.g. the downsampled training
    table here, and the full undownsampled validation-entity candidate set
    in train_baseline.py's evaluation path. Keeping this as the single
    source of truth avoids the two paths' features silently drifting apart."""
    s1, t, p = s1_alias, t_alias, prov_alias
    return f"""
            ({s1}.name_normalized IS NOT NULL AND {s1}.name_normalized = {t}.name_normalized) AS exact_normalized_name,
            ({s1}.name_core IS NOT NULL AND {s1}.name_core = {t}.name_core) AS exact_core_name,
            ({s1}.name_sorted_tokens IS NOT NULL AND {s1}.name_sorted_tokens = {t}.name_sorted_tokens) AS exact_sorted_tokens,

            CASE WHEN {s1}.name_tokens IS NOT NULL AND {t}.name_tokens IS NOT NULL
                      AND len({s1}.name_tokens) > 0 AND len({t}.name_tokens) > 0
                 THEN len(list_intersect({s1}.name_tokens, {t}.name_tokens))*1.0 /
                      greatest(len(list_distinct(list_concat({s1}.name_tokens, {t}.name_tokens))), 1)
                 ELSE 0 END AS name_token_jaccard,
            CASE WHEN {s1}.name_tokens IS NOT NULL AND {t}.name_tokens IS NOT NULL
                 THEN len(list_intersect({s1}.name_tokens, {t}.name_tokens)) ELSE 0 END AS name_token_overlap_count,

            CASE WHEN {s1}.name_core IS NOT NULL AND {t}.name_core IS NOT NULL
                      AND length({s1}.name_core) >= 4 AND length({t}.name_core) >= 4
                 THEN len(list_intersect({_ngram_expr(f'{s1}.name_core')}, {_ngram_expr(f'{t}.name_core')}))*1.0 /
                      greatest(len(list_distinct(list_concat({_ngram_expr(f'{s1}.name_core')}, {_ngram_expr(f'{t}.name_core')}))), 1)
                 ELSE 0 END AS name_char_ngram_jaccard,

            CASE WHEN {s1}.name_core IS NOT NULL AND {t}.name_core IS NOT NULL
                 THEN levenshtein({s1}.name_core, {t}.name_core) ELSE NULL END AS name_edit_distance,
            CASE WHEN {s1}.name_core IS NOT NULL AND {t}.name_core IS NOT NULL
                 THEN 1.0 - levenshtein({s1}.name_core, {t}.name_core)*1.0 /
                      greatest(length({s1}.name_core), length({t}.name_core), 1)
                 ELSE NULL END AS name_edit_similarity,

            abs(coalesce(length({s1}.name_normalized),0) - coalesce(length({t}.name_normalized),0)) AS name_length_diff,
            abs(coalesce(len({s1}.name_tokens),0) - coalesce(len({t}.name_tokens),0)) AS name_token_count_diff,

            ({s1}.address_normalized IS NOT NULL AND {s1}.address_normalized = {t}.address_normalized) AS exact_normalized_address,
            CASE WHEN {s1}.address_tokens IS NOT NULL AND {t}.address_tokens IS NOT NULL
                      AND len({s1}.address_tokens) > 0 AND len({t}.address_tokens) > 0
                 THEN len(list_intersect({s1}.address_tokens, {t}.address_tokens))*1.0 /
                      greatest(len(list_distinct(list_concat({s1}.address_tokens, {t}.address_tokens))), 1)
                 ELSE 0 END AS address_token_jaccard,
            CASE WHEN {s1}.address_tokens IS NOT NULL AND {t}.address_tokens IS NOT NULL
                 THEN len(list_intersect({s1}.address_tokens, {t}.address_tokens)) ELSE 0 END AS address_token_overlap_count,

            ({s1}.postal_code IS NOT NULL AND {t}.postal_code IS NOT NULL AND {s1}.postal_code = {t}.postal_code) AS postal_code_equal,
            ({s1}.street_number IS NOT NULL AND {t}.street_number IS NOT NULL AND {s1}.street_number = {t}.street_number) AS street_number_equal,
            ({s1}.city_candidate IS NOT NULL AND {t}.city_candidate IS NOT NULL AND {s1}.city_candidate = {t}.city_candidate) AS city_equal,
            ({s1}.state_candidate IS NOT NULL AND {t}.state_candidate IS NOT NULL AND {s1}.state_candidate = {t}.state_candidate) AS state_equal,
            abs(coalesce(length({s1}.address_normalized),0) - coalesce(length({t}.address_normalized),0)) AS address_length_diff,

            ({s1}.country = {t}.country) AS country_equal,
            {t}.name_has_non_latin AS target_name_has_non_latin,
            {t}.address_has_non_latin AS target_address_has_non_latin,

            coalesce({p}.blocked_exact_name, false) AS blocked_exact_name,
            coalesce({p}.blocked_core_name, false) AS blocked_core_name,
            coalesce({p}.blocked_sorted_tokens, false) AS blocked_sorted_tokens,
            coalesce({p}.blocked_strong_token, false) AS blocked_strong_token,
            coalesce({p}.blocked_ngram, false) AS blocked_ngram,
            coalesce({p}.blocked_postal, false) AS blocked_postal,
            coalesce({p}.blocked_city_street, false) AS blocked_city_street,
            coalesce({p}.blocked_rare_address_token, false) AS blocked_rare_address_token,
            coalesce({p}.blocked_address_ngram, false) AS blocked_address_ngram,
            coalesce({p}.blocked_name_prefix, false) AS blocked_name_prefix
    """


def build_provenance_table(con, by_rule_source_sql: str, out_table: str) -> None:
    """(source1_entity_id, candidate_id) -> 10 blocked_* boolean flags,
    aggregated from a by-rule relation. `by_rule_source_sql` should already
    be filtered down to only the pairs needed (see train_baseline.py's
    validation path, which filters by entity FIRST for speed rather than
    using the tuple-IN semi-join below on the full table)."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {out_table} AS
        SELECT source1_entity_id, candidate_id,
               bool_or(rule = 'exact_name') AS blocked_exact_name,
               bool_or(rule = 'core_name') AS blocked_core_name,
               bool_or(rule = 'sorted_tokens') AS blocked_sorted_tokens,
               bool_or(rule = 'strong_token') AS blocked_strong_token,
               bool_or(rule = 'ngram') AS blocked_ngram,
               bool_or(rule = 'postal_code') AS blocked_postal,
               bool_or(rule = 'guarded_city_street') AS blocked_city_street,
               bool_or(rule = 'rare_address_token') AS blocked_rare_address_token,
               bool_or(rule = 'rare_address_ngram') AS blocked_address_ngram,
               bool_or(rule = 'name_prefix') AS blocked_name_prefix
        FROM {by_rule_source_sql}
        GROUP BY source1_entity_id, candidate_id
    """)


def build_features(con, target_key: str, sampled_table: str) -> dict:
    s1_path = config.NORMALIZED_FILES["train_source1"]
    t_path = config.NORMALIZED_FILES[target_key]
    by_rule_path = MATCHING_DIR / f"candidates_by_rule_{target_key}.parquet"
    out_path = MATCHING_DIR / f"features_{target_key}.parquet"

    build_provenance_table(
        con,
        by_rule_source_sql=(
            f"read_parquet('{by_rule_path}') "
            f"WHERE (source1_entity_id, candidate_id) IN (SELECT source1_entity_id, candidate_id FROM {sampled_table})"
        ),
        out_table=f"provenance_{target_key}",
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE features_{target_key} AS
        SELECT
            s.source1_entity_id, s.candidate_id, s.label, sp.split,
            {feature_select_expressions('s1', 't', 'p')}
        FROM {sampled_table} s
        JOIN {io_utils.parquet_sql(s1_path)} s1 ON s1.entity_id = s.source1_entity_id
        JOIN {io_utils.parquet_sql(t_path)} t ON t.entity_id = s.candidate_id
        JOIN s1_split sp ON sp.source1_entity_id = s.source1_entity_id
        LEFT JOIN provenance_{target_key} p ON p.source1_entity_id = s.source1_entity_id AND p.candidate_id = s.candidate_id
    """)
    con.execute(f"COPY features_{target_key} TO '{out_path}' (FORMAT PARQUET)")
    n_rows = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
    return {"target": target_key, "features_path": str(out_path), "n_rows": n_rows,
            "n_features": len(FEATURE_COLUMNS)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-pair construction (labeling, split, features)")
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_SAMPLE_RATIO)
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    t0 = time.time()

    label_stats = label_candidates(con, args.target)
    print("=== STEP 3: labeling ===")
    for k, v in label_stats.items():
        print(f"  {k}: {v}")

    split_counts = build_entity_split(con)
    print("\n=== STEP 4: entity-level split ===")
    print(f"  {split_counts}")

    sampled = sample_negatives(con, args.target, ratio=args.negative_ratio)
    n_sampled = con.execute(f"SELECT count(*) FROM {sampled}").fetchone()[0]
    print(f"\n=== negative sampling (ratio={args.negative_ratio}) ===")
    print(f"  sampled rows: {n_sampled:,}")

    t1 = time.time()
    feat_stats = build_features(con, args.target, sampled)
    feat_elapsed = time.time() - t1
    print("\n=== STEP 5: features ===")
    for k, v in feat_stats.items():
        print(f"  {k}: {v}")
    print(f"  feature build time: {feat_elapsed:.1f}s")

    print(f"\n[total time: {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
