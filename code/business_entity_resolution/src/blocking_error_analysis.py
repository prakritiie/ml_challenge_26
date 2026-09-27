"""INVESTIGATION ONLY -- error analysis of the ADOPTED (V2+V5, 10-rule)
production blocking's remaining validation-split misses, and offline
testing of candidate new blocking signals to close that gap.

Does NOT modify candidate_generation.py, blocking.py, normalization.py, any
features/*.parquet training file, or any saved LightGBM model. All new
temp tables/relations here live only in this module's DuckDB connection.

Scope: VALIDATION-split S1 entities only (441,566 of 2,206,821 total S1
entities, the same split used by train_baseline.py), so every number here
is directly comparable to the already-reported S2/S3 candidate-ceiling
figures (81.47% / 81.45%), which were computed on that same scope. Rule
experiments run the candidate-side join at full target-table scale (S2:
5,034,616 rows, S3: 5,285,603 rows) -- only the S1 side is restricted to
validation, purely for computational tractability; a production adoption
decision would need to re-verify pair-volume estimates at full S1 scale
(2,206,821 entities) before shipping, which this module does not do.

Sections (mirrors the 5-part request this was built for):
  1. blocking_miss_summary       -- validation GT captured/missed by PRODUCTION candidates
  2. fetch_missed_df / classify  -- pull missed pairs, classify into categories
  3. rule builders E/F/I/J       -- new candidate blocking signals not already adopted
                                     (A/B/C/D/G/H from the original brief are ALREADY
                                     production rules -- rare_address_token, rare_address_ngram,
                                     name_prefix, guarded_city_street, ngram(name char-4gram),
                                     rare_address_ngram respectively -- so they are not
                                     re-tested as "new"; see README.md for the mapping)
  4. evaluate_rule               -- added pairs / recovered matches / recall gain / per-S1 load
  5. main()                      -- runs everything for one target and prints the report
"""

import argparse
import time

import pandas as pd

from . import config, io_utils
from .matching_dataset import build_gt_full, build_entity_split, PREFIX
from .candidate_generation import MATCHING_DIR

EXAMPLE_LIMIT = 8


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
def build_validation_s1(con) -> int:
    build_entity_split(con)  # -> temp table s1_split(source1_entity_id, split)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE val_s1 AS
        SELECT s1.* FROM {io_utils.parquet_sql(config.NORMALIZED_FILES['train_source1'])} s1
        JOIN s1_split sp ON sp.source1_entity_id = s1.entity_id AND sp.split = 'validation'
    """)
    return con.execute("SELECT count(*) FROM val_s1").fetchone()[0]


# ---------------------------------------------------------------------------
# PART 1: blocking-miss summary (validation GT vs. PRODUCTION candidates)
# ---------------------------------------------------------------------------
def blocking_miss_summary(con, target_key: str) -> dict:
    prefix = PREFIX[target_key]
    union_path = MATCHING_DIR / f"candidates_{target_key}.parquet"
    build_gt_full(con)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE val_gt_{target_key} AS
        SELECT g.source1_entity_id, g.matched_id
        FROM gt_exploded_full g
        JOIN val_s1 v ON v.entity_id = g.source1_entity_id
        WHERE g.matched_id LIKE '{prefix}%'
    """)
    total_gt = con.execute(f"SELECT count(*) FROM val_gt_{target_key}").fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE val_missed_{target_key} AS
        SELECT g.source1_entity_id, g.matched_id
        FROM val_gt_{target_key} g
        LEFT JOIN read_parquet('{union_path}') c
            ON c.source1_entity_id = g.source1_entity_id AND c.candidate_id = g.matched_id
        WHERE c.source1_entity_id IS NULL
    """)
    missed = con.execute(f"SELECT count(*) FROM val_missed_{target_key}").fetchone()[0]
    captured = total_gt - missed
    return {
        "target": target_key,
        "total_validation_gt_matches": total_gt,
        "captured_by_blocking": captured,
        "missed_by_blocking": missed,
        "miss_pct": round(100 * missed / total_gt, 2) if total_gt else None,
        "blocking_recall": round(captured / total_gt, 4) if total_gt else None,
    }


# ---------------------------------------------------------------------------
# PART 2: classify missed matches
# ---------------------------------------------------------------------------
MISSED_FIELDS_SQL = """
    m.source1_entity_id, m.matched_id,
    s1.name_normalized AS s1_norm, s1.name_core AS s1_core,
    s1.name_tokens AS s1_ntok, s1.address_normalized AS s1_addr, s1.address_tokens AS s1_atok,
    s1.postal_code AS s1_postal, s1.street_number AS s1_sn, s1.city_candidate AS s1_city,
    s1.state_candidate AS s1_state, s1.name_has_non_latin AS s1_nl, s1.address_has_non_latin AS s1_anl,
    t.name_normalized AS t_norm, t.name_core AS t_core,
    t.name_tokens AS t_ntok, t.address_normalized AS t_addr, t.address_tokens AS t_atok,
    t.postal_code AS t_postal, t.street_number AS t_sn, t.city_candidate AS t_city,
    t.state_candidate AS t_state, t.name_has_non_latin AS t_nl, t.address_has_non_latin AS t_anl
"""


def fetch_missed_df(con, target_key: str) -> pd.DataFrame:
    s1_path = config.NORMALIZED_FILES["train_source1"]
    t_path = config.NORMALIZED_FILES[target_key]
    return con.execute(f"""
        SELECT {MISSED_FIELDS_SQL}
        FROM val_missed_{target_key} m
        JOIN {io_utils.parquet_sql(s1_path)} s1 ON s1.entity_id = m.source1_entity_id
        JOIN {io_utils.parquet_sql(t_path)} t ON t.entity_id = m.matched_id
    """).fetchdf()


def _nonempty(x) -> bool:
    return x is not None and len(x) > 0


def _jaccard(a, b) -> float:
    if not _nonempty(a) or not _nonempty(b):
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


DBA_MARKERS = ["dba", "d b a", "trading as", "aka", "a k a", "formerly", "fka"]

CATEGORY_ORDER = ["cross_script", "missing_address", "typo_ocr", "address_variation", "alias_dba", "other"]


def classify_row(row) -> str:
    s1_nl = bool(row.s1_nl) if pd.notna(row.s1_nl) else False
    t_nl = bool(row.t_nl) if pd.notna(row.t_nl) else False
    if s1_nl != t_nl:
        return "cross_script"

    s1_addr, t_addr = row.s1_addr, row.t_addr
    missing_addr = not s1_addr or not t_addr or pd.isna(s1_addr) or pd.isna(t_addr)

    s1_core, t_core = row.s1_core, row.t_core
    both_core = bool(s1_core) and bool(t_core) and pd.notna(s1_core) and pd.notna(t_core)
    ed_sim = None
    if both_core:
        ed = _levenshtein(s1_core, t_core)
        ed_sim = 1.0 - ed / max(len(s1_core), len(t_core), 1)

    if missing_addr:
        # a near-exact name match with only address missing is still primarily
        # a "missing address" miss (no address signal could ever have fired)
        return "missing_address"

    addr_jac = _jaccard(row.s1_atok, row.t_atok)

    if both_core and ed_sim is not None and 0 < (1 - ed_sim) and ed_sim >= 0.75:
        return "typo_ocr"

    if addr_jac >= 0.4:
        return "address_variation"

    name_tok_jac = _jaccard(row.s1_ntok, row.t_ntok)
    has_dba = any(m in (row.s1_norm or "") or m in (row.t_norm or "") for m in DBA_MARKERS)
    s1_in_t = both_core and s1_core in t_core
    t_in_s1 = both_core and t_core in s1_core
    if has_dba or s1_in_t or t_in_s1 or name_tok_jac < 0.2:
        return "alias_dba"

    return "other"


def categorize_missed(df: pd.DataFrame) -> tuple:
    if df.empty:
        return pd.Series(dtype=str), {}
    cats = df.apply(classify_row, axis=1)
    counts = cats.value_counts()
    total = len(df)
    table = {c: {"count": int(counts.get(c, 0)), "pct": round(100 * counts.get(c, 0) / total, 2)}
             for c in CATEGORY_ORDER}
    return cats, table


EXAMPLE_COLS = ["source1_entity_id", "matched_id", "s1_norm", "t_norm", "s1_addr", "t_addr",
                 "s1_postal", "t_postal", "s1_nl", "t_nl"]


def representative_examples(df: pd.DataFrame, cats: pd.Series, n: int = 3) -> dict:
    out = {}
    for c in CATEGORY_ORDER:
        subset = df[cats == c]
        if subset.empty:
            out[c] = []
            continue
        sample = subset.sample(min(n, len(subset)), random_state=7)[EXAMPLE_COLS]
        out[c] = sample.to_dict("records")
    return out


# ---------------------------------------------------------------------------
# PART 3: new candidate rule builders (E, F, I, J -- see module docstring for
# why A/B/C/D/G/H are skipped: already in production)
# ---------------------------------------------------------------------------
def build_rule_E_postal_prefix_name_token(con, target_key: str, prefix_len: int = 3, df_threshold: int = 50) -> str:
    """E: postal-code PREFIX (tolerates ZIP+4 / minor postal differences,
    unlike the already-adopted EXACT postal_code rule) + a shared name token,
    as a compound key guarded by symmetric document frequency."""
    t_sql = f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[target_key])})"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE e_s1_keys AS
        SELECT entity_id AS source1_entity_id, country, substr(postal_code,1,{prefix_len}) AS pfx, unnest(name_tokens) AS tok
        FROM val_s1 WHERE postal_code IS NOT NULL AND length(postal_code) >= {prefix_len}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE e_t_keys AS
        SELECT entity_id AS candidate_id, country, substr(postal_code,1,{prefix_len}) AS pfx, unnest(name_tokens) AS tok
        FROM {t_sql} WHERE postal_code IS NOT NULL AND length(postal_code) >= {prefix_len}
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE e_s1_df AS
        SELECT country, pfx, tok, count(*) AS df FROM e_s1_keys WHERE length(tok) >= 3 GROUP BY country, pfx, tok
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE e_t_df AS
        SELECT country, pfx, tok, count(*) AS df FROM e_t_keys WHERE length(tok) >= 3 GROUP BY country, pfx, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE e_allowed AS
        SELECT s.country, s.pfx, s.tok FROM e_s1_df s JOIN e_t_df t USING (country, pfx, tok)
        WHERE s.df <= {df_threshold} AND t.df <= {df_threshold}
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE rule_e AS
        SELECT DISTINCT s.source1_entity_id, t.candidate_id
        FROM e_s1_keys s JOIN e_allowed a USING (country, pfx, tok)
        JOIN e_t_keys t USING (country, pfx, tok)
    """)
    return "rule_e"


def build_rule_F_streetnum_rare_addrtoken(con, target_key: str, min_sn_len: int = 3, df_threshold: int = 200) -> str:
    """F: exact street_number match + a shared address token, guarded on the
    COMPOUND (street_number, token) frequency -- looser than the already-
    adopted standalone rare_address_token rule (df<=25) because the
    street_number match itself already narrows the pool."""
    t_sql = f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[target_key])})"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE f_s1_keys AS
        SELECT entity_id AS source1_entity_id, country, street_number AS sn, unnest(address_tokens) AS tok
        FROM val_s1 WHERE street_number IS NOT NULL AND length(street_number) >= {min_sn_len}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE f_t_keys AS
        SELECT entity_id AS candidate_id, country, street_number AS sn, unnest(address_tokens) AS tok
        FROM {t_sql} WHERE street_number IS NOT NULL AND length(street_number) >= {min_sn_len}
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE f_s1_df AS
        SELECT country, sn, tok, count(*) AS df FROM f_s1_keys WHERE length(tok) >= 3 GROUP BY country, sn, tok
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE f_t_df AS
        SELECT country, sn, tok, count(*) AS df FROM f_t_keys WHERE length(tok) >= 3 GROUP BY country, sn, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE f_allowed AS
        SELECT s.country, s.sn, s.tok FROM f_s1_df s JOIN f_t_df t USING (country, sn, tok)
        WHERE s.df <= {df_threshold} AND t.df <= {df_threshold}
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE rule_f AS
        SELECT DISTINCT s.source1_entity_id, t.candidate_id
        FROM f_s1_keys s JOIN f_allowed a USING (country, sn, tok)
        JOIN f_t_keys t USING (country, sn, tok)
    """)
    return "rule_f"


def build_rule_I_multi_token_overlap(con, target_key: str, min_shared: int = 2, per_token_df_cap: int = 5000) -> str:
    """I: >=2 shared NAME tokens (a count-based guard instead of a single
    rare-token frequency guard) -- catches pairs where every individual
    token is too common to pass the strong_token/df<=100 rule alone, but the
    conjunction of two or more shared tokens together is still discriminating."""
    t_sql = f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[target_key])})"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE i_s1_tok AS
        SELECT entity_id AS source1_entity_id, country, unnest(name_tokens) AS tok FROM val_s1
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE i_t_tok AS
        SELECT entity_id AS candidate_id, country, unnest(name_tokens) AS tok FROM {t_sql}
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE i_s1_df AS SELECT country, tok, count(*) AS df FROM i_s1_tok GROUP BY country, tok
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE i_t_df AS SELECT country, tok, count(*) AS df FROM i_t_tok GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE i_allowed AS
        SELECT s.country, s.tok FROM i_s1_df s JOIN i_t_df t USING (country, tok)
        WHERE length(s.tok) >= 3 AND s.df <= {per_token_df_cap} AND t.df <= {per_token_df_cap}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rule_i AS
        SELECT source1_entity_id, candidate_id FROM (
            SELECT s.source1_entity_id, t.candidate_id, count(*) AS n_shared
            FROM i_s1_tok s JOIN i_allowed a USING (country, tok)
            JOIN i_t_tok t USING (country, tok)
            GROUP BY s.source1_entity_id, t.candidate_id
        ) WHERE n_shared >= {min_shared}
    """)
    return "rule_i"


def build_rule_J_hybrid_loose_name_plus_addr(con, target_key: str, ngram_df_threshold: int = 200, n: int = 4) -> str:
    """J: hybrid rule -- a LOOSER name character 4-gram than the adopted
    df<=50 (here df<=200), corroborated by requiring >=1 shared (any
    frequency) address token. The address corroboration is what allows
    loosening the name side without an unguarded volume explosion."""
    t_sql = f"(SELECT * FROM {io_utils.parquet_sql(config.NORMALIZED_FILES[target_key])})"

    def ngram_expr(col):
        return (f"unnest([substr({col}, i, {n}) FOR i IN "
                f"generate_series(1, greatest(length({col})-{n}+1, 0))])")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE j_s1_ng_df AS
        SELECT country, tok, count(*) AS df FROM (
          SELECT country, {ngram_expr('name_core')} AS tok FROM val_s1
          WHERE name_core IS NOT NULL AND length(name_core) >= {n}
        ) GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE j_t_ng_df AS
        SELECT country, tok, count(*) AS df FROM (
          SELECT country, {ngram_expr('name_core')} AS tok FROM {t_sql}
          WHERE name_core IS NOT NULL AND length(name_core) >= {n}
        ) GROUP BY country, tok
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE j_ng_allowed AS
        SELECT s.country, s.tok FROM j_s1_ng_df s JOIN j_t_ng_df t USING (country, tok)
        WHERE s.df <= {ngram_df_threshold} AND t.df <= {ngram_df_threshold}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE j_s1_ng AS
        SELECT entity_id AS source1_entity_id, country, {ngram_expr('name_core')} AS tok FROM val_s1
        WHERE name_core IS NOT NULL AND length(name_core) >= {n}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE j_t_ng AS
        SELECT entity_id AS candidate_id, country, {ngram_expr('name_core')} AS tok FROM {t_sql}
        WHERE name_core IS NOT NULL AND length(name_core) >= {n}
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE j_name_pairs AS
        SELECT DISTINCT s.source1_entity_id, t.candidate_id
        FROM j_s1_ng s JOIN j_ng_allowed a USING (country, tok)
        JOIN j_t_ng t USING (country, tok)
    """)
    # Address-token overlap is checked ONLY against pairs already surfaced by
    # the (loosened) name signal above -- j_name_pairs is a bounded relation,
    # so this is a per-pair list_intersect, not a second unguarded token join.
    # (An earlier version joined address tokens across the full val_s1 x
    # target cross product independently of j_name_pairs and hit a 33.4GiB
    # DuckDB OOM -- common address tokens like "street"/"road"/city names
    # have enormous document frequency with no compensating guard.)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rule_j AS
        SELECT n.source1_entity_id, n.candidate_id
        FROM j_name_pairs n
        JOIN val_s1 s1 ON s1.entity_id = n.source1_entity_id
        JOIN {t_sql} t ON t.entity_id = n.candidate_id
        WHERE s1.address_tokens IS NOT NULL AND t.address_tokens IS NOT NULL
          AND len(list_intersect(s1.address_tokens, t.address_tokens)) > 0
    """)
    return "rule_j"


# ---------------------------------------------------------------------------
# PART 4: evaluate a rule against the production baseline (validation scope)
# ---------------------------------------------------------------------------
def evaluate_rule(con, target_key: str, rule_table: str, rule_label: str, total_val_gt: int) -> dict:
    union_path = MATCHING_DIR / f"candidates_{target_key}.parquet"

    total_pairs = con.execute(f"SELECT count(*) FROM {rule_table}").fetchone()[0]

    added_pairs = con.execute(f"""
        SELECT count(*) FROM {rule_table} r
        LEFT JOIN read_parquet('{union_path}') c
            ON c.source1_entity_id = r.source1_entity_id AND c.candidate_id = r.candidate_id
        WHERE c.source1_entity_id IS NULL
    """).fetchone()[0]

    recovered = con.execute(f"""
        SELECT count(*) FROM val_missed_{target_key} m
        JOIN {rule_table} r ON r.source1_entity_id = m.source1_entity_id AND r.candidate_id = m.matched_id
    """).fetchone()[0]

    dist = con.execute(f"""
        SELECT max(n), quantile_cont(n, 0.95)
        FROM (SELECT source1_entity_id, count(*) AS n FROM {rule_table} GROUP BY source1_entity_id)
    """).fetchone()

    recall_gain = round(recovered / total_val_gt, 4) if total_val_gt else None
    recovered_per_1m = round(recovered / (added_pairs / 1e6), 2) if added_pairs > 0 else None

    return {
        "rule": rule_label,
        "candidate_pairs_total": total_pairs,
        "candidate_pairs_added_over_production": added_pairs,
        "new_gt_matches_recovered": recovered,
        "recall_gain": recall_gain,
        "recovered_matches_per_1M_added_candidates": recovered_per_1m,
        "max_candidates_per_s1": dist[0],
        "p95_candidates_per_s1": dist[1],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_for_target(con, target_key: str) -> dict:
    t0 = time.time()
    print(f"\n{'='*100}\nBLOCKING ERROR ANALYSIS -- {target_key}\n{'='*100}")

    n_val_s1 = build_validation_s1(con)
    print(f"validation S1 entities: {n_val_s1:,}")

    summary = blocking_miss_summary(con, target_key)
    print(f"\n--- PART 1: blocking miss summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n--- PART 2: fetching missed-match records for classification ---")
    t1 = time.time()
    df = fetch_missed_df(con, target_key)
    print(f"  fetched {len(df):,} missed-match rows in {time.time()-t1:.1f}s")
    cats, cat_table = categorize_missed(df)
    print(f"\n--- PART 2: miss categories ---")
    for c, v in cat_table.items():
        print(f"  {c}: count={v['count']} pct={v['pct']}")
    examples = representative_examples(df, cats)

    print(f"\n--- PART 3/4: rule experiments (E, F, I, J) ---")
    rule_builders = [
        ("E_postal_prefix_name_token", build_rule_E_postal_prefix_name_token),
        ("F_streetnum_rare_addrtoken", build_rule_F_streetnum_rare_addrtoken),
        ("I_multi_token_overlap", build_rule_I_multi_token_overlap),
        ("J_hybrid_loose_ngram_plus_addr", build_rule_J_hybrid_loose_name_plus_addr),
    ]
    rule_results = []
    for label, builder in rule_builders:
        t2 = time.time()
        table = builder(con, target_key)
        r = evaluate_rule(con, target_key, table, label, summary["total_validation_gt_matches"])
        r["build_seconds"] = round(time.time() - t2, 1)
        rule_results.append(r)
        print(f"  {r}")

    result = {
        "target": target_key,
        "summary": summary,
        "categories": cat_table,
        "examples": examples,
        "rule_results": rule_results,
        "total_seconds": round(time.time() - t0, 1),
    }
    print(f"\n[{target_key} total time: {result['total_seconds']}s]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Blocking error analysis (investigation only)")
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    run_for_target(con, args.target)


if __name__ == "__main__":
    main()
