"""V4 investigation: data-driven analysis of what deterministic
representations could recover the true matches still missed by V2
(V1 + ngram + guarded address). NOT a blocking implementation -- this
module only measures candidate representations against the fixed 20K S1
sample; it builds nothing that gets wired into blocking.py.

Areas investigated (see README.md "V4 investigation" for the full report):
  A. address variation
  B. typo/OCR (name character-level)
  C. alias/DBA / partial names
"""

import time
from typing import Optional

import pandas as pd

from . import config, io_utils
from .blocking_v2_experiments import build_s1_sample, build_gt_table, build_v1_candidates, build_ngram_rule, build_guarded_address_rule
from .blocking_v2_combined import build_union, PREFIX


# ---------------------------------------------------------------------------
# Build the full (not sampled-to-500) V2-missed set with all fields needed
# ---------------------------------------------------------------------------
def build_v2_missed(con, target_key: str, s1_sample: int) -> pd.DataFrame:
    prefix = PREFIX[target_key]
    build_s1_sample(con, s1_sample)
    build_gt_table(con)
    v1 = build_v1_candidates(con, target_key, s1_sample)
    ngram = build_ngram_rule(con, target_key, s1_sample, n=4, df_threshold=50)
    addr = build_guarded_address_rule(con, target_key, s1_sample)
    build_union(con, "v4_v2_cfg", [v1, ngram, addr])

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_missed_ids AS
        SELECT source1_entity_id, matched_id FROM (
            SELECT source1_entity_id, matched_id FROM gt_exploded WHERE matched_id LIKE '{prefix}%'
            EXCEPT
            SELECT source1_entity_id, candidate_id AS matched_id FROM v4_v2_cfg
        )
    """)

    s1_path = config.NORMALIZED_FILES["train_source1"]
    t_path = config.NORMALIZED_FILES[target_key]
    df = con.execute(f"""
        SELECT m.source1_entity_id, m.matched_id,
               s1.business_name AS s1_name, s1.name_normalized AS s1_norm, s1.name_core AS s1_core,
               s1.name_tokens AS s1_ntok, s1.business_address AS s1_addr_raw, s1.address_normalized AS s1_addr,
               s1.address_tokens AS s1_atok, s1.postal_code AS s1_postal, s1.street_number AS s1_sn,
               s1.city_candidate AS s1_city, s1.state_candidate AS s1_state, s1.landmark_text AS s1_landmark,
               s1.name_has_non_latin AS s1_nl,
               t.business_name AS t_name, t.name_normalized AS t_norm, t.name_core AS t_core,
               t.name_tokens AS t_ntok, t.business_address AS t_addr_raw, t.address_normalized AS t_addr,
               t.address_tokens AS t_atok, t.postal_code AS t_postal, t.street_number AS t_sn,
               t.city_candidate AS t_city, t.state_candidate AS t_state, t.landmark_text AS t_landmark,
               t.name_has_non_latin AS t_nl
        FROM v4_missed_ids m
        JOIN {io_utils.parquet_sql(s1_path)} s1 ON s1.entity_id = m.source1_entity_id
        JOIN {io_utils.parquet_sql(t_path)} t ON t.entity_id = m.matched_id
    """).fetchdf()
    return df


def token_df_full(con, target_key: str, column: str) -> pd.DataFrame:
    """Document frequency of each token in `column` (a VARCHAR[] column)
    across the FULL target table -- used to estimate real-world explosion
    risk for a proposed token-based key, not just risk within the misses."""
    t_path = config.NORMALIZED_FILES[target_key]
    return con.execute(f"""
        SELECT country, tok, count(*) AS df
        FROM (SELECT country, unnest({column}) AS tok FROM {io_utils.parquet_sql(t_path)})
        GROUP BY country, tok
    """).fetchdf()


def ngram_df_full(con, target_key: str, source_column: str, n: int = 4) -> pd.DataFrame:
    t_path = config.NORMALIZED_FILES[target_key]
    return con.execute(f"""
        SELECT country, tok, count(*) AS df FROM (
          SELECT country, unnest([substr({source_column},i,{n}) FOR i IN
                 generate_series(1, greatest(length({source_column})-{n}+1,0))]) AS tok
          FROM {io_utils.parquet_sql(t_path)}
          WHERE {source_column} IS NOT NULL AND length({source_column}) >= {n}
        ) GROUP BY country, tok
    """).fetchdf()


# ---------------------------------------------------------------------------
# Generic helpers (pure Python -- dataset here is a few thousand rows)
# ---------------------------------------------------------------------------
def _nonempty(x) -> bool:
    """Safe non-empty check for list-like values coming out of pandas
    (which may be numpy arrays, where plain `if x:` raises 'ambiguous
    truth value')."""
    return x is not None and len(x) > 0


def _jaccard(a, b) -> float:
    if not _nonempty(a) or not _nonempty(b):
        return 0.0
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _char_ngrams(s: Optional[str], n: int = 4) -> set:
    if not s or len(s) < n:
        return set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


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


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _common_suffix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def _lcs_tokens(a: list, b: list) -> int:
    """Longest common (not necessarily contiguous) token subsequence length."""
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[la][lb]


# ---------------------------------------------------------------------------
# AREA A: address variation
# ---------------------------------------------------------------------------
def analyze_area_a(con, df: pd.DataFrame, target_key: str) -> dict:
    n = len(df)
    both_addr = df["s1_addr"].notna() & df["t_addr"].notna()
    missing_addr = ~both_addr

    addr_token_jaccard = df.apply(lambda r: _jaccard(r.s1_atok, r.t_atok) if both_addr[r.name] else None, axis=1)
    sorted_tokens_equal = df.apply(
        lambda r: (sorted(r.s1_atok) == sorted(r.t_atok))
                  if both_addr[r.name] and _nonempty(r.s1_atok) and _nonempty(r.t_atok) else False,
        axis=1)
    addr_ngram_jaccard = df.apply(
        lambda r: _jaccard(_char_ngrams(r.s1_addr), _char_ngrams(r.t_addr)) if both_addr[r.name] else None, axis=1)

    postal_both = df["s1_postal"].notna() & df["t_postal"].notna()
    postal_agree = postal_both & (df["s1_postal"] == df["t_postal"])
    sn_both = df["s1_sn"].notna() & df["t_sn"].notna()
    sn_agree = sn_both & (df["s1_sn"] == df["t_sn"])
    city_both = df["s1_city"].notna() & df["t_city"].notna()
    city_agree = city_both & (df["s1_city"] == df["t_city"])

    one_subset_other = df.apply(
        lambda r: bool(both_addr[r.name]) and _nonempty(r.s1_atok) and _nonempty(r.t_atok) and
                  (set(r.s1_atok) <= set(r.t_atok) or set(r.t_atok) <= set(r.s1_atok)), axis=1)

    landmark_present = (df["s1_landmark"].notna() | df["t_landmark"].notna())

    # Rare-address-token overlap: does the miss share >=1 address token whose
    # document frequency in the FULL target table is below a threshold?
    # (Real-world explosion risk is about frequency in the whole population,
    # not just within this small missed-set.)
    addr_tok_df = token_df_full(con, target_key, "address_tokens")
    addr_tok_df_map = {(c, t): d for c, t, d in zip(addr_tok_df["country"], addr_tok_df["tok"], addr_tok_df["df"])}

    def shares_rare_addr_token(row, thresh):
        if not _nonempty(row.s1_atok) or not _nonempty(row.t_atok):
            return False
        shared = set(row.s1_atok) & set(row.t_atok)
        country = row.s1_city and None  # country not carried in df; look up via both possible keys below
        return any(
            addr_tok_df_map.get(("US", tok), 10**9) <= thresh or addr_tok_df_map.get(("India", tok), 10**9) <= thresh
            for tok in shared
        )

    rare_addr_share = {thresh: round(100 * df.apply(lambda r: shares_rare_addr_token(r, thresh), axis=1).mean(), 1)
                        for thresh in [50, 200, 1000]}

    # Address character n-gram rare-overlap, same style
    addr_ng_df = ngram_df_full(con, target_key, "address_normalized", n=4)
    addr_ng_df_map = {(c, t): d for c, t, d in zip(addr_ng_df["country"], addr_ng_df["tok"], addr_ng_df["df"])}

    def shares_rare_addr_ngram(row, thresh):
        if not row.s1_addr or not row.t_addr:
            return False
        shared = _char_ngrams(row.s1_addr) & _char_ngrams(row.t_addr)
        return any(
            addr_ng_df_map.get(("US", ng), 10**9) <= thresh or addr_ng_df_map.get(("India", ng), 10**9) <= thresh
            for ng in shared
        )

    rare_ngram_share = {thresh: round(100 * df.apply(lambda r: shares_rare_addr_ngram(r, thresh), axis=1).mean(), 1)
                         for thresh in [50, 200, 1000]}

    result = {
        "target": target_key,
        "n_misses": n,
        "missing_address_pct": round(100 * missing_addr.mean(), 1),
        "both_have_address_pct": round(100 * both_addr.mean(), 1),
        "addr_token_jaccard_ge_0.5_pct": round(100 * (addr_token_jaccard.dropna() >= 0.5).mean(), 1) if both_addr.any() else 0,
        "addr_token_jaccard_ge_0.3_pct": round(100 * (addr_token_jaccard.dropna() >= 0.3).mean(), 1) if both_addr.any() else 0,
        "sorted_address_tokens_exact_equal_pct": round(100 * sorted_tokens_equal.mean(), 1),
        "addr_char_ngram_jaccard_ge_0.5_pct": round(100 * (addr_ngram_jaccard.dropna() >= 0.5).mean(), 1) if both_addr.any() else 0,
        "postal_both_present_pct": round(100 * postal_both.mean(), 2),
        "postal_agree_pct": round(100 * postal_agree.mean(), 2),
        "street_number_both_present_pct": round(100 * sn_both.mean(), 1),
        "street_number_agree_pct": round(100 * sn_agree.mean(), 1),
        "city_both_present_pct": round(100 * city_both.mean(), 1),
        "city_agree_pct": round(100 * city_agree.mean(), 1),
        "one_address_subset_of_other_pct": round(100 * one_subset_other.mean(), 1),
        "landmark_present_either_side_pct": round(100 * landmark_present.mean(), 1),
        "shares_rare_address_token_pct_by_threshold": rare_addr_share,
        "shares_rare_address_4gram_pct_by_threshold": rare_ngram_share,
    }
    return result


# ---------------------------------------------------------------------------
# AREA B: typo / OCR (name character level)
# ---------------------------------------------------------------------------
def analyze_area_b(df: pd.DataFrame) -> dict:
    both_core = df["s1_core"].notna() & df["t_core"].notna()
    same_script = df["s1_nl"] == df["t_nl"]
    subset = df[both_core & same_script].copy()
    n_subset = len(subset)
    if n_subset == 0:
        return {"n_same_script_both_core": 0}

    subset["edit_dist"] = subset.apply(lambda r: _levenshtein(r.s1_core, r.t_core), axis=1)
    subset["ngram_jaccard"] = subset.apply(lambda r: _jaccard(_char_ngrams(r.s1_core), _char_ngrams(r.t_core)), axis=1)
    subset["prefix_len"] = subset.apply(lambda r: _common_prefix_len(r.s1_core, r.t_core), axis=1)
    subset["suffix_len"] = subset.apply(lambda r: _common_suffix_len(r.s1_core, r.t_core), axis=1)

    dist_buckets = subset["edit_dist"].value_counts().sort_index()
    return {
        "n_same_script_both_core": n_subset,
        "edit_dist_le_2_pct": round(100 * (subset["edit_dist"] <= 2).mean(), 1),
        "edit_dist_le_4_pct": round(100 * (subset["edit_dist"] <= 4).mean(), 1),
        "edit_dist_le_8_pct": round(100 * (subset["edit_dist"] <= 8).mean(), 1),
        "edit_dist_median": subset["edit_dist"].median(),
        "edit_dist_distribution_head": dist_buckets.head(10).to_dict(),
        "ngram_jaccard_ge_0.5_pct": round(100 * (subset["ngram_jaccard"] >= 0.5).mean(), 1),
        "ngram_jaccard_ge_0.3_pct": round(100 * (subset["ngram_jaccard"] >= 0.3).mean(), 1),
        "common_prefix_ge_3_pct": round(100 * (subset["prefix_len"] >= 3).mean(), 1),
        "common_suffix_ge_3_pct": round(100 * (subset["suffix_len"] >= 3).mean(), 1),
    }


# ---------------------------------------------------------------------------
# AREA C: alias / DBA / partial names
# ---------------------------------------------------------------------------
DBA_MARKERS = ["dba", "d b a", "trading as", "aka", "a k a", "formerly", "fka"]


def analyze_area_c(df: pd.DataFrame) -> dict:
    both_core = df["s1_core"].notna() & df["t_core"].notna()
    subset = df[both_core].copy()
    n = len(subset)
    if n == 0:
        return {"n_both_core": 0}

    subset["s1_in_t"] = subset.apply(lambda r: (r.s1_core in r.t_core) if r.s1_core else False, axis=1)
    subset["t_in_s1"] = subset.apply(lambda r: (r.t_core in r.s1_core) if r.t_core else False, axis=1)
    subset["substring_either"] = subset["s1_in_t"] | subset["t_in_s1"]

    subset["tok_jaccard"] = subset.apply(lambda r: _jaccard(r.s1_ntok, r.t_ntok), axis=1)
    subset["tok_containment"] = subset.apply(
        lambda r: _nonempty(r.s1_ntok) and _nonempty(r.t_ntok) and (set(r.s1_ntok) <= set(r.t_ntok) or set(r.t_ntok) <= set(r.s1_ntok)),
        axis=1)
    subset["lcs_tokens"] = subset.apply(
        lambda r: _lcs_tokens(list(r.s1_ntok) if _nonempty(r.s1_ntok) else [],
                               list(r.t_ntok) if _nonempty(r.t_ntok) else []), axis=1)
    subset["min_ntok_len"] = subset.apply(
        lambda r: min(len(r.s1_ntok) if _nonempty(r.s1_ntok) else 0,
                       len(r.t_ntok) if _nonempty(r.t_ntok) else 0), axis=1)
    subset["lcs_ratio"] = subset.apply(lambda r: (r.lcs_tokens / r.min_ntok_len) if r.min_ntok_len else 0, axis=1)

    has_dba_marker = subset.apply(
        lambda r: any(m in (r.s1_norm or "") or m in (r.t_norm or "") for m in DBA_MARKERS), axis=1)

    very_low_overlap = subset["tok_jaccard"] < 0.2

    return {
        "n_both_core": n,
        "substring_either_direction_pct": round(100 * subset["substring_either"].mean(), 1),
        "token_containment_pct": round(100 * subset["tok_containment"].mean(), 1),
        "token_jaccard_lt_0.2_pct": round(100 * very_low_overlap.mean(), 1),
        "lcs_ratio_ge_0.5_pct": round(100 * (subset["lcs_ratio"] >= 0.5).mean(), 1),
        "has_explicit_dba_marker_pct": round(100 * has_dba_marker.mean(), 1),
        "very_low_overlap_AND_no_dba_marker_pct": round(100 * (very_low_overlap & ~has_dba_marker).mean(), 1),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="V4 investigation: what could recover V2 misses?")
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    parser.add_argument("--s1-sample", type=int, default=20000)
    args = parser.parse_args()

    con = io_utils.get_configured_connection()
    t0 = time.time()

    df = build_v2_missed(con, args.target, args.s1_sample)
    print(f"\n{'='*100}\nV2-missed set for {args.target}: {len(df):,} true matches\n{'='*100}")

    print(f"\n--- AREA A: address variation ({args.target}) ---")
    a = analyze_area_a(con, df, args.target)
    for k, v in a.items():
        print(f"  {k}: {v}")

    print(f"\n--- AREA B: typo/OCR ({args.target}) ---")
    b = analyze_area_b(df)
    for k, v in b.items():
        print(f"  {k}: {v}")

    print(f"\n--- AREA C: alias/DBA/partial names ({args.target}) ---")
    c = analyze_area_c(df)
    for k, v in c.items():
        print(f"  {k}: {v}")

    print(f"\n[total time: {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
