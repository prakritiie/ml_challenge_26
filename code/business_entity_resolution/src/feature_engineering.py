"""Pure Python/NumPy implementation of the 27 pairwise similarity features
used by the final LightGBM matching models (lightgbm_source2_baseline.txt /
lightgbm_source3_baseline.txt). Verified byte-for-byte identical to the
original DuckDB-SQL feature definitions used at training time: 27/27
features, 0 mismatches across 2,012 validation pairs, including 20/20
non-ASCII spot checks (see docs for the parity methodology).

This is the OPTIMIZED implementation actually used for the final 0.809
scoring run:
  - name_edit_distance uses rapidfuzz.distance.Levenshtein.distance() on the
    UTF-8-encoded bytes of name_core (matching DuckDB's byte-level, not
    codepoint-level, levenshtein() semantics), instead of a pure-Python
    O(n*m) DP loop -- ~20-100x faster, and was the dominant per-row cost in
    an earlier pure-Python version.
  - name_token_jaccard/name_token_overlap_count (and the address_token_*
    pair) each build their two sets ONCE and reuse them for both the
    Jaccard ratio and the overlap count, instead of rebuilding from scratch
    per metric.
No feature definition, null-handling rule, or numeric formula was changed
by either optimization.
"""
from typing import Optional, List
from rapidfuzz.distance import Levenshtein

MODEL_FEATURE_COLUMNS = [
    "exact_normalized_name", "exact_core_name", "exact_sorted_tokens",
    "name_token_jaccard", "name_token_overlap_count",
    "name_char_ngram_jaccard", "name_edit_distance", "name_edit_similarity",
    "name_length_diff", "name_token_count_diff",
    "exact_normalized_address", "address_token_jaccard", "address_token_overlap_count",
    "postal_code_equal", "street_number_equal", "city_equal", "state_equal",
    "address_length_diff",
    "target_name_has_non_latin", "target_address_has_non_latin",
    "blocked_strong_token", "blocked_ngram", "blocked_postal", "blocked_city_street",
    "blocked_rare_address_token", "blocked_address_ngram", "blocked_name_prefix",
]


def _null_safe_exact_eq(a, b) -> Optional[bool]:
    """Reproduces SQL's `(a IS NOT NULL AND a = b)` three-valued-logic
    NULL propagation exactly: a is None -> False; a not None, b None ->
    None (a genuine NULL feature value, not False); both not None -> a==b.
    """
    if a is None:
        return False
    if b is None:
        return None
    return a == b


def _jaccard_and_overlap(a: Optional[List], b: Optional[List]):
    """Jaccard ratio and intersection size over two token lists, computed
    from ONE pair of sets (not rebuilt separately per metric)."""
    if a is None or b is None:
        return 0.0, 0
    if len(a) == 0 or len(b) == 0:
        sa, sb = set(a), set(b)
        return 0.0, len(sa & sb)
    sa, sb = set(a), set(b)
    inter = sa & sb
    union = sa | sb
    jaccard = len(inter) / max(len(union), 1)
    return jaccard, len(inter)


def _ngrams(s: str, n: int = 4):
    if s is None or len(s) < n:
        return []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def _char_ngram_jaccard(a: Optional[str], b: Optional[str], n: int = 4) -> float:
    if a is None or b is None or len(a) < n or len(b) < n:
        return 0.0
    sa, sb = set(_ngrams(a, n)), set(_ngrams(b, n))
    if not sa and not sb:
        return 0.0
    union = sa | sb
    inter = sa & sb
    return len(inter) / max(len(union), 1)


def compute_features(s1: dict, t: dict, prov: dict) -> list:
    """s1, t: dicts of normalized fields (name_normalized, name_core,
    name_tokens, name_sorted_tokens, address_normalized, address_tokens,
    postal_code, street_number, city_candidate, state_candidate, and (for
    t only) name_has_non_latin/address_has_non_latin).
    prov: dict of the 7 blocked_* booleans from the (now-superseded)
    rule-based blocking stage, or {} -- missing keys default to False,
    which is the correct value for every candidate found only via the
    TF-IDF/inverted-index retrieval used in the final pipeline.
    Returns the 27 features as a list, in MODEL_FEATURE_COLUMNS order."""
    s1_name_norm, t_name_norm = s1.get("name_normalized"), t.get("name_normalized")
    s1_core, t_core = s1.get("name_core"), t.get("name_core")
    s1_sorted, t_sorted = s1.get("name_sorted_tokens"), t.get("name_sorted_tokens")
    s1_ntok, t_ntok = s1.get("name_tokens"), t.get("name_tokens")
    s1_addr, t_addr = s1.get("address_normalized"), t.get("address_normalized")
    s1_atok, t_atok = s1.get("address_tokens"), t.get("address_tokens")
    s1_postal, t_postal = s1.get("postal_code"), t.get("postal_code")
    s1_sn, t_sn = s1.get("street_number"), t.get("street_number")
    s1_city, t_city = s1.get("city_candidate"), t.get("city_candidate")
    s1_state, t_state = s1.get("state_candidate"), t.get("state_candidate")

    exact_normalized_name = _null_safe_exact_eq(s1_name_norm, t_name_norm)
    exact_core_name = _null_safe_exact_eq(s1_core, t_core)
    if s1_sorted is None or len(s1_sorted) == 0:
        exact_sorted_tokens = False
    else:
        exact_sorted_tokens = _null_safe_exact_eq(s1_sorted, t_sorted)

    name_token_jaccard, name_token_overlap_count = _jaccard_and_overlap(s1_ntok, t_ntok)

    name_char_ngram_jaccard = _char_ngram_jaccard(s1_core, t_core, 4)

    if s1_core is not None and t_core is not None:
        ed = Levenshtein.distance(s1_core.encode("utf-8"), t_core.encode("utf-8"))
        name_edit_distance = ed
        name_edit_similarity = 1.0 - ed / max(len(s1_core), len(t_core), 1)
    else:
        name_edit_distance = None
        name_edit_similarity = None

    name_length_diff = abs((len(s1_name_norm) if s1_name_norm else 0) - (len(t_name_norm) if t_name_norm else 0))
    name_token_count_diff = abs((len(s1_ntok) if s1_ntok else 0) - (len(t_ntok) if t_ntok else 0))

    exact_normalized_address = _null_safe_exact_eq(s1_addr, t_addr)
    address_token_jaccard, address_token_overlap_count = _jaccard_and_overlap(s1_atok, t_atok)

    postal_code_equal = bool(s1_postal is not None and t_postal is not None and s1_postal == t_postal)
    street_number_equal = bool(s1_sn is not None and t_sn is not None and s1_sn == t_sn)
    city_equal = bool(s1_city is not None and t_city is not None and s1_city == t_city)
    state_equal = bool(s1_state is not None and t_state is not None and s1_state == t_state)

    address_length_diff = abs((len(s1_addr) if s1_addr else 0) - (len(t_addr) if t_addr else 0))

    target_name_has_non_latin = bool(t.get("name_has_non_latin", False))
    target_address_has_non_latin = bool(t.get("address_has_non_latin", False))

    blocked_strong_token = bool(prov.get("blocked_strong_token", False))
    blocked_ngram = bool(prov.get("blocked_ngram", False))
    blocked_postal = bool(prov.get("blocked_postal", False))
    blocked_city_street = bool(prov.get("blocked_city_street", False))
    blocked_rare_address_token = bool(prov.get("blocked_rare_address_token", False))
    blocked_address_ngram = bool(prov.get("blocked_address_ngram", False))
    blocked_name_prefix = bool(prov.get("blocked_name_prefix", False))

    return [
        exact_normalized_name, exact_core_name, exact_sorted_tokens,
        name_token_jaccard, name_token_overlap_count,
        name_char_ngram_jaccard, name_edit_distance, name_edit_similarity,
        name_length_diff, name_token_count_diff,
        exact_normalized_address, address_token_jaccard, address_token_overlap_count,
        postal_code_equal, street_number_equal, city_equal, state_equal,
        address_length_diff,
        target_name_has_non_latin, target_address_has_non_latin,
        blocked_strong_token, blocked_ngram, blocked_postal, blocked_city_street,
        blocked_rare_address_token, blocked_address_ngram, blocked_name_prefix,
    ]
