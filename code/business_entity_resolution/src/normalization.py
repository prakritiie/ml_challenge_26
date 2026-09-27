"""Deterministic, dataset-only normalization of business_name and
business_address fields.

This module implements ONLY the normalization stage of the pipeline:
no blocking, no embeddings, no feature engineering, no ML. See README.md
for the overall pipeline roadmap.

Design notes
------------
- All logic is pure Python (regex/unicodedata) so it is easy to unit-test
  and reason about; DuckDB + Arrow (see io_utils.py) are used purely as the
  scalable I/O layer that streams rows through this logic in bounded-size
  batches, so a 5M+ row file never needs to fit in memory at once.
- Every transformation is derived only from the string itself (this file's
  lookup tables in config.py are static, hand-authored, and contain no
  external business/geocoding data).
"""

import argparse
import re
import time
import tracemalloc
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa

from . import config, io_utils

# ---------------------------------------------------------------------------
# Regexes (compiled once at import time)
# ---------------------------------------------------------------------------
_JUNK_TOKEN_RE = re.compile(r"(>>+|<<+|#+|\*+|~+)")
_BRACKET_RE = re.compile(r"[\[\]{}()<>]")
_MULTI_PUNCT_RE = re.compile(r"([.,;:!?'\"\-])\1+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:\-]+$")
_LEADING_PUNCT_RE = re.compile(r"^[.,;:\-]+")
_AMP_RE = re.compile(r"\s*&\s*")
_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[a-zA-Z]\.){1,}[a-zA-Z]\.?\b")
_NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")
_NAME_ALLOWED_PUNCT = set("&/-.,'")
_ADDR_ALLOWED_PUNCT = set(",./-")
# Python's \w (Unicode mode) matches letters/digits but NOT combining marks
# (category Mn/Mc) -- Devanagari/Tamil/etc. vowel signs are combining marks,
# so a naive `[^\w...]` filter silently deletes them and corrupts the script
# into disconnected consonants (found during testing). Filtering by Unicode
# category keeps any Letter (L*), Mark (M*), or Number (N*) character intact
# regardless of script.
_KEEP_CATEGORIES = ("L", "M", "N")
_POSTAL_RE = re.compile(r"\b(\d{6}|\d{5}(?:-\d{4})?)\b")
_STREET_NUM_RE = re.compile(r"\d+[a-z]?(?:[/-]\d+[a-z]?)?")
_NUMERIC_SEGMENT_RE = re.compile(r"^\d{4,6}(-\d{4})?$")

_LANDMARK_PATTERN = "|".join(re.escape(k) for k in config.LANDMARK_KEYWORDS)
_LANDMARK_RE = re.compile(rf"\b(?:{_LANDMARK_PATTERN})\b.*?(?=,|$)")

# Derived from config.LEGAL_SUFFIX_GROUPS: variant token -> canonical group name.
_SUFFIX_SYNONYMS: Dict[str, str] = {
    variant: canonical
    for canonical, variants in config.LEGAL_SUFFIX_GROUPS.items()
    for variant in variants
}
_LEGAL_SUFFIX_CANONICAL = set(config.LEGAL_SUFFIX_GROUPS.keys())


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def is_missing(value: Optional[str]) -> bool:
    if value is None:
        return True
    return value.strip().lower() in config.MISSING_TOKENS


def has_non_latin(raw: Optional[str]) -> bool:
    if not raw:
        return False
    return bool(_NON_LATIN_RE.search(raw))


def to_ascii(text: Optional[str]) -> Optional[str]:
    """Strip non-Latin characters, preserving surrounding Latin/alphanumeric
    content. Does NOT attempt transliteration (per spec: no external
    transliteration in this stage)."""
    if not text:
        return None
    stripped = _NON_LATIN_RE.sub(" ", text)
    stripped = _MULTI_SPACE_RE.sub(" ", stripped).strip()
    return stripped or None


def _filter_keep_chars(s: str, allowed_punct: set) -> str:
    """Replace any character that is not a Letter/Mark/Number, whitespace,
    or explicitly allowed punctuation with a space. Script-agnostic: keeps
    Devanagari/Tamil/Kannada/etc. combining marks attached to their base
    letters instead of the '\\w'-only behaviour that strips them."""
    out = []
    for ch in s:
        if ch.isspace() or ch in allowed_punct or unicodedata.category(ch)[0] in _KEEP_CATEGORIES:
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def tokenize(text: Optional[str]) -> List[str]:
    """Split on whitespace/punctuation while keeping any run of
    Letter/Mark/Number characters together as one token, so combining marks
    (e.g. Devanagari matras) stay attached to their base consonant."""
    if not text:
        return []
    tokens: List[str] = []
    current: List[str] = []
    for ch in text:
        if unicodedata.category(ch)[0] in _KEEP_CATEGORIES:
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def squash_dotted_acronyms(s: str) -> str:
    """'l.l.c.' -> 'llc', 'p.c.' -> 'pc', so the legal-suffix table (which is
    keyed on undotted forms) recognizes them uniformly."""
    return _DOTTED_ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), s)


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------
def normalize_name(raw: Optional[str]) -> Optional[str]:
    """Unicode-aware cleanup: casefold, normalize '&', strip bracket/junk
    noise and repeated punctuation, collapse whitespace. Legal suffixes are
    NOT removed here -- name_normalized keeps them (see name_core)."""
    if raw is None or is_missing(raw):
        return None
    s = unicodedata.normalize("NFKC", raw)
    s = s.casefold()
    s = squash_dotted_acronyms(s)
    s = _JUNK_TOKEN_RE.sub(" ", s)
    s = _AMP_RE.sub(" and ", s)
    s = _BRACKET_RE.sub(" ", s)
    s = _MULTI_PUNCT_RE.sub(r"\1", s)
    s = _filter_keep_chars(s, _NAME_ALLOWED_PUNCT)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = _LEADING_PUNCT_RE.sub("", s)
    s = s.strip()
    return s or None


def _canonical_token(tok: str) -> str:
    return _SUFFIX_SYNONYMS.get(tok, tok)


def strip_legal_suffix_tokens(tokens: List[str]) -> List[str]:
    """Repeatedly remove trailing tokens that are recognized legal-suffix
    words (e.g. 'private', 'limited' both get popped from '... Private
    Limited'). Never returns an empty list from a non-empty input --
    falls back to the original tokens if stripping would remove everything
    (do not aggressively delete information)."""
    if not tokens:
        return []
    stripped = list(tokens)
    while stripped and _canonical_token(stripped[-1]) in _LEGAL_SUFFIX_CANONICAL:
        stripped.pop()
    return stripped if stripped else list(tokens)


def normalize_name_fields(raw_name: Optional[str]) -> Dict[str, object]:
    normalized = normalize_name(raw_name)
    tokens = tokenize(normalized)
    sorted_tokens = sorted(tokens)
    core_tokens = strip_legal_suffix_tokens(tokens)
    name_core = " ".join(core_tokens) if core_tokens else normalized
    return {
        "name_normalized": normalized,
        "name_core": name_core,
        "name_tokens": tokens,
        "name_sorted_tokens": sorted_tokens,
        "name_ascii": to_ascii(normalized),
        "name_has_non_latin": has_non_latin(raw_name),
    }


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------
def _expand_address_abbreviations(s: str) -> str:
    for abbr, full in config.ADDRESS_ABBREVIATIONS.items():
        s = re.sub(rf"\b{re.escape(abbr)}\b\.?", full, s)
    # "st" -> "street", except as the first token (often "Saint", e.g. "St Louis")
    s = re.sub(r"(?<!^)\bst\b\.?", "street", s)
    # "dr" -> "drive", only when "dr" is the LAST word of its comma segment
    # (typical of the street-type abbreviation, e.g. "123 Main Dr"). A "Dr"
    # followed by more words is left alone, since it is usually the title
    # "Doctor" inside a person's name the road is named after (e.g.
    # "Dr Rajkumar Road", "Dr Martin Luther King Jr Blvd"). See config.py.
    s = re.sub(r"\bdr\b\.?(?=\s*(?:,|$))", "drive", s)
    # "no"/"no." -> "number" only when immediately followed by a digit
    s = re.sub(r"\bno\.?\s*(?=\d)", "number ", s)
    return s


def _drop_missing_components(s: str) -> str:
    """Drop individual comma-separated components that are themselves a
    missing-value marker (e.g. the literal '<NULL>' placeholder found
    embedded inside otherwise-valid addresses in the EDA), rather than only
    checking the field as a whole."""
    parts = [p.strip() for p in s.split(",")]
    kept = [p for p in parts if p and p.strip(" .") not in config.MISSING_TOKENS]
    return ", ".join(kept)


def normalize_address(raw: Optional[str]) -> Optional[str]:
    if raw is None or is_missing(raw):
        return None
    s = unicodedata.normalize("NFKC", raw)
    s = s.casefold()
    s = squash_dotted_acronyms(s)
    s = _JUNK_TOKEN_RE.sub(" ", s)
    s = _BRACKET_RE.sub(" ", s)
    s = _MULTI_PUNCT_RE.sub(r"\1", s)
    s = _filter_keep_chars(s, _ADDR_ALLOWED_PUNCT)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    s = _expand_address_abbreviations(s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    s = _drop_missing_components(s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = s.strip()
    return s or None


def extract_landmark(address_normalized: Optional[str]) -> Optional[str]:
    if not address_normalized:
        return None
    m = _LANDMARK_RE.search(address_normalized)
    return m.group(0).strip() if m else None


def extract_street_number(address_normalized: Optional[str]) -> Optional[str]:
    if not address_normalized:
        return None
    first_segment = address_normalized.split(",")[0]
    m = _STREET_NUM_RE.search(first_segment)
    return m.group(0) if m else None


def extract_postal_code(address_normalized: Optional[str]) -> Optional[str]:
    """Search only the components AFTER the first (street) segment, so a
    5/6-digit street number (e.g. '10404 Glade Road, ...') is never
    mistaken for a postal code -- real ZIP/PIN codes trail the address."""
    if not address_normalized:
        return None
    parts = _split_components(address_normalized)
    if len(parts) <= 1:
        return None
    tail = " ".join(parts[1:])
    matches = _POSTAL_RE.findall(tail)
    return matches[-1] if matches else None


def _split_components(address_normalized: str) -> List[str]:
    return [p.strip() for p in address_normalized.split(",") if p.strip()]


def extract_city_state_candidates(
    address_normalized: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Heuristic only: comma-separated trailing components tend to carry
    city/state in both the US ('..., City, ST') and Indian ('..., City,
    State') formats seen in the EDA. These are candidates, not guaranteed
    truth -- purely positional, no gazetteer/geocoding involved."""
    if not address_normalized:
        return None, None
    parts = _split_components(address_normalized)
    parts = [p for p in parts if not _NUMERIC_SEGMENT_RE.match(p)]
    if not parts:
        return None, None
    state_candidate = parts[-1]
    city_candidate = parts[-2] if len(parts) >= 2 else None
    return city_candidate, state_candidate


def normalize_address_fields(raw_address: Optional[str]) -> Dict[str, object]:
    normalized = normalize_address(raw_address)
    city_candidate, state_candidate = extract_city_state_candidates(normalized)
    return {
        "address_normalized": normalized,
        "address_tokens": tokenize(normalized),
        "address_ascii": to_ascii(normalized),
        "address_has_non_latin": has_non_latin(raw_address),
        "street_number": extract_street_number(normalized),
        "postal_code": extract_postal_code(normalized),
        "city_candidate": city_candidate,
        "state_candidate": state_candidate,
        "landmark_text": extract_landmark(normalized),
    }


# ---------------------------------------------------------------------------
# Batch-level application (Arrow RecordBatch in, RecordBatch out)
# ---------------------------------------------------------------------------
DERIVED_NAME_COLUMNS = [
    "name_normalized", "name_core", "name_tokens", "name_sorted_tokens",
    "name_ascii", "name_has_non_latin",
]
DERIVED_ADDRESS_COLUMNS = [
    "address_normalized", "address_tokens", "address_ascii", "address_has_non_latin",
    "street_number", "postal_code", "city_candidate", "state_candidate", "landmark_text",
]


def normalize_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    names = batch.column("business_name").to_pylist()
    addresses = batch.column("business_address").to_pylist()

    cols: Dict[str, list] = {c: [] for c in DERIVED_NAME_COLUMNS + DERIVED_ADDRESS_COLUMNS}
    for name, addr in zip(names, addresses):
        nf = normalize_name_fields(name)
        af = normalize_address_fields(addr)
        for k, v in nf.items():
            cols[k].append(v)
        for k, v in af.items():
            cols[k].append(v)

    type_map = {
        "name_tokens": pa.list_(pa.string()),
        "name_sorted_tokens": pa.list_(pa.string()),
        "address_tokens": pa.list_(pa.string()),
        "name_has_non_latin": pa.bool_(),
        "address_has_non_latin": pa.bool_(),
    }
    new_arrays = [pa.array(cols[name], type=type_map.get(name, pa.string())) for name in cols]
    new_names = list(cols.keys())

    out = batch
    for name, arr in zip(new_names, new_arrays):
        out = out.append_column(name, arr)
    return out


# ---------------------------------------------------------------------------
# File-level and CLI entry points
# ---------------------------------------------------------------------------
def normalize_file(con, input_path: Path, output_path: Path, batch_size: int = config.BATCH_SIZE) -> int:
    """Stream `input_path` through normalization in `batch_size`-row chunks
    and write the result to a single Parquet file at `output_path`. Memory
    use is bounded by batch_size, not by total file size."""
    batches = io_utils.iter_record_batches(con, input_path, batch_size=batch_size)
    normalized = (normalize_batch(b) for b in batches)
    return io_utils.write_batches_to_parquet(normalized, output_path)


_SAMPLE_PRINT_COLUMNS_NAME = [
    "business_name", "name_normalized", "name_core", "name_sorted_tokens", "name_ascii",
]
_SAMPLE_PRINT_COLUMNS_ADDR = [
    "business_address", "address_normalized", "street_number", "postal_code",
    "city_candidate", "state_candidate", "landmark_text",
]


def run_sample(source_key: str, n: int = 1000, show: int = 8) -> None:
    con = io_utils.get_connection()
    path = config.SOURCE_FILES[source_key]

    tracemalloc.start()
    t0 = time.time()
    table = io_utils.sample_rows(con, path, n=n)
    normalized_batches = [normalize_batch(b) for b in table.to_batches()]
    normalized_table = pa.Table.from_batches(normalized_batches)
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    df = normalized_table.to_pandas()
    print(f"\n=== {source_key}: sampled {df.shape[0]} rows in {elapsed:.3f}s "
          f"(peak Python memory during normalization: {peak / 1e6:.1f} MB) ===")

    print(f"\n-- name fields ({show} examples) --")
    print(df[_SAMPLE_PRINT_COLUMNS_NAME].head(show).to_string(index=False))

    print(f"\n-- address fields ({show} examples) --")
    print(df[_SAMPLE_PRINT_COLUMNS_ADDR].head(show).to_string(index=False))


def run_full(source_key: str, batch_size: int = config.BATCH_SIZE) -> None:
    con = io_utils.get_connection()
    input_path = config.SOURCE_FILES[source_key]
    output_path = config.NORMALIZED_FILES[source_key]

    tracemalloc.start()
    t0 = time.time()
    n_rows = normalize_file(con, input_path, output_path, batch_size=batch_size)
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    size_mb = output_path.stat().st_size / 1e6
    print(f"{source_key}: normalized {n_rows:,} rows in {elapsed:.1f}s "
          f"(peak Python memory: {peak / 1e6:.1f} MB) -> {output_path} ({size_mb:.1f} MB)")


def run_demo() -> None:
    """Pull a handful of real rows known (from EDA) to exercise each noise
    pattern, and print their normalized output for manual inspection."""
    con = io_utils.get_connection()
    s1 = config.SOURCE_FILES["train_source1"]
    s2 = config.SOURCE_FILES["train_source2"]
    s3 = config.SOURCE_FILES["train_source3"]

    patterns = [
        ("legal suffix variation", s1, "lower(business_name) LIKE '%private limited%'"),
        ("punctuation noise", s2, "business_name LIKE '%[%' OR business_name LIKE '%]%' OR business_name LIKE '%>>%'"),
        ("ALL CAPS", s2, "business_name = upper(business_name) AND business_name != lower(business_name)"),
        ("None/null address", s2, "business_address = 'None' OR business_address IS NULL"),
        ("Indian non-Latin script", s2, "country = 'India' AND business_name != regexp_replace(business_name, '[^\\x00-\\x7F]', '', 'g')"),
        ("landmark-based address", s3, "lower(business_address) LIKE '%near %' OR lower(business_address) LIKE '%opp %'"),
        ("partial / short address", s3, "length(business_address) < 20 AND business_address IS NOT NULL"),
    ]

    for title, path, where in patterns:
        print(f"\n=== DEMO: {title} ===")
        rel = io_utils.run_sql(con, path, f" WHERE {where} LIMIT 3")
        rows = rel.fetchall()
        cols = [d[0] for d in rel.description]
        for row in rows:
            rec = dict(zip(cols, row))
            nf = normalize_name_fields(rec.get("business_name"))
            af = normalize_address_fields(rec.get("business_address"))
            print(f"  entity_id       : {rec.get('entity_id')}")
            print(f"  business_name   : {rec.get('business_name')!r}")
            print(f"    -> name_normalized   : {nf['name_normalized']!r}")
            print(f"    -> name_core         : {nf['name_core']!r}")
            print(f"    -> name_sorted_tokens: {nf['name_sorted_tokens']!r}")
            print(f"    -> name_ascii        : {nf['name_ascii']!r}")
            print(f"    -> name_has_non_latin: {nf['name_has_non_latin']}")
            print(f"  business_address: {rec.get('business_address')!r}")
            print(f"    -> address_normalized: {af['address_normalized']!r}")
            print(f"    -> street_number     : {af['street_number']!r}")
            print(f"    -> postal_code       : {af['postal_code']!r}")
            print(f"    -> city_candidate    : {af['city_candidate']!r}")
            print(f"    -> state_candidate   : {af['state_candidate']!r}")
            print(f"    -> landmark_text     : {af['landmark_text']!r}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalization stage CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="Sample ~n rows and print normalized examples")
    p_sample.add_argument("--source", required=True, choices=list(config.SOURCE_FILES))
    p_sample.add_argument("--n", type=int, default=1000)
    p_sample.add_argument("--show", type=int, default=8)

    p_run = sub.add_parser("run", help="Normalize a full source file to Parquet")
    p_run.add_argument("--source", required=True, choices=list(config.SOURCE_FILES) + ["all"])
    p_run.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)

    sub.add_parser("demo", help="Print curated examples for each noise pattern")

    args = parser.parse_args()
    if args.cmd == "sample":
        run_sample(args.source, n=args.n, show=args.show)
    elif args.cmd == "run":
        sources = list(config.SOURCE_FILES) if args.source == "all" else [args.source]
        for s in sources:
            run_full(s, batch_size=args.batch_size)
    elif args.cmd == "demo":
        run_demo()


if __name__ == "__main__":
    main()
