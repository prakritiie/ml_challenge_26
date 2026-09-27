"""Central configuration: paths, schema constants, and normalization lookup
tables (legal-suffix equivalences, address abbreviations, missing-value
markers). Nothing here calls out to the network or any external database --
every table is a static, hand-authored list derived from patterns observed
in the EDA of this challenge's own data.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at:
#   <student_resource>/code/business_entity_resolution/src/config.py
# so parents[3] resolves to <student_resource>/, which is where dataset/
# lives in the layout the challenge provides. Both roots can be overridden
# with environment variables if the code is relocated or run against a
# different data drop.
_THIS_FILE = Path(__file__).resolve()
_STUDENT_RESOURCE_DIR = _THIS_FILE.parents[3]

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(_STUDENT_RESOURCE_DIR / "dataset")))
DERIVED_ROOT = Path(os.environ.get("DERIVED_ROOT", str(_STUDENT_RESOURCE_DIR / "derived")))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", str(_STUDENT_RESOURCE_DIR / "output")))

# Final-pipeline (TF-IDF retrieval + LightGBM scoring) intermediate files:
# per-target retrieved candidates and per-target scored/decoded TSVs, kept
# separate from BLOCKING_DIR below (which belongs to the earlier,
# superseded rule-based blocking approach).
TFIDF_RETRIEVAL_DIR = DERIVED_ROOT / "tfidf_retrieval"
SCORING_TMP_DIR = DERIVED_ROOT / "scoring_tmp"

SOURCE_FILES = {
    "train_source1": DATA_ROOT / "train" / "train_source1.tsv",
    "train_source2": DATA_ROOT / "train" / "train_source2.tsv",
    "train_source3": DATA_ROOT / "train" / "train_source3.tsv",
    "test_source1": DATA_ROOT / "test" / "test_source1.tsv",
    "test_source2": DATA_ROOT / "test" / "test_source2.tsv",
    "test_source3": DATA_ROOT / "test" / "test_source3.tsv",
}

TRAIN_GROUND_TRUTH = DATA_ROOT / "train" / "train_ground_truth.tsv"

# Derived (normalized) outputs, one Parquet file per source file above.
# Parquet is used instead of TSV because several derived columns are lists
# (name_tokens, name_sorted_tokens, address_tokens) which do not round-trip
# cleanly through a comma/tab-delimited text file.
NORMALIZED_DIR = DERIVED_ROOT / "normalized"
NORMALIZED_FILES = {key: NORMALIZED_DIR / f"{key}.parquet" for key in SOURCE_FILES}

# Blocking-stage outputs: one candidate-pairs Parquet per (S1 -> target)
# direction, kept separate for train_source2 and train_source3 (never
# combined into a single S2+S3 index).
BLOCKING_DIR = DERIVED_ROOT / "blocking"
DUCKDB_TEMP_DIR = DERIVED_ROOT / "duckdb_tmp"

# ---------------------------------------------------------------------------
# Blocking-stage tuning
# ---------------------------------------------------------------------------
# Document-frequency ceiling for the "strong name-token" blocking rule: a
# token is only used as a blocking key if its document frequency is <= this
# value on BOTH the S1 side and the target side (see blocking.py). Chosen
# empirically, not just from raw frequency stats: an initial threshold of
# 1000 (based only on looking at the token-frequency distribution, where the
# ~1,000-1,500 most frequent tokens per source account for ~13M of the total
# token occurrences) turned out to still cause a severe blow-up in practice
# -- a 5,000-entity sanity sample produced 1.3M candidate pairs from this
# rule alone (up to ~2,400 candidates for a single S1 entity), driven not by
# legal suffixes (already far above 1000 and excluded) but by moderately
# common surnames/business nouns ("weaver", "sandoval", "bankshares") with
# df in the hundreds that still combine combinatorially across a name's
# several tokens. Measuring recall-added vs. pairs-added on that same sample
# at several thresholds showed steeply diminishing returns above ~100
# (100->1000 roughly triples candidate volume for a relatively small
# additional recall gain), so 100 is used as the conservative cutoff.
TOKEN_DF_THRESHOLD = 100

# Tokens shorter than this are dropped from the strong-token rule regardless
# of frequency -- short tokens (e.g. single letters) are inherently
# non-discriminating and are almost always high-frequency anyway.
TOKEN_MIN_LEN = 3

# Conservative DuckDB resource limits for the blocking joins, given this
# machine's available RAM. DuckDB spills to DUCKDB_TEMP_DIR on disk rather
# than erroring once memory_limit is hit.
DUCKDB_MEMORY_LIMIT = "6GB"
DUCKDB_THREADS = 4

# Options used everywhere we read one of the raw TSVs with DuckDB. Columns
# are read as VARCHAR because entity_id / business_name / business_address /
# country are all string columns and we don't want DuckDB's type sniffer
# guessing numeric types for values like "448 37, Gates, NC".
CSV_READ_OPTIONS = dict(delim="\t", header=True, quote="", escape="", all_varchar=True)

# Row count processed per Arrow record batch. This bounds peak memory: only
# BATCH_SIZE rows (plus the derived columns computed from them) are ever
# resident in Python/pandas/pyarrow memory at once, regardless of how many
# millions of rows are in the source file.
BATCH_SIZE = 200_000

# ---------------------------------------------------------------------------
# Missing-value markers
# ---------------------------------------------------------------------------
# EDA found business_address containing the literal strings "None"/"null" in
# addition to true empty cells. These must be treated as missing, not as text.
MISSING_TOKENS = {"none", "null", "n/a", "na", "nil", "nan", "-", ""}

# ---------------------------------------------------------------------------
# Legal suffix equivalence table (used to build name_core)
# ---------------------------------------------------------------------------
# Each entry maps a canonical legal-entity concept to the spelling variants
# seen in the data. Dotted acronyms ("L.L.C.", "P.C.") are squashed to their
# undotted form ("llc", "pc") by normalization.squash_dotted_acronyms() as a
# pre-processing step, so e.g. "llc": ["llc"] transparently also covers
# "L.L.C." / "L.L.C" / "l.l.c." input -- we don't need to enumerate every
# punctuation variant here.
LEGAL_SUFFIX_GROUPS = {
    "private": ["pvt", "private"],
    "limited": ["ltd", "limited"],
    "corporation": ["corp", "corporation"],
    "incorporated": ["inc", "incorporated"],
    "llc": ["llc"],
    "pllc": ["pllc"],
    "llp": ["llp"],
    "lp": ["lp"],
    "company": ["co", "company"],
    "plc": ["plc"],
    "pc": ["pc"],
}

# ---------------------------------------------------------------------------
# Address abbreviation expansion table
# ---------------------------------------------------------------------------
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "twp": "township",
}
# "st" -> "street" is handled separately in normalization.py: a leading "St"
# is frequently "Saint" (e.g. "St. Louis"), so it is only expanded when it is
# NOT the first token of the address -- a cheap, deterministic heuristic
# documented as a known limitation in README.md.
#
# "dr" -> "drive" is also handled separately: "Dr" is frequently the title
# "Doctor" as part of a person's name that the road is named after (e.g.
# "Dr Rajkumar Road", "Dr Martin Luther King Jr Blvd" -- extremely common in
# Indian and US addresses alike), whereas the street-type "Drive" abbreviation
# almost always sits at the END of its comma-separated segment (e.g.
# "123 Main Dr" / "123 Main Dr, Springfield, IL"). "dr" is therefore only
# expanded to "drive" when it is the LAST word before a comma or the end of
# the address string; a "Dr" followed by more words (a name, another street
# word) is left untouched. Not perfect (e.g. "123 Old Dr" as a genuine
# surname at the very end would still be misread as Drive), but it fixes the
# common, high-frequency failure mode without a broad find/replace.
#
# "no" -> "number" is also handled separately, and only when immediately
# followed by a digit (e.g. "No 58" -> "number 58"), so the common English
# word "no" is never corrupted.

# Phrases that introduce a landmark reference rather than a structural
# address component. Extracted into landmark_text instead of being deleted.
LANDMARK_KEYWORDS = [
    "near", "opp", "opposite", "behind", "adjacent to", "next to", "beside", "close to",
]
