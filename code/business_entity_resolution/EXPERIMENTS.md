# Business Entity Resolution — Pipeline Code

## Purpose

This is the ML Challenge 2026 "Business Entity Resolution" pipeline: given
business records from three independent, noisy sources (Source 1 is the
deduplicated reference set), find all Source 2 / Source 3 records that refer
to the same real-world business as each Source 1 record.

## Dataset

Provided by the challenge as tab-separated files under `dataset/` (a sibling
of this `code/` directory, i.e. `<student_resource>/dataset/`):

```
dataset/train/train_source1.tsv        2,206,821 rows
dataset/train/train_source2.tsv        5,034,616 rows
dataset/train/train_source3.tsv        5,285,603 rows
dataset/train/train_ground_truth.tsv   (source1_entity_id -> matched_entity_ids)
dataset/test/test_source1.tsv          ~1.7M rows
dataset/test/test_source2.tsv          ~4.9M rows
dataset/test/test_source3.tsv          ~5.1M rows
```

Each source file has columns `entity_id, business_name, business_address,
country`. Training countries are US and India; the test set additionally
includes France. ~26M rows total across all files — see the project's EDA
notes for the full noise-pattern analysis (abbreviations, punctuation,
word-order swaps, typos, transliteration to Devanagari/Tamil/Kannada/
Malayalam, landmark-based addresses, partial addresses, missing values
encoded as literal `"None"`/`"null"` strings, etc.).

## Current pipeline stage: **normalization + blocking V1**

Implemented so far:

- `src/normalization.py` — deterministic, dataset-only text normalization
  of `business_name` and `business_address`.
- `src/blocking.py` — Blocking V1: candidate-pair generation, S1→S2 and
  S1→S3 kept as separate indexes.
- `src/blocking_eval.py` — recall/coverage evaluation of a candidate set
  against `train_ground_truth.tsv`.
- `tests/test_normalization.py` — regression tests for the normalization
  stage (run with `python -m unittest tests.test_normalization -v`).

**Not yet implemented** (future stages): similarity feature engineering, the
pair-matching model, and prediction / output-file generation. Do not assume
any of those exist yet.

### What normalization produces

For each record, in addition to the original `entity_id`, `business_name`,
`business_address`, `country`:

**Name fields**
| Column | Meaning |
|---|---|
| `name_normalized` | Casefolded, whitespace/punctuation-cleaned full name. Legal suffixes are kept. |
| `name_core` | `name_normalized` with a recognized legal suffix (Pvt/Private, Ltd/Limited, Corp/Corporation, Inc/Incorporated, LLC, PLLC, LLP, LP, Co/Company, PLC, PC — see `config.LEGAL_SUFFIX_GROUPS`) stripped from the end. Falls back to the full name if stripping would remove everything. |
| `name_tokens` | Word tokens of `name_normalized` (Unicode letter/mark/number runs). |
| `name_sorted_tokens` | Same tokens, alphabetically sorted, so word-order swaps become comparable. |
| `name_ascii` | Non-Latin characters stripped (no transliteration attempted). |
| `name_has_non_latin` | True if the original name contains any non-ASCII character. |

**Address fields**
| Column | Meaning |
|---|---|
| `address_normalized` | Casefolded, cleaned address with common abbreviations expanded (Rd→Road, Ave→Avenue, No 5→Number 5, etc.). |
| `address_tokens` | Word tokens of `address_normalized`. |
| `address_ascii` | Non-Latin characters stripped. |
| `address_has_non_latin` | True if the original address contains any non-ASCII character. |
| `street_number` | Leading digit/unit sequence from the first comma-separated segment (heuristic). |
| `postal_code` | 5-6 digit code found in the address components *after* the first (street) segment, so a numeric street number is never mistaken for a ZIP/PIN. |
| `city_candidate` / `state_candidate` | The last two non-numeric comma-separated components — a positional **candidate**, not verified against any gazetteer. |
| `landmark_text` | The substring starting at a landmark keyword (near/opp/opposite/behind/adjacent to/next to/beside/close to) up to the next comma, kept separate from `address_normalized` rather than deleted. |

All lookup tables (`LEGAL_SUFFIX_GROUPS`, `ADDRESS_ABBREVIATIONS`,
`MISSING_TOKENS`, `LANDMARK_KEYWORDS`) live in `src/config.py` and are
static, hand-authored, and derived only from patterns observed in this
challenge's own data — no external database, geocoding, or business lookup
is used anywhere in this stage (or planned for any later stage).

## Output location

Normalized data is written as **Parquet** (not TSV) to:

```
<student_resource>/derived/normalized/<source_key>.parquet
```

e.g. `derived/normalized/train_source1.parquet`. Parquet is used because
several derived columns (`name_tokens`, `name_sorted_tokens`,
`address_tokens`) are lists, which don't round-trip cleanly through a
comma/tab-delimited text file. The `derived/` directory is a build artifact,
kept separate from `dataset/` — **the original TSVs are never modified.**

Both `DATA_ROOT` (default `<student_resource>/dataset`) and `DERIVED_ROOT`
(default `<student_resource>/derived`) can be overridden with environment
variables if you relocate the code or dataset.

## How to run

From this directory (`code/business_entity_resolution/`):

```bash
pip install -r requirements.txt

# Print normalized examples from a ~1000-row sample of one source
python -m src.normalization sample --source train_source1 --n 1000

# Print curated examples covering each noise pattern found in the EDA
# (legal suffixes, punctuation noise, ALL CAPS, None/null addresses,
# non-Latin scripts, landmark addresses, partial addresses)
python -m src.normalization demo

# Normalize one full source file to Parquet (streamed, constant memory)
python -m src.normalization run --source train_source1

# Normalize all six source files
python -m src.normalization run --source all
```

Valid `--source` values: `train_source1`, `train_source2`, `train_source3`,
`test_source1`, `test_source2`, `test_source3`.

## Scalability notes

- Rows are never fully materialized into pandas. `src/io_utils.py` reads
  each TSV through DuckDB and streams it out as Arrow `RecordBatch`es
  (`BATCH_SIZE = 200,000` rows by default, configurable), so peak memory is
  bounded by one batch's worth of rows, not the file size.
- `normalize_file()` consumes that batch stream, applies the (pure-Python)
  normalization functions to each batch, and writes each result batch
  incrementally to the output Parquet file via `pyarrow.parquet.ParquetWriter`
  — so no output is buffered in memory either.
- `sample_rows()` uses DuckDB's single-pass reservoir sampling
  (`USING SAMPLE n (reservoir)`) rather than a full sort or full load, so
  even the "print 1000 examples" path never scans more of the file than
  necessary.
- The same code path (`normalize_file`) works unmodified for any of the six
  source files — only the `--source` argument changes.

## Blocking V1

Goal: for every Source1 entity, produce a candidate set of Source2/Source3
records that is small enough for a matching model to score, while missing as
few true matches as possible (blocking recall is the ceiling on everything
downstream — a match that never becomes a candidate can never be predicted).

**S1→S2 and S1→S3 are built and stored as two completely separate indexes**
(never unioned into one S2+S3 table), per the project's design constraints.
Country is treated as an open string throughout — no rule hardcodes `{US,
India}`, so the same code works unmodified once France appears in the test
set.

### Rules (all equi-joins in DuckDB SQL — never an all-pairs comparison)

| Rule | Key | Notes |
|---|---|---|
| `exact_name` | `country + name_normalized` | |
| `core_name` | `country + name_core` | legal suffix stripped on both sides |
| `sorted_tokens` | `country + name_sorted_tokens` | catches word-order swaps |
| `strong_token` | `country + one name token` | only tokens whose document frequency is `<= TOKEN_DF_THRESHOLD` (config.py) **on both the S1 side and the target side** |
| `postal_code` | `country + postal_code` | only ~1% of rows have one (see normalization limitations) |

All five rules run independently against the full target table and are
UNIONed (deduplicated) into the final candidate set; `candidates_by_rule_*`
keeps the per-rule attribution for evaluation.

### `strong_token` threshold: how `TOKEN_DF_THRESHOLD = 100` was chosen

Token frequency was analyzed on `train_source2`/`train_source3` first, as
required: median token document-frequency is 1 (a long discriminating tail
of ~800K distinct tokens), but a first attempt at a threshold of 1000
(picked by eyeballing the raw frequency distribution) caused a severe
blow-up in practice — a 5,000-entity sanity sample produced **1.3M candidate
pairs from this rule alone**, up to ~2,400 candidates for a single S1
entity. The cause wasn't legal suffixes (already far above 1000 and
excluded) but moderately common surnames/business nouns (`"weaver"`,
`"sandoval"`, `"bankshares"`) with document frequency in the hundreds that
still combine combinatorially across a name's several tokens.

Measuring **recall added vs. pairs added** at several thresholds on that
same sample (S1→S2, recall of `strong_token` layered on top of the other
four rules) showed steeply diminishing returns above ~100:

| threshold | extra pairs vs. base | extra true matches found |
|---|---|---|
| 50 | +21,133 | +587 |
| 100 | +74,648 | +962 |
| 200 | +137,034 | +1,176 |
| 500 | +312,714 | +1,372 |
| 1000 | +1,310,203 | +1,552 |

`100` was chosen as the conservative cutoff (a token must appear at most
100 times in both S1 and the target source, and be at least 3 characters).
This is a first-pass, empirically-checked choice, not a final one — see
"Next iteration" below.

### V1 results on the full training set (2,206,821 S1 entities)

Full metrics (per-rule and per-target) are reproducible via
`python -m src.blocking_eval` and are not repeated in full here; headline
numbers:

| Target | Union candidate pairs | Avg / median / p95 / max candidates per S1 entity | Reduction ratio | Recall |
|---|---|---|---|---|
| S1→S2 | 72,763,181 | 32.97 / 12 / 107 / 759 | 99.9993% | **59.77%** |
| S1→S3 | 76,083,683 | 34.48 / 13 / 106 / 671 | 99.9993% | **58.61%** |

**Overall blocking recall (S2 ∪ S3): 59.17%** (4,519,753 / 7,638,365 true
matches recovered). Recall is essentially uniform across entity segments
(only-S2 59.76%, only-S3 58.80%, both-S2-and-S3 59.17%) — V1 is not
systematically worse for any one segment. **82.26% of true singleton
entities (no ground-truth match) still receive at least one candidate**
(avg 66, median 28), which is expected at this stage but means the matching
model carries the full precision burden for those.

Missed matches are dominated by three patterns none of these rules can
catch: (1) cross-script transliteration (`"Urban Services LLP"` vs. its
Devanagari transliteration, sharing zero Latin tokens), (2) typos that
change token identity rather than casing/punctuation (`"Stokley"` vs.
`"Sótkby"`), and (3) DBA/alias names with only partial token overlap that
falls outside the token-frequency window (`"Rizariza Sys DBA: Unique
Entertainments Private Limited"` vs. `"Unique Entertainments Private
Limited"`).

### Output location

```
<student_resource>/derived/blocking/candidates_by_rule_<target>.parquet   # (source1_entity_id, candidate_id, rule) -- not deduplicated across rules
<student_resource>/derived/blocking/candidates_<target>.parquet          # deduplicated union -- the candidate set for the matching model
```
`<target>` is `train_source2` or `train_source3`.

### How to run

```bash
# Sanity-test mode: restrict S1 to a small reservoir sample, target source stays full
python -m src.blocking --target train_source2 --s1-sample 5000
python -m src.blocking --target train_source3 --s1-sample 5000

# Full run (one process per target -- never run both concurrently on this
# machine; DuckDB's join memory footprint is much larger than normalization's)
python -m src.blocking --target train_source2
python -m src.blocking --target train_source3

# Evaluate recall against train_ground_truth.tsv (per-rule, per-target, and overall)
python -m src.blocking_eval
```

### Scalability / resource notes

- All candidate generation is SQL executed by DuckDB directly against the
  normalized Parquet files — no Python loop ever touches a full source
  table, and results are streamed to Parquet via `COPY (...) TO ... (FORMAT
  PARQUET)` rather than being materialized as Python lists.
- `io_utils.get_configured_connection()` caps DuckDB at `DUCKDB_MEMORY_LIMIT`
  (6GB) with `DUCKDB_TEMP_DIR` configured for disk spilling, since this
  machine has limited free RAM. In the full runs performed, DuckDB stayed
  within the memory cap (no spill files were produced).
- **Never run two blocking processes concurrently on this machine.** Unlike
  the lightweight per-row Python work in normalization (~360MB peak per
  process), DuckDB's join memory footprint is large enough that running
  `train_source2` and `train_source3` blocking at the same time pushed free
  system RAM close to zero; they were run sequentially instead. Each target's
  process only ever writes to its own `candidates_*_<target>.parquet` file,
  so there is no shared-file write hazard either way.

### Next iteration (not yet implemented)

- The three biggest missed-match categories above (transliteration, token-
  altering typos, DBA/alias names) each call for a different fix:
  edit-distance/phonetic blocking for typos (e.g. token n-grams or a
  Jaro-Winkler-filtered pass), a same-address-different-name signal for
  DBA/alias cases (blocking on address alone when postal_code/street_number
  + city_candidate all agree), and — separately from blocking — a
  transliteration or cross-script embedding signal for the matching stage,
  since blocking on tokens alone structurally cannot bridge two disjoint
  scripts.
- `city_candidate`/`state_candidate` were deliberately *not* used as a
  blocking key in V1 per the task's instruction not to depend on the
  positional address heuristics alone; combining `city_candidate` (or
  `state_candidate`) with a name signal (rather than address alone) is worth
  testing once the matching-model stage can absorb the extra candidates.
- 59% recall means ~41% of true matches are structurally unreachable by the
  matching model as V1 stands; before investing further in the ML stage it
  is worth raising blocking recall, since no classifier can recover a match
  that was never generated as a candidate.

## Blocking V2 experiments (evaluation only — not yet adopted)

Three candidate additions to V1 were each evaluated **independently** against
a fixed 20,000-entity reservoir sample of S1 (`src/blocking_v2_experiments.py`,
`--s1-sample`), matched against the full S2/S3 tables, so recall/candidate
comparisons are apples-to-apples without re-running the full V1 build per
experiment. Ground truth is restricted to the sampled S1 entities' matches
for a valid recall denominator (an early version of this harness computed
recall against ALL 3.7-3.9M matches while only generating candidates for
20,000 entities, making every rule look useless — a bug caught before
drawing any conclusions, see git history / commit messages). Sample-based V1
recall (59.68% S2 / 58.39% S3) matches the full-scale run (59.77% / 58.61%)
closely, validating the methodology.

**None of these are wired into `blocking.py` yet.** This section reports
findings only, per the instruction to evaluate additions independently
before combining anything into a new blocking version.

### Experiment A — character n-gram blocking (typo/OCR tolerance)

4-character grams of `name_core`, symmetric document-frequency filter (same
design as V1's `strong_token`, threshold chosen the same way: an initial
frequency-only look suggested a high threshold, but measuring recall-added
vs. pairs-added at several thresholds on a sample showed steep diminishing
returns above ~50, so `TOKEN_DF_THRESHOLD`-style symmetric cap of 50 was used).

| Target | V1 recall | rule alone recall | V1+rule recall | additional pairs | additional matches | recall/1M added pairs |
|---|---|---|---|---|---|---|
| S1→S2 | 59.68% | 11.90% | 61.14% | 68,770 | 489 | 7,111 |
| S1→S3 | 58.39% | 11.47% | 59.85% | 65,407 | 524 | 8,011 |

Recovers real typo/OCR cases (`"FZ Cion"`/`"FZ Ci0n"`, `"Vv Manufacturing"`/
`"Vv Mánufacturing"`) and, as a bonus, concatenated-no-space name variants
(`"Aerocity Nursing Home"`/`"aerocitynursinghome.com"`). Gain is modest
(+1.2-1.5 recall points) — worthwhile but not transformative on its own.
Still misses cases where the frequency filter happens to exclude every
n-gram around the one character that differs (seen with a common
surname+typo combination).

### Experiment B — address-assisted blocking

**Reliability was measured directly from ground-truth true-match pairs
first** (not assumed), by checking how often each field agrees between S1
and its known match:

| Field | Coverage (both sides non-null) | Agreement given both present (S2 / S3) |
|---|---|---|
| `postal_code` | ~0.1% | 91.0% / 93.2% |
| `street_number` | ~64% | 76.7% / 77.8% |
| `city_candidate` | ~95.5% | **52.3% / 58.9%** |
| `state_candidate` | ~95.5% | 67.2% / **5.7%** |

`postal_code` is reliable but has almost no coverage. `city_candidate` is
high-coverage but agrees only about half the time — confirmed unreliable, as
warned. **`state_candidate` for train_source3 is only 5.7% reliable** — a
concrete, measured instance of the "reversed component order" risk called
out in the task, specific to one target source.

Three combinations were tested (S1 sample vs. full target):

| Rule | Target | rule alone recall | V1+rule recall | additional pairs | additional matches | recall/1M added |
|---|---|---|---|---|---|---|
| `postal_code` only | S2 | 0.10% | 59.68% (no change) | 0 | 0 | — |
| `postal_code` only | S3 | 0.13% | 58.39% (no change) | 0 | 0 | — |
| `postal_code + street_number` | S2 | 0.01% | 59.68% (no change) | 0 | 0 | — |
| `postal_code + street_number` | S3 | 0.03% | 58.39% (no change) | 0 | 0 | — |
| **`city_candidate + street_number`** | **S2** | **32.32%** | **73.05%** | 511,405 | 4,473 | 8,746 |
| **`city_candidate + street_number`** | **S3** | **36.70%** | **74.33%** | 620,498 | 5,710 | 9,202 |

`postal_code`-based combinations add **zero** recall beyond V1 on the sample
(V1 already includes a `postal_code` rule, and there are too few rows where
S1 also has a `street_number` match to move the needle at this sample size).

`city_candidate + street_number` is the single biggest lever found across
all of V2: **+13.4 points (S2) / +15.9 points (S3) recall**, at a better
recall-per-candidate ratio than the n-gram rule, and it recovers DBA/alias
cases the name-based rules structurally cannot (e.g. `"Kirk, Celie, D.C.,
M.D., P.C."` vs `"#kirkcelie"`) plus, as a side effect, some cross-script
cases too (address matching doesn't depend on the name at all, so it
recovered `"Royal Power"` vs its Devanagari transliteration `"रॉयल पावर"`).

**However, this rule needs a frequency guard before adoption**: the max
candidate count for a single entity was 2,030 (S2 sample), traced to a
pathological `city_candidate="mumbai", street_number="1"` collision shared
by thousands of unrelated businesses — almost certainly an extraction
artifact (many Indian addresses' first segment doesn't start with a real
street number), not real address information. A document-frequency cap on
the `(city_candidate, street_number)` key, exactly like V1's `strong_token`
guard, is the recommended next step before adopting this rule.

### Experiment C — cross-script transliteration (analysis only, not implemented)

Scripts detected in `business_name` (Unicode block ranges), train_source2/3:

| Script | S2 | S3 |
|---|---|---|
| Devanagari | 5.35% | 2.99% |
| Telugu | 0.78% | 0.44% |
| Kannada | 0.74% | 0.42% |
| Tamil | 0.67% | 0.37% |
| Gujarati | 0.61% | 0.34% |
| Bengali | 0.61% | 0.34% |
| Malayalam | 0.37% | 0.21% |
| Odia | 0.15% | 0.08% |
| Gurmukhi | 0.13% | 0.08% |

This is **material, not a long-tail curiosity**: among ground-truth true
matches, **15.8% (S2) / 12.1% (S3) are cross-script** (S1 side is always
Latin; the matched record is in a native script) — over 1 million true
matches combined. No blocking rule that operates on Latin tokens/n-grams
can ever recover these; per the task's own framing, a matcher-stage feature
cannot help either if the pair was never generated as a candidate.

**Feasibility of a local, deterministic transliteration**: the observed
transliterations are highly regular, direct phonetic renderings, word for
word, that even preserve the legal-suffix boundary (`"Limited"` ->
`"लिमिटेड"`, `"Private"` -> `"प्राइवेट"`). This regularity means a **static
character-mapping table per script** (consonant/vowel/matra -> Latin
phonetic fragment — the same kind of fixed, hand-authored lookup already
used for `LEGAL_SUFFIX_GROUPS`/`ADDRESS_ABBREVIATIONS`, not a live
service) is plausible. A quick proof-of-concept Devanagari->Latin map
(not committed to the codebase) produced:

```
गोल्डन ग्लोबल प्राइवेट लिमिटेड  -> goldana globala praaiveta limiteda   (true: "Golden Global Private Limited")
ब्लैक इंफोटेक प्रा. लि.           -> blaika iphoteka praa li               (true: "Black Infotech Pvt Ltd")
```

Close enough that character n-gram overlap with the true Latin name would
recover some (not all) shared fragments (`"limiteda"`/`"limited"` share
3+ four-grams), but not exact/token-equal — this only becomes useful
combined with the same fuzzy/n-gram blocking technique from Experiment A,
not as an exact-match key. A real implementation needs: (a) one table per
script (9 scripts observed, each with its own consonant/vowel inventory —
meaningful engineering effort), (b) handling schwa deletion (the crude demo
above keeps a spurious trailing "a" that modern Hindi drops in speech,
e.g. "goldana" vs "golden"), and (c) validation that it does not hurt
precision by inventing spurious overlaps.

**Recommendation**: do not implement now (confirms the task's default), but
flag as the highest-value next investment given ~1M+ affected true matches
dwarf both Experiment A and B's gains — the honest ceiling on any matcher
trained on V1+A+B candidates is still capped by this ~12-16% unreachable
slice per target source.

### Net effect if A + B (address, unguarded) were both adopted

Not directly tested as a combined union in this round (per the instruction
to evaluate additions independently first), but the two are largely
orthogonal signal sources (name n-grams vs. address), so their overlap in
newly-recovered matches is likely small — a combined V1+A+B run is a
natural next step once the `city_candidate+street_number` frequency guard
(above) is added.

## Blocking V2 combined evaluation (evaluation only — not yet adopted)

Follow-up to the V2 experiments above: (1) added a data-driven guard to the
`city_candidate + street_number` rule, then (2) evaluated four configurations
(A=V1, B=V1+ngram, C=V1+guarded address, D=V1+ngram+guarded address)
independently, (3) analyzed how much the ngram and address additions
overlap, and (4)/(5) categorized what's still missed. Code:
`src/blocking_v2_experiments.py` (guard function), `src/blocking_v2_combined.py`
(A/B/C/D + overlap + miss categorization). Same fixed 20K S1 sample as
before; S2 and S3 run as separate sequential processes. Nothing here is
wired into `blocking.py` yet.

### Step 1: guarding the address rule

The raw `city_candidate + street_number` rule's blow-up (max 2,030
candidates for one entity) traced to `city_candidate="mumbai",
street_number="1"`. Two **data-driven** guards were added (neither
references any specific city):

1. **Minimum street_number length of 3.** Measured directly: average
   composite-key frequency falls from ~8,700 (1-digit street numbers) to
   ~1,200 (2-digit) to ~280 (3-digit) to ~40 (4-digit) — short numeric
   street numbers are structurally non-discriminating regardless of city.
2. **Symmetric composite-key document-frequency cap of 50** (same design as
   V1's `strong_token` guard): the `(country, city_candidate,
   street_number)` key must occur at most 50 times in both the S1 sample
   and the target source.

Measured trade-off (S1→S2, standalone rule): guard removes **93.5% of the
raw rule's candidate pairs** (517,822 → 33,834) while keeping **77% of its
standalone recall** (8,320 of 10,815 matches) and capping the max candidate
count at 50 (from 2,030). A stricter length-4 guard was also tested and
rejected — it loses more than half the rule's recall for little additional
pair reduction beyond length-3.

### Step 2: A/B/C/D, evaluated separately for S2 and S3

| Config | S2 recall | S2 pairs (+over V1) | S2 matches +over V1 | S3 recall | S3 pairs (+over V1) | S3 matches +over V1 |
|---|---|---|---|---|---|---|
| A. V1 baseline | 59.68% | 651,271 (+0) | +0 | 58.39% | 680,705 (+0) | +0 |
| B. V1 + ngram | 61.14% | 720,041 (+68,770) | +489 | 59.85% | 746,112 (+65,407) | +524 |
| C. V1 + guarded address | **69.42%** | 680,002 (+28,731) | +3,257 | **70.18%** | 715,304 (+34,599) | +4,223 |
| D. V1 + ngram + guarded address | **70.52%** | 748,650 (+97,379) | +3,627 | **71.21%** | 780,554 (+99,849) | +4,594 |

Recall-per-million-additional-candidates: B ≈ 7,100-8,000/M, **C ≈
113,000-122,000/M** (the guard didn't just shrink the rule, it made it
*far* more efficient than the unguarded version's 8,746-9,202/M), D ≈
37,000-46,000/M (blended, as expected between B and C's efficiency).

Candidate-count distribution (avg/median/p95/max per S1 entity, over all
20,000 sampled entities including zero): config D reaches avg 37.4-39.0,
median 19-25, p95 113-115, max 658-739 (max is set by V1's own worst-case
entity, unchanged by adding the guarded rules).

**Union across S2+S3** (combined found / combined total, matching the V1
full-scale reporting convention):

| Config | Combined recall | Combined pairs | Additional pairs | Additional matches | Recall/M added |
|---|---|---|---|---|---|
| A | 59.01% | 1,331,976 | 0 | 0 | — |
| B | 60.47% | 1,466,153 | +134,177 | +1,013 | 7,550 |
| C | 69.81% | 1,395,306 | +63,330 | +7,480 | 118,111 |
| **D** | **70.88%** | 1,529,204 | +197,228 | +8,221 | 41,683 |

**D improves overall blocking recall from ~59% (V1) to ~71%** — a 12-point
gain — for a 15% increase in total candidate volume.

### Step 3: overlap between the ngram and address additions

| | S2 | S3 |
|---|---|---|
| V1 misses (total) | 13,491 | 14,905 |
| Recovered by ngram only | 370 | 371 |
| Recovered by address only | 3,138 | 4,070 |
| Recovered by both | 119 | 153 |
| Still missed after D | 9,864 | 10,311 |

The two rules are **largely complementary, not redundant**: only ~3% of
newly-recovered matches (119/3627 for S2, 153/4594 for S3) are found by
both rules. Address contributes ~8x more recovered matches than ngram, but
neither subsumes the other — both are worth keeping.

### Step 4: categorizing what config D still misses

Heuristic categorization (precedence order: cross-script check via
`name_has_non_latin` mismatch → missing address → small `name_core` edit
distance → low name-token Jaccard → high name-token Jaccard with
differing address → other), applied to a 500-row sample of each target's
remaining misses:

| Category | S2 | S3 |
|---|---|---|
| cross_script | 34.0% | 21.4% |
| typo_ocr | 22.8% | 18.6% |
| address_variation | 21.2% | 31.6% |
| alias_dba | 15.0% | 17.4% |
| other | 4.8% | 8.0% |
| missing_address | 2.2% | 3.0% |

This is a heuristic, not a ground-truth-labeled categorization — treat the
percentages as directional, not exact. `address_variation` misses (e.g.
`"White Media Private Limited"` vs `"White Media-Limited Center"`) suggest
the guard, while a large efficiency win, does cost some real recall on
plausible-but-not-identical address formatting; `typo_ocr` misses that
survive both V1 and the n-gram rule (e.g. `"South Quality Guggenheim"` vs
`"South Qililty Guggenheim"`) suggest the symmetric document-frequency
filter sometimes excludes every n-gram touching the corrupted region for
less-common names.

### Step 5: remaining cross-script misses (inspection only — not implemented)

| | S2 | S3 |
|---|---|---|
| Cross-script misses remaining after D | 3,597 | 2,711 |
| Scripts involved | Devanagari (1,472), Kannada (212), Telugu (201), Bengali (207), Tamil (193), Gujarati (166), Malayalam (102), Gurmukhi (52), Odia (48) | Devanagari (911), Kannada (134), Telugu (106), Tamil (99), Bengali (113), Gujarati (91), Malayalam (61), Gurmukhi (38), Odia (32) |

Examples: `"Anand Power Private Limited"` / `"आनंद पावर प्राइवेट लिमिटेड"`,
`"Bharat Producer Pvt Ltd"` / `"ഭാരത് പ്രൊഡ്യൂസർ പ്രൈവറ്റ് ലിമിറ്റഡ്"`.
Devanagari dominates (as in the raw script-frequency analysis), but all 9
scripts detected earlier still appear in the residual misses — address
blocking recovered some cross-script matches as a side effect (Experiment
A/B section above), but clearly not most of them, since address fields are
independently missing/non-matching for a large share of records too.

Confirms the Experiment C conclusion unchanged: a deterministic local
transliteration table remains feasible in principle (same regular,
word-for-word phonetic pattern observed here) and is still the largest
single remaining lever, but **it has not been implemented**, per instruction.

## V3 transliteration experiment (evaluation only — NOT adopted)

Tests whether a self-contained, deterministic Devanagari->Latin
transliteration, used as an *additional* representation, recovers cross-
script true matches without candidate explosion. Code:
`src/transliteration.py` (the mapping/function, fully documented per
component), `src/blocking_v3_transliteration.py` (evaluation harness).
Production `normalization.py`/`blocking.py` are untouched; no existing
field is overwritten anywhere.

### Step 1: script frequency, refined

The V2 "cross-script" figure (15.8%/12.1% of matches) was based on a coarse
`name_has_non_latin` flag, which also fires on accented-Latin text (e.g.
`"WÁRE"`). Restricting to true Indic-script characters gives a more honest
number: **9.33% (S2) / 5.24% (S3) of all true matches involve a genuine
Indic script** (~344K + ~207K matches). Devanagari alone accounts for
**56.7% of that Indic-script total in both S2 and S3** — more than 2x the
next-largest script (Telugu) — which is why it was chosen first, per the
instruction not to assume all 9 scripts need immediate implementation.

### Step 2: Devanagari prototype

Static character tables (documented in full in `transliteration.py`'s
docstring): consonants (with inherent "a", aspirated pairs like ख=kha),
independent vowels, matras (replace the inherent vowel), virama (suppresses
it for clusters), anusvara/chandrabindu (both -> "n", a simplification),
visarga (-> "h"), a small nukta table for loanword sounds (ज़=z, फ़=f,
etc.), Devanagari digits and danda. **Schwa deletion: deliberately not
implemented** — every consonant's inherent "a" is kept, per the instruction
to measure the simple version first.

### Step 3: A/B/C/D (S2/S3 separate; B = the approved V2 config)

| Config | S2 recall | S2 pairs (+over V1) | S2 matches +over V1 | S3 recall | S3 pairs (+over V1) | S3 matches +over V1 |
|---|---|---|---|---|---|---|
| A. V1 | 59.68% | 651,271 (+0) | +0 | 58.39% | 680,705 (+0) | +0 |
| B. V1+ngram+guarded address | 70.52% | 748,650 (+97,379) | +3,627 | 71.21% | 780,554 (+99,849) | +4,594 |
| **C. V1+translit (exact/core)** | **59.68%** | **651,271 (+0)** | **+0** | **58.39%** | **680,705 (+0)** | **+0** |
| D. B + translit | 70.52% | 748,650 (+97,379) | +3,627 | 71.21% | 780,554 (+99,849) | +4,594 |

**Config C is byte-for-byte identical to config A.** The transliteration
rule generated **zero** candidate pairs on both targets — confirmed not a
bug (269,424 target rows were transliterated for S2 alone; every one of
them was checked against the 20K S1 sample and matched nothing). D is
therefore identical to B — translit adds nothing on top.

### Step 4: cross-script-specific recall (the key metric for V3)

| | S2 | S3 |
|---|---|---|
| Total cross-script true matches (in-sample) | 3,158 | 1,903 |
| Captured by V2 baseline (config B) | 505 | 318 |
| Captured by translit alone | **0** | **0** |
| Captured by D (combined) | 505 (unchanged from B) | 318 (unchanged from B) |
| Still missed after D | 2,653 | 1,585 |

The cross-script matches B already recovers (505/318) come entirely from
the **guarded address rule** as a side effect (address blocking doesn't
depend on the name at all — see the V2 report's `"Royal Power"`/`"रॉयल
पावर"` example). Transliteration contributed nothing on top.

### Step 5: transliteration quality samples

35 examples pulled per target (full tables in the run logs); representative
rows from train_source2:

| S1 name (Latin) | Target (Devanagari) | Transliterated | Exact match? |
|---|---|---|---|
| City Tech Private Limited | सिटी टेक प्राइवेट लिमिटेड | sitii teka praaiveta limiteda | No |
| Green Infotech Private Limited | ग्रीन इंफोटेक प्राइवेट लिमिटेड | griina inphoteka praaiveta limiteda | No |
| Gold Systems LLP | गोल्ड सिस्टम्स एलएलपी | golda sistamsa elaelapii | No |
| Eastern Engineering Limited | ईस्टर्न इंजीनियरिंग लिमिटेड | iistarna injiiniyaringa limiteda | No |
| Big Hospitality | बिग हॉस्पिटैलिटी | biga haspitailitii | No |

Every single sampled pair fails exact match, and the reason is systematic
and visible in the table: `"Limited"` -> `"limiteda"`, `"Gold"` ->
`"golda"`, `"Green"` -> `"griina"` — the retained inherent vowel and
long/short vowel doubling (aa/ii/uu) touch nearly every syllable, so the
transliterated string is *close* to the English spelling but essentially
never identical.

### Step 6: candidate explosion analysis

Moot given Step 3's result: the exact-match rule produced 0 candidates
before any guard, so a document-frequency guard has nothing to remove
(0 pairs removed, 0 matches lost, since there were 0 to begin with).
The frequency analysis of the transliterated keys themselves is still
informative: 92,007 distinct `translit_normalized` keys from 269,424 rows
(S2), with the most frequent key occurring 68 times (e.g. multiple
`"Lakshmi ... Private Limited"`-pattern businesses transliterating
identically) — so *had* an exact-match hit occurred, explosion risk would
have been low; the actual problem is the opposite one (near-zero hits, not
too many).

**Supplementary check (exploratory, not a formal step):** given exact
match's clean failure, a character 4-gram version of the same
transliteration (mirroring Experiment A's design, symmetric df<=50 guard)
was tried instead of exact-string equality. Result: 17 matches recovered
from 65,033 candidate pairs (S1->S2) — a recall-per-million of ~261, worse
than every other rule measured in V1/V2/V3. This indicates the problem is
not "exact match is too strict" but that the **transliteration itself is
too systematically different** (extra vowels touch nearly every syllable)
for even fuzzy character-overlap matching to help much without a real fix.

### Step 7: decision report

| Metric | Value |
|---|---|
| Devanagari transliteration recall gain (exact/core rule) | **0.0 percentage points**, both targets |
| Cross-script recall gain from transliteration | **0 of 3,158 (S2) / 1,903 (S3)** in-sample cross-script matches |
| Candidate cost | 0 additional pairs (nothing generated) |
| Efficiency | undefined (0 matches / 0 pairs); exploratory n-gram variant measured at ~261 recall/M, far below every other rule tested |
| **Adopt?** | **No.** |

**Root cause, not just an outcome:** the deliberate no-schwa-deletion
policy (correctly following the instruction to measure the simple version
first) produces a systematically vowel-heavy Latin string that differs
from natural English spelling in nearly every syllable, not just at word
boundaries. This is why both exact-match AND fuzzy n-gram matching on top
of it fail — the gap isn't a matching-strategy problem, it's a
transliteration-fidelity problem.

**What already helps cross-script cases without any transliteration**: the
V2 guarded address rule recovers 505 (S2) / 318 (S3) cross-script matches
in-sample as a side effect, since it never looks at the name at all. This
is already captured in the approved V2 config and requires no further work.

**Recommendation**: do not adopt this Devanagari prototype. **Do not
implement additional scripts** — the measured evidence here (0 recall gain)
does not justify replicating this approach for Telugu/Kannada/Tamil/etc.
before fixing the underlying transliteration-fidelity problem. If cross-
script recall is pursued further, the necessary next step is a schwa-
deletion pass (or a coarser phonetic-normalization step, e.g. collapsing
vowel-length distinctions and dropping trailing inherent vowels at word
end) applied to the *existing* Devanagari-only prototype and re-measured
with the same harness, BEFORE spending effort on any other script — since
Devanagari alone is 56.7% of the addressable Indic-script opportunity, it
remains the right script to iterate on first if this path is pursued.

## V4 investigation (evidence only — no new blocking implemented)

Data-driven investigation of what could recover the true matches still
missed after V2 (V1 + ngram + guarded address), on the full (not sampled-
to-500) missed set from the fixed 20K S1 sample: **9,864 misses (S2)**,
**10,311 misses (S3)**. Code: `src/blocking_v4_investigation.py`. Nothing
here is wired into blocking.py — this is measurement only.

### AREA A — address variation

| Metric | S2 | S3 |
|---|---|---|
| Missing address (either side) | 3.9% | 4.2% |
| Both have address | 96.1% | 95.8% |
| Address token Jaccard ≥ 0.5 | 87.2% | 67.1% |
| Address token Jaccard ≥ 0.3 | 96.7% | 89.1% |
| Sorted address tokens exactly equal | 18.0% | 7.5% |
| Address char-4gram Jaccard ≥ 0.5 | 80.2% | 68.2% |
| Postal code both present | 0.02% | 0.02% |
| Street number both present / agree | 53.1% / 33.8% | 51.2% / 32.3% |
| City both present / agree | 96.1% / 37.0% | 95.8% / 40.5% |
| One address is a subset of the other | 48.9% | 11.1% |
| Landmark present either side | 6.6% | 6.1% |
| **Shares a rare address token** (target-side df ≤ 50 / ≤200 / ≤1000) | **72.5% / 85.3% / 93.2%** | **66.8% / 80.5% / 90.4%** |
| **Shares a rare address 4-gram** (df ≤ 50 / ≤200 / ≤1000) | **87.6% / 94.4% / 96.0%** | **85.7% / 94.0% / 95.7%** |

**This is the single strongest signal found in the entire V4 investigation.**
Full exact-string address equality (`sorted address tokens`) only covers
18.0%/7.5% — most misses have a *similar but not identical* address (extra/
missing/reordered components), which is exactly why the strict guarded
`city_candidate+street_number` rule doesn't catch them, but their **address
token/n-gram Jaccard is very high** (87%/67% at ≥0.5). Critically, **sharing
just one rare (low document-frequency) address token already covers
72.5%/66.8%** of misses even at a conservative df≤50 threshold — the same
threshold methodology already validated for name tokens/n-grams (frequency
measured against the FULL target population, not just the misses, so this
already accounts for real-world explosion risk). This generalizes V1's
narrow `postal_code` rule and V2's `city_candidate+street_number` heuristic
into a broader "rare address token" rule, much like `strong_token` does for
names.

Example (S2) — near-identical address, zero name overlap:
`"Sullivan, Vinita L., MD"` / `"2119 h street, sacramento, ca"` vs.
`"-- sullivanvinital.com"` / `"ca, 2119 h street, sacramento"` (address
Jaccard 1.0, name Jaccard 0.0). Another: `"Shree Impex Private Limited"` vs.
its Devanagari transliteration, addresses identical after reordering
(Jaccard 1.0) — address blocking would recover this cross-script case
without any transliteration.

Obvious failure case for a naive version of this rule: very common address
tokens (city/state names, "road", "floor") must be excluded via the
document-frequency filter — which is exactly what makes the 72-93%
coverage number meaningful (it already reflects that filter) rather than a
raw, unguarded token-overlap number.

### AREA B — typo/OCR

Restricted to same-script pairs with both `name_core` present (6,267 S2 /
7,600 S3 of the misses — the rest are cross-script or missing a core name):

| Metric | S2 | S3 |
|---|---|---|
| Edit distance ≤ 2 | 15.7% | 14.0% |
| Edit distance ≤ 4 | 32.9% | 29.5% |
| Edit distance ≤ 8 | 64.2% | 56.8% |
| Edit distance median | 6 | 7 |
| Char n-gram Jaccard ≥ 0.5 | 41.0% | 38.1% |
| Char n-gram Jaccard ≥ 0.3 | 72.3% | 69.7% |
| Common prefix ≥ 3 chars | 72.8% | 65.2% |
| Common suffix ≥ 3 chars | 30.0% | 32.9% |

Median edit distance of 6-7 is **larger than a simple 1-2 character typo**
— consistent with V1's n-gram rule (df≤50) already having captured the
easy, small-edit-distance cases, leaving behind harder, more substantially
different name pairs (`"Varun & Co"` / `"VRBEN & CO"`, edit distance 3;
`"Strategic Furniture, Inc"` / `"STRATEGIC FRNITRUR, (INC)"`, edit distance
3 but visually a bigger scramble). The **72.8%/65.2% common-prefix-≥3**
finding is notable and cheap to exploit: most of these pairs still start
the same way even when the rest of the name is corrupted. A prefix-based
blocking key (e.g. first 4-5 characters of `name_core`) is a plausible,
very cheap candidate signature — cheaper than n-grams — worth measuring
explicitly in a future iteration, though it would need its own frequency
guard (common prefixes like "the "/"new " would need exclusion).

### AREA C — alias/DBA/partial names

| Metric | S2 | S3 |
|---|---|---|
| One name is a substring of the other | 18.1% | 25.5% |
| Token-set containment (one ⊆ other) | 13.1% | 18.6% |
| Token Jaccard < 0.2 (very low overlap) | 43.5% | 33.4% |
| Longest-common-token-subsequence ratio ≥ 0.5 | 55.0% | 65.2% |
| Has an explicit DBA/AKA marker | 0.2% | 3.0% |
| Very low overlap AND no DBA marker | 43.5% | 33.3% |

**Confirms the task's caution against blindly stripping "DBA" text**:
explicit markers (`"dba"`, `"trading as"`, `"aka"`) are present in only
0.2%/3.0% of cases — almost none of the real alias/partial-name misses are
announced by a marker word. The 43.5%/33.4% "very low token overlap" group
is the true alias/DBA-like population, and it is large. Substring
containment (18.1%/25.5%) is a cheap, cross-checkable signal
(`"Dental Cascade Specialists Inc"` contains `"Dental Cascade"`;
`"Grand Technical Group"` / `"GRAND TECHNICAL"`) but only covers a fifth to
a quarter of this category on its own — most alias cases (e.g.
`"Mccloud's Trading"` / `"MT"`, `"Sullivan, Vinita L., MD"` /
`"sullivanvinital.com"`) share **no usable name signal at all**. These are
exactly the cases where Area A's address-token finding matters most: many
of the low-name-overlap examples above have address Jaccard 1.0 (see the
Area A examples — several rows appear in both lists), meaning **address is
the only remaining deterministic signal for the true alias/DBA cases**,
not a smarter name comparison.

### Cross-area synthesis

The three areas are not independent discoveries — they converge on one
conclusion: for the matches still missed after V2, **the address fields
already carry the recovery signal in the large majority of cases** (72-93%
share a rare address token; 87-96% share a rare address 4-gram), including
for cases categorized as cross-script, alias/DBA, or "address_variation" in
the earlier V2 miss taxonomy. Name-level signals (edit distance, n-grams,
substring/containment) help but cover a visibly smaller share and require
larger/more permissive thresholds to reach comparable coverage. **A
broadened, frequency-guarded "rare address token" rule is the most
promising concrete next blocking addition** to measure and validate before
any other new representation, followed distantly by a cheap name-prefix
signature for the typo/OCR residual.

Per the task's instructions, none of this has been built into a blocking
rule yet — this section reports evidence only.

## V5 controlled blocking experiments (evaluation only — not yet adopted)

Follows directly from the V4 investigation. All experiments use the fixed
20K S1 sample against full S2/S3 target tables, evaluated as an addition on
top of **V2** (V1 + ngram + guarded address). Code:
`src/blocking_v5_experiments.py`. Nothing here is wired into `blocking.py`.

**Incident during development**: an early version of the name-prefix
builder (Experiment C) had no pre-filter before its entity-level join and
hung/exploded on `prefix_len=3` — even though no single 3-letter prefix has
target-side frequency above ~46K, the *combination* across dozens of
moderately-common prefixes (each shared by hundreds of S1 sample entities)
summed to a join far too large to complete. Fixed by adopting the same
"pre-filter at a loose threshold, then join" pattern already used for
Experiments A/B (see `build_prefix_shared`'s docstring). This is the same
class of bug as V1's original `strong_token` blow-up — a reminder that this
"join once at the loosest interesting threshold, then cheaply re-slice"
pattern needs to be applied consistently to every new blocking key, not
assumed safe by analogy.

### Experiment A — rare address token blocking

| Threshold | S2 recall | S2 +pairs | S2 +matches | S2 recall/M | S3 recall | S3 +pairs | S3 +matches | S3 recall/M |
|---|---|---|---|---|---|---|---|---|
| ≤25 | 78.84% | +57,814 | +2,782 | **48,120** | 78.72% | +59,236 | +2,688 | **45,378** |
| ≤50 | 81.06% | +158,683 | +3,528 | 22,233 | 80.78% | +159,823 | +3,426 | 21,436 |
| ≤100 | 83.69% | +443,852 | +4,407 | 9,929 | 83.26% | +444,023 | +4,314 | 9,716 |
| ≤200 | 86.45% | +1,208,004 | +5,331 | 4,413 | 86.03% | +1,189,071 | +5,309 | 4,465 |
| ≤500 | 90.68% | +4,380,193 | +6,745 | 1,540 | 90.17% | +4,357,624 | +6,791 | 1,558 |
| ≤1000 | 93.51% | +10,928,553 | +7,694 | 704 | 92.92% | +10,793,066 | +7,774 | 720 |

This is by far the strongest V5 signal, confirming V4's investigation:
even the most conservative threshold (≤25) lifts recall by **+8.3pp (S2) /
+8.9pp (S3)** over V2 at excellent efficiency. Efficiency degrades smoothly
and predictably as the threshold loosens — classic diminishing returns, no
cliff — but candidate volume at ≤500/≤1000 becomes large (avg 256-584
candidates/entity, max up to 4,009) for a shrinking marginal return.

### Experiment B — rare address 4-gram blocking

| Threshold | S2 recall/M | S3 recall/M |
|---|---|---|
| ≤25 | 25,762 | 25,805 |
| ≤50 | 11,260 | 11,441 |
| ≤100 | 4,696 | 4,699 |
| ≤200 | 1,839 | 1,866 |
| ≤500 | 538 | 539 |
| ≤1000 | 205 | 211 |

**Consistently and substantially less efficient than Experiment A at every
threshold** (roughly half the recall/M at the loose end, ~3x worse at the
tight end), while also reaching higher absolute candidate volumes for the
same threshold value (e.g. ≤1000: 41M pairs vs. A's 11.7M). Character
n-grams of an address carry less identifying signal per shared unit than
whole address words — **rare address tokens dominate rare address 4-grams
as a family**, a clean, direct answer to the question these two
experiments were designed to compare.

### Experiment C — name-prefix blocking

Profiling (pure aggregation, no join) found max target-side prefix
frequency of only ~46K at every length (3-6) — the pathology is combinatorial
(explained above), not any single catastrophic key. Guarded results:

| Prefix len | Threshold=100 recall/M | Threshold=100 +matches (S2/S3) |
|---|---|---|
| 3 | 2,670 / 2,893 | +146 / +161 |
| 4 | 4,135 / 4,102 | +285 / +285 |
| 5 | 5,950 / 5,752 | +447 / +451 |
| 6 | **6,846 / 7,227** | +609 / +643 |

**Efficiency improves monotonically with prefix length** (len 3 is the
worst performer, confirming the task's caution not to assume it's safe) but
even the best case (len=6) remains far weaker than Experiment A's weakest
tested threshold (≤1000: 704/720 recall/M) let alone its best (≤25:
48,120/45,378). Name-prefix blocking is a real but minor signal.

### Experiment D — combined incremental test

Most efficient candidate from each family: address token ≤25, address
4-gram ≤25, name-prefix len=6 ≤100.

| Config | S2 recall | S3 recall | Combined (S2+S3) recall | Combined +pairs | Combined +matches | Combined recall/M |
|---|---|---|---|---|---|---|
| V2 only | 70.52% | 71.21% | 70.88% | — | — | — |
| V2 + addr_token | 78.84% | 78.72% | — | — | — | — |
| V2 + addr_4gram | 75.95% | 76.24% | — | — | — | — |
| V2 + prefix | 72.34% | 73.01% | — | — | — | — |
| **V2 + all three** | **81.64%** | **81.49%** | **81.56%** | +409,007 | +7,401 | **18,095** |

**V2 + all three lifts overall recall from 70.88% to 81.56% (+10.68pp)**
for a candidate-volume increase from ~1.53M to ~1.94M pairs (~27% more) —
a genuinely controlled, efficient addition, not recall-at-any-cost.

**Overlap** (S2 / S3, among newly-recovered matches):

| | S2 | S3 |
|---|---|---|
| Only address token | 1,401 | 1,326 |
| Only address 4-gram | 493 | 485 |
| Only name-prefix | 412 | 484 |
| Found by multiple rules | 1,381 | 1,362 |

Address token is the dominant exclusive contributor (as expected from A/B/C
individually), but address 4-gram and name-prefix each still contribute a
meaningful exclusive share (~400-500 matches each) not covered by the
others — none of the three is fully redundant given the other two, though
the ~1,350-1,400 "found by multiple" pairs confirm real (not total)
overlap, mostly between the two address-based rules as expected (they draw
on the same underlying field).

### Synthesis and recommendation

Rank order by efficiency, confirmed both independently and in combination:
**rare address token ≫ rare address 4-gram > name-prefix**. If only one new
rule were adopted, it would be rare address token blocking at a conservative
threshold (≤25 to ≤50) — it alone recovers most of Experiment D's gain. All
three together are not redundant and remain efficient in combination, so
adopting the full V2+A+B+C set is defensible if the ~27% candidate-volume
increase is acceptable for the matching stage. None of this has been
adopted into `blocking.py` yet, per the task's instructions — this section
reports evidence only.

## Training-pair construction (adopted candidate generation + labeling + features)

**This is the first stage that adopts a blocking configuration into
production code** (`src/candidate_generation.py`, `src/matching_dataset.py`).
No model is trained here — this stage stops at a validated, feature-engineered,
entity-split training dataset.

### Adopted candidate-generation configuration (10 rules)

V1 base (reused directly from the already-computed full-scale
`derived/blocking/candidates_by_rule_train_source{2,3}.parquet` — **not
recomputed**): `exact_name`, `core_name`, `sorted_tokens`, `strong_token`,
`postal_code`. V2 additions: `ngram` (character 4-gram of `name_core`,
df≤50), `guarded_city_street` (V2's guarded `city_candidate`+`street_number`
rule). V5 additions: `rare_address_token` (df≤25), `rare_address_ngram`
(character 4-gram of `address_normalized`, df≤25), `name_prefix` (first 6
chars of `name_core`, df≤100). All 5 new rules are built at **full S1 scale**
(2,206,821 entities, not the 20K experimental sample) by reusing the exact
rule-building functions from `blocking_v2_experiments.py` /
`blocking_v5_experiments.py` unmodified — only the population they run
against changes.

### Step 2: sanity test (before full-scale generation)

Ran on a 2,000-entity S1 sample first: all 10 rules present, all
`source1_entity_id`/`candidate_id` prefixes valid, zero duplicate pairs,
every `candidate_id` verified to exist in the target table, and rebuild
determinism confirmed (identical output on a second run). Full-scale
generation was only started after these passed.

### Full-scale candidate counts (real numbers, not sample-projected)

| | S1→S2 | S1→S3 |
|---|---|---|
| By-rule rows (10 rules, not deduplicated) | 147,485,490 | 154,227,912 |
| **Deduplicated candidate pairs** | **104,391,442** | **109,429,964** |
| Build time (5 new rules only) | 73.5s | 63.7s |
| Write time | 98.9s | 79.7s |

Post-build sanity checks on the real output (not a sample) all passed: valid
ID prefixes, zero duplicate pairs, every candidate/source ID exists in its
respective normalized table, every union pair traces to ≥1 rule, and S2/S3
candidate IDs never cross-contaminate.

### Step 3: ground-truth labeling

| | S1→S2 | S1→S3 |
|---|---|---|
| Total ground-truth matches | 3,693,619 | 3,944,746 |
| Captured by blocking (= positives) | 3,008,241 | 3,213,006 |
| Missed by blocking | 685,378 | 731,740 |
| **Blocking recall** | **81.44%** | **81.45%** |
| Negative candidates | 101,383,201 | 106,216,958 |
| Positive rate | 2.88% | 2.94% |

Real full-scale recall (81.44%/81.45%) lands almost exactly on the V5
sample-based projection (81.56% combined) — a strong consistency check
between the 20K-sample experiments and full production behavior. The
685,378 + 731,740 = 1,417,118 missed positives are **not available to the
matcher and are never treated as negatives** — they simply don't appear in
the candidate/feature tables at all, per the task's explicit instruction.

### Step 4: entity-level train/validation split

80/20 split by S1 `entity_id` via a deterministic hash (`abs(hash(entity_id))
% 100 < 80`), computed **once** for the full S1 population and reused
identically for both targets (confirmed: both runs produced identical
train=1,765,255 / validation=441,566 entity counts). A dedicated check
(`no S1 entity spans both splits`) passed with zero violations on both
feature tables — no candidate row's S1 entity ever appears in both splits.

### Step 5: feature engineering (31 features)

Negatives were downsampled 10:1 per positive **before** per-pair feature
computation (see "resource constraints" below for why), giving 32,225,559
(S2) and 34,400,328 (S3) feature rows.

**Name (10)**: `exact_normalized_name`, `exact_core_name`, `exact_sorted_tokens`,
`name_token_jaccard`, `name_token_overlap_count`, `name_char_ngram_jaccard`,
`name_edit_distance`, `name_edit_similarity`, `name_length_diff`, `name_token_count_diff`.
**Address (8)**: `exact_normalized_address`, `address_token_jaccard`,
`address_token_overlap_count`, `postal_code_equal`, `street_number_equal`,
`city_equal`, `state_equal`, `address_length_diff`.
**Metadata (3)**: `country_equal`, `target_name_has_non_latin`,
`target_address_has_non_latin`.
**Blocking provenance (10)**: `blocked_exact_name`, `blocked_core_name`,
`blocked_sorted_tokens`, `blocked_strong_token`, `blocked_ngram`,
`blocked_postal`, `blocked_city_street`, `blocked_rare_address_token`,
`blocked_address_ngram`, `blocked_name_prefix`.

Feature build time (on the downsampled data): 699.5s (S2) / 918.3s (S3).

### Step 6: feature validation — two real issues found

1. **`country_equal` is a constant feature** (100% of rows = 1.0/True, on
   both targets). This is structural, not a bug: every one of the 10
   blocking rules joins `ON s1.country = t.country`, so no candidate pair
   can ever have mismatched countries. The feature carries zero predictive
   value and should be dropped before training.
2. **Three feature pairs are perfectly redundant**: `exact_normalized_name`
   ≡ `blocked_exact_name`, `exact_core_name` ≡ `blocked_core_name`,
   `exact_sorted_tokens` ≡ `blocked_sorted_tokens` — confirmed by identical
   positive/negative class means for each pair (e.g. both `exact_core_name`
   and `blocked_core_name` show pos-mean 0.5187/0.4949, neg-mean
   0.3551/0.3696 on S2/S3). This happens because those three blocking
   rules' join conditions are the exact same boolean test the corresponding
   feature recomputes independently. The other 7 `blocked_*` provenance
   flags are NOT redundant (their means differ from any single `exact_*`
   feature, since they encode looser/threshold-based or multi-field
   criteria). Recommendation for the modeling stage: drop `country_equal`
   and one column from each redundant pair (6 columns → 3), leaving 27
   effective features.

No other issues found: 0% missing on every feature except
`exact_normalized_address` (2.79-3.16% null, exactly matching known address
missingness — expected, not a bug), no feature references
`source1_entity_id`/`candidate_id`/`label` (verified both structurally and
by unit test), and every feature shows sensible positive/negative class
separation in the expected direction (e.g. `name_edit_distance`: pos-mean
7.95/6.69 vs. neg-mean 11.80/11.18 — positives are more similar, as expected).

### Step 7: resource constraints and the negative-downsampling decision

The raw candidate volume (104-109M pairs per target, ~208M combined) is too
large to compute per-pair string-similarity features (n-gram Jaccard,
Levenshtein) on every row without a very long runtime for rows that would
mostly never be used in training anyway (imbalanced 2.9% positive rate).
**Decision, applied before feature computation**: keep 100% of positives
and a reservoir sample of negatives at a 10:1 ratio, executed via DuckDB
directly on Parquet (no pandas materialization of the full candidate set at
any point). This is both a memory-safety measure and standard practice for
this class of imbalanced pair-classification problem.

### Bottlenecks observed

- Free system RAM was ~0.4GB at points during this stage; S2 and S3 were
  run strictly sequentially (never concurrently) as required.
- The dominant cost was **not** feature computation but the **blocking
  provenance join** (recomputing which of the 10 rules produced each
  sampled candidate pair, joined against the 147-154M-row by-rule table) —
  S2's full pipeline (label → split → sample → provenance-join → features)
  took 3,661.8s (~61 min), of which only 699.5s (~11%) was the feature
  computation step itself. The same pattern held for S3 (3,880.4s total,
  918.3s / ~24% feature step). A future iteration should materialize the
  provenance join before sampling (or index/pre-filter it) to reduce this.
- Disk usage for this stage: ~9.6GB (`derived/matching/`), on a machine
  with 40GB free at completion — comfortable, but growing; the by-rule
  files (10 rules, undeduplicated) are the largest artifacts and are not
  needed after the provenance features are extracted.

**Stopped exactly where instructed**: adopted candidate generation, ground-
truth labeling, entity-level split, and feature construction/validation are
complete; no model has been trained.

## Model training stage: environment check + feature cleanup + smoke test

**No model has been trained on the real data yet.** This section covers
cleaning up the feature schema found in the previous stage's validation,
checking the actual GPU/LightGBM environment (not assumed), and a tiny
smoke test proving the pipeline works end to end. The first real LightGBM
baseline is the next step.

### Step 1: feature schema cleanup

`src/matching_dataset.py` now defines `MODEL_FEATURE_COLUMNS` (27 features)
alongside the original `FEATURE_COLUMNS` (31, kept for reference/documentation)
and an explicit `REMOVED_FEATURES` dict recording why each of the 4 dropped
columns was removed:

| Removed | Reason |
|---|---|
| `country_equal` | Constant (100% True on both targets) — every one of the 10 blocking rules joins `ON s1.country = t.country` by construction |
| `blocked_exact_name` | Exact duplicate of `exact_normalized_name` (same underlying boolean test) |
| `blocked_core_name` | Exact duplicate of `exact_core_name` |
| `blocked_sorted_tokens` | Exact duplicate of `exact_sorted_tokens` |

**No data was regenerated** — the existing `features_train_source{2,3}.parquet`
files still contain all 31 raw columns; `MODEL_FEATURE_COLUMNS` is simply the
column subset selected at training time. New unit tests
(`tests/test_matching_dataset.py::TestModelFeatureCleanup`, 7 tests) verify:
the constant and all 3 duplicate pairs are excluded, the count is exactly 27,
no duplicate names remain, removed features are a subset of the raw list,
and no ID/label/split column can leak into the model feature list. Full
suite: **49/49 tests pass.**

### Step 2: GPU environment (checked, not assumed)

```
nvidia-smi: NVIDIA GeForce RTX 3050 (laptop), 4096 MiB VRAM, driver 592.82, driver CUDA capability 13.1
torch: 2.10.0+cpu — cuda available: False — cuda version: None
```

The installed PyTorch build is **CPU-only**. The driver itself supports up
to CUDA 13.1, but that is irrelevant while the installed PyTorch wheel has
no CUDA support compiled in. Per instructions, this was **not** "fixed" by
reinstalling — it's reported as-is, since we are not training with PyTorch
here anyway (LightGBM is the planned first baseline).

### Step 3: LightGBM GPU support (checked empirically, not assumed)

LightGBM was **not installed at all** prior to this stage. Installed the
standard PyPI wheel (`pip install lightgbm`, a routine, low-risk operation —
not the "risky system-wide installation" the task warned against) →
**lightgbm 4.7.0**.

Verified GPU support empirically (tiny synthetic 200-row dataset, not the
real data) rather than assuming from the PyTorch result:

```
device_type='gpu'  -> LightGBMError: GPU Tree Learner was not enabled in this build. Recompile with -DUSE_GPU=1
device_type='cuda' -> LightGBMError: CUDA Tree Learner was not enabled in this build. Recompile with -DUSE_CUDA=1
device_type='cpu'  -> SUCCESS
```

**The standard PyPI wheel has no GPU/CUDA support compiled in, confirmed by
direct test, independent of the PyTorch finding.** Getting GPU-enabled
LightGBM on Windows requires a from-source CMake build with either an
OpenCL SDK (`-DUSE_GPU=1`) or the CUDA Toolkit + a matching MSVC/Visual
Studio Build Tools install (`-DUSE_CUDA=1`) — a multi-step, failure-prone
process on Windows, and exactly the kind of "risky system-wide installation"
the task says to avoid for now.

**Safest recommended path (not executed)**: stay on CPU LightGBM. This is a
deliberate recommendation, not a fallback of convenience — with only 27
features, LightGBM's CPU histogram algorithm is already highly optimized
and the 4GB VRAM budget would constrain batch/tree size on GPU anyway; the
realistic speedup from a GPU build here is small relative to the setup risk.
If GPU LightGBM is wanted later, the safest route is `conda install -c
conda-forge lightgbm` first (sometimes ships a GPU-enabled build without a
local compile) and only fall back to a from-source CMake build if that
doesn't work — done in an isolated environment, never system-wide.

### Step 4: memory / dataset plan

| | S1→S2 | S1→S3 |
|---|---|---|
| Format | Parquet | Parquet |
| File size | 916.9 MB | 975.7 MB |
| Rows (post 10:1 downsampling) | 32,225,559 | 34,400,328 |
| Positive : negative ratio | 1 : 9.71 | 1 : 9.71 |

Schema: 2 VARCHAR ID columns (`source1_entity_id`, `candidate_id`, not fed
to the model), 1 VARCHAR `split`, 1 INTEGER `label`, and the 27 model
features as a mix of BOOLEAN (17, incl. one nullable due to ~3% missing
`business_address`), BIGINT (6), and DOUBLE (4).

**Estimated RAM if the combined S2+S3 feature matrix were loaded at once**:
27 float32 features + 1 int32 label × (32,225,559 + 34,400,328) = 66,625,887
rows × 112 bytes/row ≈ **7.5 GB** — before accounting for the ID/split
string columns, which pandas stores as Python object pointers (each short
string object costs far more than its character count) and were **excluded
from this estimate entirely because they must never be loaded at this
scale**. 7.5GB is not safe on this machine (free RAM has been observed as
low as 0.4GB during earlier stages) — **the full combined dataset must
never be loaded into a single pandas DataFrame**, confirmed both by this
estimate and by the task's explicit instruction.

**Recommended safe training configuration for the next stage** (not
implemented yet):
1. Train on **one target at a time** first (S2 alone ≈32.2M rows ≈3.6GB as
   float32 — still large but far safer than the 7.5GB combined figure), or
   apply additional downsampling if a combined S2+S3 model is wanted later.
2. Select only `MODEL_FEATURE_COLUMNS` (never the ID/split columns) before
   any pandas materialization.
3. Read via DuckDB in bounded batches (the same `fetch_arrow_reader`
   pattern already used in `io_utils.iter_record_batches` for
   normalization) and cast directly to float32 numpy arrays, rather than
   one large `fetchdf()` call, to keep peak memory bounded regardless of
   row count.
4. No GPU is available, so there is no VRAM budget to design around for
   LightGBM specifically — training stays entirely on CPU/RAM.

### Step 5: small-subset smoke test (250K train / 50K validation rows)

Sampled independently **within** each `split` value (never mixing train
and validation entities — verified with a direct entity-overlap query
returning **0** on the full 32.2M-row file, not just the sample).

| Check | Result |
|---|---|
| Feature loading | 199,707 train / 10,037 valid rows loaded in 6.1s |
| Pandas memory (small subset) | 31.2 MB (train) + 1.6 MB (valid) |
| Label loading | train pos=18,727/neg=180,980; valid pos=926/neg=9,111 |
| dtype handling | Mixed bool/int64/float64/nullable-`boolean`/object columns all cast cleanly to float32 |
| NaN/Inf check (post-cast) | 6,212 legitimate NaN (from ~3% missing `business_address` → `exact_normalized_address` NULL), **0 Inf** |
| Entity-safe split | Confirmed 0 entity overlap between train/validation on the full file |
| GPU path | Failed cleanly with the expected LightGBM error (re-confirms Step 3, no silent fallback) |
| **CPU training** | **20 rounds in 0.30s**, validation AUC 0.999, logloss 0.042 |
| Prediction sanity | range [0.013, 0.892], mean 0.095, pred-label correlation 0.961 |

**Result: SMOKE TEST SUCCESS.** The pipeline — Parquet → DuckDB query →
pandas → float32 cast → LightGBM `Dataset` → CPU training → prediction —
works end to end with the cleaned 27-feature schema, on real (subsampled)
data, with no dtype, NaN, or entity-leakage problems.

## Known edge cases / limitations (discovered while testing)

- **"St" ambiguity**: `St` is expanded to `Street` everywhere except as the
  first token of an address, to avoid corrupting `"St. Louis"` into
  `"Street Louis"`. This heuristic is not perfect for mid-string "Saint ..."
  references.
- **`postal_code` / `street_number` are heuristics**: a small fraction of
  addresses could still have a numeric segment mis-attributed (e.g. a house
  number that happens to look like a unit code). There is no ground truth
  for these components to validate against.
- **`city_candidate` / `state_candidate` are positional guesses**: for
  addresses with unusual comma placement (very common in the Indian records,
  which sometimes list city/state before the street), these can be wrong.
  They are intentionally named "candidate" and should be treated as a soft
  feature later, not ground truth.
- **Non-Latin scripts are preserved, not transliterated**: Devanagari/Tamil/
  Kannada/Malayalam text passes through `name_normalized`/`address_normalized`
  unchanged (after NFKC normalization + whitespace cleanup). Comparing a
  Latin name against its native-script counterpart is deferred to a later
  stage (an explicit design decision — no external transliteration is used
  anywhere in this project).
- **Unicode combining marks**: an early version of the character filter used
  Python's `\w`, which does *not* count Unicode combining marks (category
  Mn/Mc — e.g. Devanagari vowel signs) as "word" characters, silently
  deleting them and corrupting non-Latin names into disconnected consonants.
  Fixed by filtering on Unicode general category (Letter/Mark/Number) instead.
- **Component-level missing markers**: some addresses embed a placeholder
  like `<NULL>` as one comma-separated component inside an otherwise valid
  address (e.g. `"Shop No. 49, <NULL>, Hyderabad, Andhra Pradesh"`). This is
  handled by dropping any individual component that matches a missing-value
  token, in addition to treating a fully-missing field as `None`.
