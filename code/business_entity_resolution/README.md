# Business Entity Resolution — Final Pipeline (F0.5 = 0.809)

This README documents the pipeline that actually produced the submitted
`output/matching_results.tsv` / `output/candidate_pairs.tsv` and scored
**F0.5 = 0.809** on the leaderboard. For the earlier rule-based blocking
approach (scored ~0.540) and the experimentation that led to this final
design, see [`EXPERIMENTS.md`](EXPERIMENTS.md) — normalization is shared
between both, everything downstream of candidate generation is not.

## Pipeline overview

```
dataset/  ->  normalization  ->  TF-IDF / inverted-index retrieval
          ->  candidate union (name view + address view)
          ->  27 similarity features  ->  LightGBM  ->  probability-gap decoding
          ->  S2 + S3 merge  ->  output/matching_results.tsv, output/candidate_pairs.tsv
```

## Dataset

Six tab-separated files under `dataset/` (`train_source{1,2,3}.tsv`,
`test_source{1,2,3}.tsv`) plus `dataset/train/train_ground_truth.tsv`. See
the root [README.md](../../README.md) for the field-level format.

## 1. Normalization

Unchanged from the earlier pipeline — see `src/normalization.py` and
[`EXPERIMENTS.md`](EXPERIMENTS.md) for the full design rationale (legal-suffix
folding, address abbreviation expansion, token/n-gram derivation, landmark
extraction, non-Latin-script flags). Produces one Parquet file per source
under `derived/normalized/`.

```bash
# Normalize one full source file to Parquet (streamed, constant memory)
python -m src.normalization run --source test_source1

# Normalize every source file (train + test)
python -m src.normalization run --source all
```

## 2. Candidate retrieval: inverted-index name + address search

`src/tfidf_retrieval.py` replaces the earlier rule-based blocking (and an
initial brute-force TF-IDF-cosine prototype that was too slow — see
`EXPERIMENTS.md`) with two in-memory inverted indexes built over each
target corpus (`test_source2`, `test_source3`):

- **View B (name):** `name_core`, whitespace-tokenized
- **View C (address):** `address_tokens` (already tokenized during
  normalization)

For each token, an IDF weight is precomputed; a token whose document
frequency exceeds `MAX_DF` (20,000) is dropped from the index entirely, so a
single very common token can never blow up per-query cost regardless of
corpus size. Each S1 query is scored against only the postings its own
tokens actually hit (via a vectorized `concatenate` + `unique` +
`bincount`, never an array the size of the whole target corpus), the
top `K=20` candidates per view are kept, and the two views' top-K sets are
unioned into the final candidate set for that entity.

```bash
python -m src.tfidf_retrieval
```

Writes `derived/tfidf_retrieval/test_candidates_source2.parquet` and
`test_candidates_source3.parquet` (`source1_entity_id`, `candidate_id`
pairs), streamed out in bounded batches so peak RSS stays under ~2GB
regardless of test-set size.

### Retrieval results (validated on an 8,000-entity training sample, full
target corpus, against `train_ground_truth.tsv`)

| View | S2 recall | S3 recall |
|---|---:|---:|
| Name TF-IDF (B) alone | 52.4% | 50.6% |
| Address TF-IDF (C) alone | 87.2% | 81.3% |
| **B + C combined** | **93.65%** | **90.88%** |

Combined recall ≈ 92.2% overall, vs. ~81.5% for the earlier rule-based
blocking, at ~29 candidates/entity average (p95 = 30, max = 30 in the
validation sample; K=20 per view raises this ceiling slightly at full
scale).

## 3. Feature computation: 27 similarity features

`src/feature_engineering.py` computes the same 27 features the LightGBM
models were trained on (name/address exact-match flags, token Jaccard and
overlap counts, character 4-gram Jaccard, byte-level Levenshtein distance
and similarity, length/token-count diffs, postal/street/city/state equality,
non-Latin-script flags, and the 7 `blocked_*` blocking-provenance flags —
always `False` here, since every candidate comes from retrieval, not the
old blocking rules). Verified byte-for-byte identical (27/27 features, 0
mismatches, 2,012 validation pairs, 20/20 non-ASCII spot checks) against the
original DuckDB-SQL feature definitions used at training time.

The one performance change from a naive per-row Python implementation:
`name_edit_distance` uses `rapidfuzz.distance.Levenshtein.distance()` on the
UTF-8-encoded bytes of `name_core` (matching DuckDB's byte-level, not
codepoint-level, `levenshtein()`), a C implementation ~20-100x faster than
a pure-Python edit-distance loop — this was the dominant per-row cost
before the change.

## 4. LightGBM scoring

`src/score_candidates.py` scores every retrieved candidate with the
existing, **not retrained**, models:

- `models/lightgbm_source2_baseline.txt`
- `models/lightgbm_source3_baseline.txt`

```bash
python -m src.score_candidates
```

Processes each target's candidates in 60 hash-buckets
(`abs(hash(source1_entity_id)) % 60`), filtering the raw candidates parquet
scan *before* joining against the normalized S1/target tables — filtering
only after the join makes DuckDB hash-join the full unfiltered candidate
table on every bucket, which spills tens of GB to disk and crashes (see
`EXPERIMENTS.md`). Feature computation and LightGBM prediction for each
bucket's entities run in a `ProcessPoolExecutor` pool (8 workers); each
worker receives only a small sliced `pyarrow.Table` and does its own
`to_pylist()` conversion, so that cost is parallelized instead of paid
serially in the main process.

Writes, per target, under `derived/scoring_tmp/`:
- `scored_candidates_<target>.tsv` — every (S1, candidate) pair that was
  scored, with no filtering
- `decoded_matches_<target>.tsv` — the pairs accepted by probability-gap
  decoding (S1, candidate, probability)

## 5. Probability-gap decoding

Applied per S1 entity, per target, inside `score_candidates.py`
(`decode_probability_gap`): accept the highest-probability candidate, plus
any other candidate within `gap` of it, gated by a minimum `threshold` on
the top probability itself.

```
threshold = 0.75
gap       = 0.10
```

## 6. Final submission generation

`src/generate_submission.py` merges the two targets' decoded matches and
scored candidates into the two final, one-row-per-S1-entity submission
files.

```bash
python -m src.generate_submission
```

- `output/matching_results.tsv` — built with plain Python file I/O rather
  than DuckDB's CSV writer, which was found to quote an empty
  `matched_entity_ids` field as the literal string `""` instead of leaving
  it truly empty (both fails strict TSV validation and manufactures a fake
  "matched ID").
- `output/candidate_pairs.tsv` — built via the same 60-hash-bucket
  streaming pattern as retrieval/scoring, reading directly from the
  retrieval parquet files. A single global `GROUP BY ... string_agg(...)`
  over the full ~133M-row candidate set reliably hit DuckDB's memory
  ceiling (OOM'd at both a 4GB and a 6GB limit); bucketing was the only
  version that completed without spilling or crashing.

## 7. Validate

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Model training (already done — models are checked in)

The two LightGBM models in `models/` were trained once, ahead of this final
retrieval pipeline, on the earlier rule-based blocking's candidate/feature
set (see `EXPERIMENTS.md` for the full training-pair construction and
feature-selection process). They are used as-is here — retraining them on
the new, higher-recall retrieval candidates was not attempted for the
0.809 submission. To retrain from scratch:

```bash
python -m src.train_baseline --target train_source2
python -m src.train_baseline --target train_source3
```

## Repository layout

```
code/business_entity_resolution/
├── README.md              <- this file (final 0.809 pipeline)
├── EXPERIMENTS.md          <- earlier rule-based blocking + experimentation log
├── requirements.txt
├── models/
│   ├── lightgbm_source2_baseline.txt
│   └── lightgbm_source3_baseline.txt
├── src/
│   ├── config.py                 (paths, shared constants)
│   ├── io_utils.py                (DuckDB connection/parquet helpers)
│   ├── normalization.py           (stage 1)
│   ├── tfidf_retrieval.py         (stage 2 -- final pipeline)
│   ├── feature_engineering.py     (stage 3 -- final pipeline)
│   ├── score_candidates.py        (stage 4/5 -- final pipeline)
│   ├── generate_submission.py     (stage 6 -- final pipeline)
│   ├── train_baseline.py          (LightGBM training, already run)
│   ├── matching_dataset.py        (training-pair/feature construction)
│   └── transliteration.py, blocking*.py, candidate_generation.py
│                                    (earlier rule-based blocking stage --
│                                     superseded, kept for EXPERIMENTS.md)
└── tests/
```
