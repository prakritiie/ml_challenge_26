# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

Business identities from three noisy, independently-sourced record sets are
matched by generating a small, high-recall candidate set per Source-1
entity via IDF-weighted inverted-index retrieval over normalized name and
address tokens, scoring every candidate with a LightGBM classifier over 27
similarity features, and accepting matches via probability-gap decoding.
This replaced an earlier rule-based blocking approach (F0.5 ≈ 0.540) with
one that raised blocking recall from ~81.5% to ~92% while keeping candidate
volume small, lifting the final leaderboard score to **F0.5 = 0.809**.

---

## 2. Methodology

### 2.1 Problem Analysis

- Name noise: legal-suffix abbreviations (Corp/Corporation, Pvt/Private,
  Ltd/Limited), DBA/trade-name variants, punctuation differences (`&` vs.
  "and"), word-order transpositions, typos, and transliteration variants for
  non-Latin scripts.
- Address noise: abbreviated street types (Rd/Road, St/Street), missing
  components (no postal code, no state), landmark-based references ("Near
  SBI ATM"), inconsistent municipal numbering, and component reordering.
- Scale: millions of records per source rules out any all-pairs comparison;
  candidate generation must be sub-quadratic and, in practice, fast enough
  to run on a single 8-core/16GB machine within a bounded time budget.
- An initial rule-based blocking approach (exact/core/sorted-token name
  matches, strong-token joins, postal-code joins) reached only ~81.5%
  candidate recall and was the ceiling on the 0.540 submission — its
  exact-match keys could not recover matches separated by moderate name or
  address variation.

### 2.2 Solution Strategy

**Approach Type:** Retrieval (inverted index) + Classifier (LightGBM),
i.e. a blocking-and-matching hybrid where the blocking stage is
similarity-ranked rather than exact-match/rule-based.

**Core Innovation:** Two independent inverted indexes (name-token view,
address-token view) built over each target source, with IDF-weighted
scoring and a document-frequency ceiling that keeps a single very common
token from exploding per-query cost. Each Source-1 entity's top-K matches
from each view are unioned into its candidate set. This was chosen after an
initial brute-force TF-IDF-cosine-similarity prototype (scikit-learn
`TfidfVectorizer` + sparse matrix multiplication) proved too slow to run at
full test-set scale (a few hundred queries/sec, single-threaded); the
inverted-index reformulation reached ~700 queries/sec/target with
comparable recall, by scoring each query only against the postings its own
tokens actually hit instead of a dense similarity computation.

---

## 3. Candidate Generation (Blocking)

- **Retrieval keys used:** whitespace tokens of the normalized business
  name (`name_core`) and normalized address tokens, each indexed separately
  with IDF weighting; top `K=20` candidates per view, unioned.
- **Candidate pairs generated:** ~66.7M (test_source2) + ~66.8M
  (test_source3) ≈ 133.5M total, averaging ~38.5 candidates per Source-1
  entity per target (~77 combined).
- **How true matches were not lost:** validated on an 8,000-entity training
  sample against `train_ground_truth.tsv` with the FULL target corpus (not
  a subsample) before committing to the full test run — combined recall
  93.65% (S2) / 90.88% (S3), vs. ~81.5% for the earlier rule-based
  blocking. A document-frequency ceiling (`MAX_DF=20,000`) was tuned by
  trading recall against retrieval speed on this same validation sample:
  too low a ceiling (2,000) gave a fast but low-recall index (~70%
  combined); too high (50,000) recovered recall (~93%) but made full-scale
  retrieval too slow to finish in the available time; 20,000 was the
  chosen middle ground (~91% combined recall at a feasible full-scale
  runtime).

---

## 4. Matching Model

**Features used (27 total):**
- Name features: exact-match flags on normalized/core/sorted-token forms,
  token Jaccard similarity and overlap count, character 4-gram Jaccard,
  byte-level Levenshtein distance and derived similarity, name length and
  token-count differences.
- Address features: exact normalized-address match, address-token Jaccard
  and overlap count, postal-code/street-number/city/state equality flags,
  address length difference.
- Other: target-side non-Latin-script flags (name and address), and 7
  blocking-provenance flags inherited from the earlier rule-based stage
  (always `False` for candidates found only via the new retrieval, which is
  the correct value — they were never evaluated by those rules).

**Model type:** LightGBM binary classifier (gradient-boosted trees), one
model per target source (`lightgbm_source2_baseline.txt`,
`lightgbm_source3_baseline.txt`), trained on the rule-based blocking
stage's candidate/label set and reused as-is for the final retrieval
candidates (not retrained).

**Threshold selection method:** probability-gap decoding per Source-1
entity — accept the highest-probability candidate plus any other candidate
within `gap` of it, gated by a minimum `threshold` on the top probability:
`threshold = 0.75`, `gap = 0.10`.

---

## 5. Results & Error Analysis

- **Leaderboard F0.5:** 0.809 (final TF-IDF retrieval + LightGBM pipeline),
  vs. 0.540 for the earlier rule-based blocking baseline.
- **Common false negatives (missed matches):** entities whose true match
  shares neither a common name token nor a common address token with the
  Source-1 record (e.g. a full DBA/trade-name change combined with a
  relocated address) fall outside both inverted-index views entirely and
  are never retrieved as candidates, so no amount of downstream scoring can
  recover them.
- **Common false positives (wrong merges):** candidates that score above
  the 0.75 threshold and within the 0.10 gap of the top match despite being
  a different real-world entity — most often near-duplicate franchise/chain
  locations sharing a name and a similar (but distinct) address.

---

## 6. Conclusion

Replacing exact-match rule-based blocking with IDF-weighted inverted-index
retrieval raised candidate recall from ~81.5% to ~92% while keeping
candidate volume small enough to score with the existing LightGBM models
unmodified, lifting the leaderboard score from 0.540 to 0.809. The main
lesson: a brute-force similarity computation (dense TF-IDF cosine) can have
excellent recall but be practically unusable at scale, while an
IDF-weighted inverted index reformulation of the same idea can reach
comparable recall at a fraction of the compute cost.

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/`
(all source in `src/`, with a `README.md` and `requirements.txt`). Entry
points, run in order, to reproduce `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the raw dataset:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt

python -m src.normalization run --source all      # stage 1: normalize
python -m src.tfidf_retrieval                      # stage 2: retrieve candidates
python -m src.score_candidates                      # stage 3/4: features + LightGBM + decode
python -m src.generate_submission                   # stage 5: merge into output/*.tsv
```

Key modules: `src/tfidf_retrieval.py` (inverted-index retrieval),
`src/feature_engineering.py` (27 similarity features),
`src/score_candidates.py` (bucketed LightGBM scoring),
`src/generate_submission.py` (final-file merge). The earlier rule-based
blocking stage and the full experimentation history are documented
separately in `code/business_entity_resolution/EXPERIMENTS.md`.

### B. Additional Results

See the "Results" table and retrieval-recall breakdown in the root
`README.md` for the full comparison between the rule-based baseline and the
final retrieval + LightGBM pipeline.

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
