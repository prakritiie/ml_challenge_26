"""Score the TF-IDF/inverted-index retrieval candidates (src/tfidf_retrieval.py
output) with the existing, NOT retrained, LightGBM models
(models/lightgbm_source{2,3}_baseline.txt) and the 27-feature
implementation in feature_engineering.py, then apply probability-gap
decoding.

Architecture notes (both load-bearing -- earlier, less careful versions of
this stage failed or were 5-10x slower):

1. The candidate-bucket WHERE filter (`abs(hash(source1_entity_id)) %
   NUM_BUCKETS = bucket_id`) MUST be applied to the raw candidates parquet
   scan alone, inside its own CTE, BEFORE joining against the S1/target
   normalized tables. Filtering only after the join makes DuckDB hash-join
   the FULL unfiltered candidates table on every one of the NUM_BUCKETS
   iterations, which spilled 30+ GB to disk and crashed on this machine.
   Filtering first lets DuckDB push the predicate into the parquet scan, so
   each bucket only ever materializes its own small (~1.1M-row) slice.

2. Each bucket's Arrow result table is hashed into per-entity batches using
   ONLY its single, cheap `source1_entity_id` string column in the main
   process (fast), then `table.take(row_indices)` slices out each batch as
   a small Arrow table passed to a worker. The expensive full 24-column
   `to_pylist()` conversion happens INSIDE the worker (in parallel across
   N_WORKERS processes) rather than serially in the main process --
   benchmarked at 37s/bucket when done serially (the actual bottleneck),
   vs. negligible once parallelized.

No provenance/blocking-rule join is needed: every candidate here comes from
the TF-IDF retrieval stage, not the old blocking rules, so the 7 blocked_*
features are correctly all False (feature_engineering.compute_features
already defaults a missing/empty provenance dict to False for each).
"""
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import psutil
import lightgbm as lgb

from . import config, io_utils
from .feature_engineering import compute_features, MODEL_FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

DECODE_THRESHOLD = 0.75
DECODE_GAP = 0.10
BATCH_SIZE_ENTITIES = 500
NUM_BUCKETS = 60
N_WORKERS = 8
PROGRESS_EVERY_BATCHES = 20

TARGET_MODELS = {
    "test_source2": MODELS_DIR / "lightgbm_source2_baseline.txt",
    "test_source3": MODELS_DIR / "lightgbm_source3_baseline.txt",
}

_worker_model = None


def _mem_gb():
    return psutil.Process().memory_info().rss / 1e9


def _disk_free_gb():
    return shutil.disk_usage(str(config.DERIVED_ROOT)).free / 1e9


def _worker_init(model_path):
    global _worker_model
    _worker_model = lgb.Booster(model_file=model_path)


def _worker_process_batch(batch_table):
    """batch_table: a pyarrow Table slice (not pre-converted rows) -- the
    to_pylist() conversion happens here, inside the worker (see module
    docstring point 2)."""
    global _worker_model
    n = batch_table.num_rows
    cols = batch_table.column_names
    col_data = [batch_table.column(c).to_pylist() for c in cols]
    rows = [tuple(col_data[j][i] for j in range(len(cols))) for i in range(n)]
    n_feat = len(MODEL_FEATURE_COLUMNS)
    X = np.empty((n, n_feat), dtype=np.float32)
    s1_ids = [None] * n
    cand_ids = [None] * n
    for i, row in enumerate(rows):
        (s1_id, cand_id,
         s1_nn, s1_core, s1_ntok, s1_sorted, s1_addr, s1_atok, s1_postal, s1_sn, s1_city, s1_state,
         t_nn, t_core, t_ntok, t_sorted, t_addr, t_atok, t_postal, t_sn, t_city, t_state,
         t_nl, t_anl) = row
        s1_d = {
            "name_normalized": s1_nn, "name_core": s1_core,
            "name_tokens": list(s1_ntok) if s1_ntok is not None else None,
            "name_sorted_tokens": list(s1_sorted) if s1_sorted is not None else None,
            "address_normalized": s1_addr,
            "address_tokens": list(s1_atok) if s1_atok is not None else None,
            "postal_code": s1_postal, "street_number": s1_sn,
            "city_candidate": s1_city, "state_candidate": s1_state,
        }
        t_d = {
            "name_normalized": t_nn, "name_core": t_core,
            "name_tokens": list(t_ntok) if t_ntok is not None else None,
            "name_sorted_tokens": list(t_sorted) if t_sorted is not None else None,
            "address_normalized": t_addr,
            "address_tokens": list(t_atok) if t_atok is not None else None,
            "postal_code": t_postal, "street_number": t_sn,
            "city_candidate": t_city, "state_candidate": t_state,
            "name_has_non_latin": t_nl, "address_has_non_latin": t_anl,
        }
        feats = compute_features(s1_d, t_d, {})  # no provenance -> blocked_* all False
        X[i, :] = [np.nan if v is None else float(v) for v in feats]
        s1_ids[i] = s1_id
        cand_ids[i] = cand_id
    probs = _worker_model.predict(X)
    return s1_ids, cand_ids, probs.tolist()


def decode_probability_gap(pairs, threshold=DECODE_THRESHOLD, gap=DECODE_GAP):
    """Accept the top-probability candidate plus any other within `gap` of
    it, gated by a minimum `threshold` on the top probability itself."""
    if not pairs:
        return []
    pairs_sorted = sorted(pairs, key=lambda x: -x[1])
    top_prob = pairs_sorted[0][1]
    if top_prob < threshold:
        return []
    return [p for p in pairs_sorted if p[1] >= top_prob - gap]


def build_bucket_query(target_key, bucket_id):
    s1_path = config.NORMALIZED_FILES["test_source1"]
    t_path = config.NORMALIZED_FILES[target_key]
    candidates_path = config.TFIDF_RETRIEVAL_DIR / f"test_candidates_{target_key.split('_')[1]}.parquet"
    return f"""
        WITH bucketed_candidates AS (
            SELECT source1_entity_id, candidate_id
            FROM read_parquet('{candidates_path}')
            WHERE abs(hash(source1_entity_id)) % {NUM_BUCKETS} = {bucket_id}
        )
        SELECT
            bc.source1_entity_id, bc.candidate_id,
            s1.name_normalized AS s1_name_normalized, s1.name_core AS s1_name_core,
            s1.name_tokens AS s1_name_tokens, s1.name_sorted_tokens AS s1_name_sorted_tokens,
            s1.address_normalized AS s1_address_normalized, s1.address_tokens AS s1_address_tokens,
            s1.postal_code AS s1_postal_code, s1.street_number AS s1_street_number,
            s1.city_candidate AS s1_city_candidate, s1.state_candidate AS s1_state_candidate,
            t.name_normalized AS t_name_normalized, t.name_core AS t_name_core,
            t.name_tokens AS t_name_tokens, t.name_sorted_tokens AS t_name_sorted_tokens,
            t.address_normalized AS t_address_normalized, t.address_tokens AS t_address_tokens,
            t.postal_code AS t_postal_code, t.street_number AS t_street_number,
            t.city_candidate AS t_city_candidate, t.state_candidate AS t_state_candidate,
            t.name_has_non_latin AS t_name_has_non_latin, t.address_has_non_latin AS t_address_has_non_latin
        FROM bucketed_candidates bc
        JOIN {io_utils.parquet_sql(s1_path)} s1 ON s1.entity_id = bc.source1_entity_id
        JOIN {io_utils.parquet_sql(t_path)} t ON t.entity_id = bc.candidate_id
    """


def run_target(target_key, con, executor):
    t_start = time.time()
    config.SCORING_TMP_DIR.mkdir(parents=True, exist_ok=True)
    cand_out_path = config.SCORING_TMP_DIR / f"scored_candidates_{target_key}.tsv"
    match_out_path = config.SCORING_TMP_DIR / f"decoded_matches_{target_key}.tsv"

    print(f"\n{'='*100}\n[{target_key}] STARTING\n{'='*100}", flush=True)
    print(f"[{target_key}] RSS: {_mem_gb():.2f}GB, disk free: {_disk_free_gb():.1f}GB", flush=True)

    n_scored = 0
    n_matches = 0
    n_entities_done = 0
    batch_count = 0
    max_prob = -1.0
    min_prob = 2.0
    n_nonfinite = 0

    cand_f = open(cand_out_path, "w", encoding="utf-8", newline="\n")
    match_f = open(match_out_path, "w", encoding="utf-8", newline="\n")
    in_flight = []

    def handle_result(s1_ids, cand_ids, probs):
        nonlocal n_scored, n_matches, max_prob, min_prob, n_nonfinite
        n = len(s1_ids)
        n_scored += n
        probs_arr = np.array(probs, dtype=np.float64)
        finite_mask = np.isfinite(probs_arr)
        n_bad = int((~finite_mask).sum())
        if n_bad:
            n_nonfinite += n_bad
        if finite_mask.any():
            max_prob_local = float(probs_arr[finite_mask].max())
            min_prob_local = float(probs_arr[finite_mask].min())
            if max_prob_local > max_prob:
                max_prob = max_prob_local
            if min_prob_local < min_prob:
                min_prob = min_prob_local
        for sid, cid in zip(s1_ids, cand_ids):
            cand_f.write(f"{sid}\t{cid}\n")
        by_entity = {}
        for sid, cid, p in zip(s1_ids, cand_ids, probs):
            by_entity.setdefault(sid, []).append((cid, float(p)))
        for sid, pairs in by_entity.items():
            accepted = decode_probability_gap(pairs)
            for cid, p in accepted:
                match_f.write(f"{sid}\t{cid}\t{p:.6f}\n")
            n_matches += len(accepted)

    def drain_completed(block_all=False):
        nonlocal in_flight
        still_pending = []
        for fut in in_flight:
            if block_all or fut.done():
                s1_ids, cand_ids, probs = fut.result()
                handle_result(s1_ids, cand_ids, probs)
            else:
                still_pending.append(fut)
        in_flight = still_pending

    MAX_IN_FLIGHT = N_WORKERS * 2

    t_loop = time.time()
    for bucket_id in range(NUM_BUCKETS):
        query = build_bucket_query(target_key, bucket_id)
        table = con.sql(query).to_arrow_table()
        n_rows = table.num_rows
        if n_rows == 0:
            continue
        s1_id_col = table.column("source1_entity_id").to_pylist()
        by_entity_idx = {}
        for i, sid in enumerate(s1_id_col):
            by_entity_idx.setdefault(sid, []).append(i)
        del s1_id_col

        entity_ids = list(by_entity_idx.keys())
        for start in range(0, len(entity_ids), BATCH_SIZE_ENTITIES):
            chunk_entities = entity_ids[start:start + BATCH_SIZE_ENTITIES]
            idx = [i for eid in chunk_entities for i in by_entity_idx[eid]]
            batch_table = table.take(idx)

            in_flight.append(executor.submit(_worker_process_batch, batch_table))
            n_entities_done += len(chunk_entities)
            batch_count += 1

            if len(in_flight) >= MAX_IN_FLIGHT:
                drain_completed(block_all=False)
                while len(in_flight) >= MAX_IN_FLIGHT:
                    in_flight[0].result()
                    drain_completed(block_all=False)

            if batch_count % PROGRESS_EVERY_BATCHES == 0:
                drain_completed(block_all=False)
                elapsed = time.time() - t_loop
                rate = n_entities_done / elapsed if elapsed > 0 else 0
                print(f"[{target_key}] bucket {bucket_id}/{NUM_BUCKETS} batch {batch_count}: "
                      f"{n_entities_done:,} entities, {n_scored:,} scored, {n_matches:,} matches, "
                      f"RSS={_mem_gb():.2f}GB, elapsed={elapsed:.1f}s, rate={rate:.1f} ent/s, "
                      f"disk_free={_disk_free_gb():.1f}GB", flush=True)

        del by_entity_idx, table

    drain_completed(block_all=True)
    cand_f.close()
    match_f.close()

    elapsed = time.time() - t_start
    print(f"[{target_key}] DONE: {n_entities_done:,} entities, {n_scored:,} candidates scored, "
          f"{n_matches:,} matches decoded, {batch_count} batches, {elapsed:.1f}s", flush=True)
    print(f"  probability range: [{min_prob:.4f}, {max_prob:.4f}], non-finite: {n_nonfinite}", flush=True)
    return dict(target=target_key, n_entities=n_entities_done, n_scored=n_scored, n_matches=n_matches,
                elapsed_seconds=round(elapsed, 1), min_prob=min_prob, max_prob=max_prob,
                n_nonfinite=n_nonfinite, cand_out_path=str(cand_out_path), match_out_path=str(match_out_path))


def main():
    print(f"RAM available: {psutil.virtual_memory().available/1e9:.2f} GB", flush=True)
    print(f"Disk free: {_disk_free_gb():.1f} GB", flush=True)
    print(f"N_WORKERS: {N_WORKERS}", flush=True)

    con = io_utils.get_configured_connection()
    con.execute("SET memory_limit='3GB'")
    con.execute("SET preserve_insertion_order=false")

    results = {}
    for target_key, model_path in TARGET_MODELS.items():
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init,
                                  initargs=(str(model_path),)) as executor:
            r = run_target(target_key, con, executor)
        results[target_key] = r

    print("\n=== SUMMARY ===", flush=True)
    for k, v in results.items():
        print(k, v, flush=True)


if __name__ == "__main__":
    main()
