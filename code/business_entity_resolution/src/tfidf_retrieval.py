"""Final-pipeline candidate retrieval: inverted-index name + address search,
replacing the earlier rule-based blocking (see EXPERIMENTS.md) and an
initial brute-force sklearn-TF-IDF-cosine prototype that was too slow to run
at full scale (a few hundred queries/sec single-threaded).

Two views, each built as an in-memory inverted index (token -> postings) over
the TARGET corpus, with IDF weighting and a document-frequency ceiling
(MAX_DF) that keeps a very common token from blowing up per-query cost:
  B (name)    -- name_core, whitespace-tokenized
  C (address) -- address_tokens (already tokenized during normalization)

Per S1 query: look up postings for its own tokens in each view, score
candidates by summed IDF of shared tokens (via a vectorized
concatenate+unique+bincount, never a dense N-target-length array), take the
top K per view, and UNION the two views' top-K sets as the final candidate
set for that S1 entity.

Validated on an 8,000-entity training sample (MAX_DF=20,000, K=20):
  combined recall 91.17% (S2=93.20%, S3=89.26%), ~29 candidates/entity,
  vs. ~81.5% for the earlier rule-based blocking -- see docs/retrieval
  results in the root README for the full comparison table.

Full test-set runtime: ~700 q/s sustained per target (~40-45 min/target on
an 8-core/16GB machine), RSS held under ~2GB throughout by writing results
in bounded batches (FLUSH_EVERY) via pyarrow.parquet.ParquetWriter rather
than holding all (source1_entity_id, candidate_id) pairs in memory at once.
"""
import time
import gc
import shutil

import numpy as np
import psutil
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from . import config

K = 20
MAX_DF = 20_000
FLUSH_EVERY = 50_000  # bound in-memory pair accumulation before each parquet write


def _mem_gb():
    return psutil.Process().memory_info().rss / 1e9


def build_index(tokens_list, max_df=MAX_DF):
    """token -> (postings int32 array, idf weight). A token with document
    frequency above max_df is dropped from the index entirely -- the device
    that keeps per-query cost bounded regardless of corpus size."""
    N = len(tokens_list)
    df_count = {}
    for toks in tokens_list:
        if toks is None:
            continue
        for tk in set(toks):
            df_count[tk] = df_count.get(tk, 0) + 1
    postings_lists = {tk: [] for tk, c in df_count.items() if c <= max_df}
    for i, toks in enumerate(tokens_list):
        if toks is None:
            continue
        for tk in set(toks):
            lst = postings_lists.get(tk)
            if lst is not None:
                lst.append(i)
    postings = {tk: np.array(lst, dtype=np.int32) for tk, lst in postings_lists.items()}
    idf = {tk: (np.log((N + 1) / (df_count[tk] + 1)) + 1.0) for tk in postings}
    return postings, idf, N


def query_topk(tokens, postings, idf, k):
    """Top-k candidate row indices by summed IDF of shared tokens, computed
    via concatenate+unique+bincount over only the postings that actually
    matched -- never an O(target_size) array per query."""
    if not tokens:
        return np.array([], dtype=np.int32)
    ids_parts = []
    w_parts = []
    for tk in set(tokens):
        p = postings.get(tk)
        if p is None or len(p) == 0:
            continue
        w = idf[tk]
        ids_parts.append(p)
        w_parts.append(np.full(len(p), w, dtype=np.float64))
    if not ids_parts:
        return np.array([], dtype=np.int32)
    ids_concat = np.concatenate(ids_parts)
    w_concat = np.concatenate(w_parts)
    uniq_ids, inverse = np.unique(ids_concat, return_inverse=True)
    scores = np.bincount(inverse, weights=w_concat)
    if len(uniq_ids) > k:
        top_idx = np.argpartition(scores, -k)[-k:]
    else:
        top_idx = np.arange(len(uniq_ids))
    return uniq_ids[top_idx]


def name_toks(s):
    return s.split() if s else []


def main():
    config.TFIDF_RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    t0 = time.time()

    s1_path = config.NORMALIZED_FILES["test_source1"]
    print(f"[{time.time()-t0:.1f}s] loading full S1 test set...", flush=True)
    s1_df = con.execute(f"""
        SELECT entity_id, name_core, address_tokens
        FROM read_parquet('{s1_path}')
    """).df()
    n_s1 = len(s1_df)
    s1_ids = s1_df["entity_id"].values
    s1_name_list = s1_df["name_core"].tolist()
    s1_addr_list = s1_df["address_tokens"].tolist()
    print(f"[{time.time()-t0:.1f}s] S1: {n_s1:,} rows, RSS={_mem_gb():.2f}GB", flush=True)

    for target_key in ["test_source2", "test_source3"]:
        t_target0 = time.time()
        t_path = config.NORMALIZED_FILES[target_key]
        print(f"\n=== {target_key} ===", flush=True)
        t_df = con.execute(f"""
            SELECT entity_id, name_core, address_tokens
            FROM read_parquet('{t_path}')
        """).df()
        t_ids = t_df["entity_id"].values
        n_t = len(t_df)
        print(f"[{time.time()-t0:.1f}s] target: {n_t:,} rows, RSS={_mem_gb():.2f}GB", flush=True)

        name_tokens_list = [name_toks(x) for x in t_df["name_core"].tolist()]
        addr_tokens_list = [list(x) if x is not None else [] for x in t_df["address_tokens"].tolist()]

        tvb0 = time.time()
        postings_name, idf_name, _ = build_index(name_tokens_list)
        print(f"[{time.time()-t0:.1f}s] name index built ({time.time()-tvb0:.1f}s), "
              f"tokens={len(postings_name):,}, RSS={_mem_gb():.2f}GB", flush=True)

        tvb0 = time.time()
        postings_addr, idf_addr, _ = build_index(addr_tokens_list)
        print(f"[{time.time()-t0:.1f}s] address index built ({time.time()-tvb0:.1f}s), "
              f"tokens={len(postings_addr):,}, RSS={_mem_gb():.2f}GB", flush=True)

        del name_tokens_list, addr_tokens_list, t_df
        gc.collect()

        out_path = config.TFIDF_RETRIEVAL_DIR / f"test_candidates_{target_key.split('_')[1]}.parquet"
        schema = pa.schema([("source1_entity_id", pa.string()), ("candidate_id", pa.string())])
        writer = pq.ParquetWriter(str(out_path), schema)

        out_s1 = []
        out_cand = []
        counts = np.empty(n_s1, dtype=np.int32)
        total_pairs = 0
        tq0 = time.time()
        for i in range(n_s1):
            nt = name_toks(s1_name_list[i])
            at = list(s1_addr_list[i]) if s1_addr_list[i] is not None else []
            idx_name = query_topk(nt, postings_name, idf_name, K)
            idx_addr = query_topk(at, postings_addr, idf_addr, K)
            union_idx = np.union1d(idx_name, idx_addr)
            counts[i] = len(union_idx)
            if len(union_idx):
                sid = s1_ids[i]
                for cid in t_ids[union_idx]:
                    out_s1.append(sid)
                    out_cand.append(cid)

            if (i + 1) % FLUSH_EVERY == 0 or i == n_s1 - 1:
                if out_s1:
                    writer.write_table(pa.table({"source1_entity_id": out_s1, "candidate_id": out_cand}, schema=schema))
                    total_pairs += len(out_s1)
                    out_s1 = []
                    out_cand = []

            if (i + 1) % 100_000 == 0:
                elapsed = time.time() - tq0
                rate = (i + 1) / elapsed
                eta = (n_s1 - (i + 1)) / rate
                disk_free = shutil.disk_usage(str(config.DERIVED_ROOT)).free / 1e9
                print(f"  {i+1:,}/{n_s1:,} S1, {elapsed:.1f}s, rate={rate:.0f} q/s, "
                      f"ETA={eta/60:.1f}min, cands={total_pairs:,}, RSS={_mem_gb():.2f}GB, "
                      f"disk_free={disk_free:.1f}GB", flush=True)
        writer.close()
        tq1 = time.time()
        print(f"[{time.time()-t0:.1f}s] retrieval done for {target_key}: {tq1-tq0:.1f}s, "
              f"rate={n_s1/(tq1-tq0):.0f} q/s", flush=True)

        print(f"[{time.time()-t0:.1f}s] {target_key} DONE: {total_pairs:,} total pairs, "
              f"avg/s1={counts.mean():.1f}, p95={np.percentile(counts,95):.0f}, max={counts.max()}, "
              f"zero={int((counts==0).sum())}, target_time={time.time()-t_target0:.1f}s, wrote {out_path}", flush=True)

        del postings_name, postings_addr, idf_name, idf_addr, counts, t_ids
        gc.collect()

    print(f"\n[{time.time()-t0:.1f}s] ALL RETRIEVAL DONE, RSS={_mem_gb():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
