"""Merge the per-target (test_source2 / test_source3) scored-candidate and
decoded-match TSVs from score_candidates.py into the two final submission
files, one row per test S1 entity each:

  output/matching_results.tsv   -- source1_entity_id, matched_entity_ids
  output/candidate_pairs.tsv    -- source1_entity_id, candidate_entity_ids

candidate_pairs.tsv is built via the SAME 60-hash-bucket streaming pattern
used throughout the retrieval and scoring stages, reading directly from the
retrieval parquet files (test_candidates_source{2,3}.parquet) rather than
from the (much larger, duplicated) scored_candidates TSVs -- both contain
the exact same (source1_entity_id, candidate_id) universe, since every
retrieved candidate was scored with nothing filtered out beforehand. A
single global `GROUP BY ... string_agg(...)` over the full ~133M-row
candidate set was tried first and reliably hit DuckDB's memory ceiling
(OOM'd at both a 4GB and a 6GB limit); processing 60 small hash-buckets
instead keeps each in-memory aggregation tiny and was the only version that
completed without spilling or crashing.

matching_results.tsv is built with plain Python file I/O rather than
DuckDB's CSV writer: DuckDB's writer was found to quote an empty
matched_entity_ids field as the literal two-character string `""` instead
of leaving it truly empty, which both fails strict TSV validation and
silently manufactures a fake "matched ID" that can never appear in
candidate_pairs.tsv. The match volume here (a few million rows total) is
small enough that a pure-Python read + group + write needs no bucketing.
"""
import time

import duckdb

from . import config

NUM_BUCKETS = 60


def _all_s1_ids(con):
    s1_path = config.NORMALIZED_FILES["test_source1"]
    return con.execute(f"SELECT entity_id FROM read_parquet('{s1_path}')").df()["entity_id"]


def build_candidate_pairs(con, t0):
    all_s1_ids = set(_all_s1_ids(con))
    print(f"[{time.time()-t0:.1f}s] total S1 entities: {len(all_s1_ids):,}", flush=True)

    s2_path = config.TFIDF_RETRIEVAL_DIR / "test_candidates_source2.parquet"
    s3_path = config.TFIDF_RETRIEVAL_DIR / "test_candidates_source3.parquet"

    seen_ids = set()
    out_path = config.OUTPUT_ROOT / "candidate_pairs.tsv"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for bucket_id in range(NUM_BUCKETS):
            query = f"""
                SELECT source1_entity_id, candidate_id FROM read_parquet('{s2_path}')
                WHERE abs(hash(source1_entity_id)) % {NUM_BUCKETS} = {bucket_id}
                UNION ALL
                SELECT source1_entity_id, candidate_id FROM read_parquet('{s3_path}')
                WHERE abs(hash(source1_entity_id)) % {NUM_BUCKETS} = {bucket_id}
            """
            table = con.sql(query).to_arrow_table()
            s1_col = table.column("source1_entity_id").to_pylist()
            cand_col = table.column("candidate_id").to_pylist()
            by_entity = {}
            for sid, cid in zip(s1_col, cand_col):
                by_entity.setdefault(sid, []).append(cid)
            for sid, cids in by_entity.items():
                f.write(f"{sid}\t{','.join(cids)}\n")
                seen_ids.add(sid)
            del table, s1_col, cand_col, by_entity
            if (bucket_id + 1) % 10 == 0:
                print(f"[{time.time()-t0:.1f}s] candidate_pairs: bucket {bucket_id+1}/{NUM_BUCKETS} done, "
                      f"{len(seen_ids):,} entities written so far", flush=True)

        missing = all_s1_ids - seen_ids
        print(f"[{time.time()-t0:.1f}s] writing {len(missing):,} zero-candidate entities", flush=True)
        for sid in missing:
            f.write(f"{sid}\t\n")

    print(f"[{time.time()-t0:.1f}s] candidate_pairs.tsv DONE: "
          f"{len(seen_ids) + len(missing):,} total S1 rows written to {out_path}", flush=True)


def build_matching_results(con, t0):
    all_s1_ids = _all_s1_ids(con).tolist()
    print(f"[{time.time()-t0:.1f}s] total S1 entities: {len(all_s1_ids):,}", flush=True)

    by_entity = {}
    for target_key in ["test_source2", "test_source3"]:
        path = config.SCORING_TMP_DIR / f"decoded_matches_{target_key}.tsv"
        print(f"[{time.time()-t0:.1f}s] reading {path}...", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                sid, cid, _prob = line.rstrip("\n").split("\t")
                by_entity.setdefault(sid, []).append(cid)
        print(f"[{time.time()-t0:.1f}s] done, {len(by_entity):,} entities with matches so far", flush=True)

    out_path = config.OUTPUT_ROOT / "matching_results.tsv"
    print(f"[{time.time()-t0:.1f}s] writing {out_path}...", flush=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            cids = by_entity.get(sid)
            f.write(f"{sid}\t{','.join(cids) if cids else ''}\n")

    print(f"[{time.time()-t0:.1f}s] matching_results.tsv DONE: {len(all_s1_ids):,} rows written", flush=True)


def main():
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET preserve_insertion_order=false")

    t0 = time.time()
    build_matching_results(con, t0)
    build_candidate_pairs(con, t0)
    print(f"\n[{time.time()-t0:.1f}s] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
