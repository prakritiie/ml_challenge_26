"""First real LightGBM baseline matcher. Baseline only -- no hyperparameter
tuning, no new features, no blocking changes, no embeddings.

GPU: NOT used. LightGBM's PyPI wheel has no GPU/CUDA support compiled in
(confirmed empirically in the previous stage), so this trains on CPU only,
per instructions.

Memory safety:
- Only MODEL_FEATURE_COLUMNS (27) are ever loaded, never the ID/split columns.
- The 32M/34M-row feature Parquet files are streamed through DuckDB's Arrow
  batch reader directly into a pre-allocated float32 numpy array -- pandas
  is never used for the bulk row data (only for small, already-aggregated
  summary tables of a few hundred rows at most).
- S2 and S3 are never loaded simultaneously (this run only touches S2 --
  training S3 is a separate, later invocation, not this stage).

Usage:
    python -m src.train_baseline --target train_source2
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import psutil
import lightgbm as lgb
import duckdb

from . import config, io_utils
from .matching_dataset import (
    MODEL_FEATURE_COLUMNS, LABEL_COLUMN, SPLIT_COLUMN, PREFIX,
    MATCHING_DIR, build_entity_split, build_gt_full,
    feature_select_expressions, build_provenance_table,
)

MODELS_DIR = config.DERIVED_ROOT.parent / "code" / "business_entity_resolution" / "models"

LGB_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "verbosity": -1,
    "device_type": "cpu",
    "num_threads": 4,
    # scale_pos_weight deliberately NOT set -- see README.md "Model training
    # stage: first LightGBM baseline" for the documented reasoning: the
    # dataset was already downsampled 10:1 (not the raw ~34:1 imbalance),
    # and upweighting positives further would push a precision-heavy (F0.5)
    # problem toward more recall at precision's expense, the wrong direction
    # for this challenge. Revisit at the tuning stage, not here.
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30
BATCH_SIZE = 500_000
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def _mem_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


# ---------------------------------------------------------------------------
# Memory-safe streaming load: Parquet -> DuckDB Arrow batches -> float32 numpy
# (never a pandas DataFrame of the full row count)
# ---------------------------------------------------------------------------
def load_split_arrays(con, feature_path: Path, split_value: str,
                       feature_columns: List[str], batch_size: int = BATCH_SIZE,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    cols_sql = ", ".join(feature_columns + [LABEL_COLUMN])
    n_total = con.execute(
        f"SELECT count(*) FROM read_parquet('{feature_path}') WHERE {SPLIT_COLUMN}='{split_value}'"
    ).fetchone()[0]
    n_features = len(feature_columns)

    est_batch_mb = batch_size * n_features * 4 / 1e6
    print(f"  [{split_value}] rows={n_total:,}, est. memory for one batch of {batch_size:,} rows: {est_batch_mb:.1f} MB")
    print(f"  [{split_value}] est. final array size: {n_total * n_features * 4 / 1e9:.2f} GB (float32 features) "
          f"+ {n_total * 4 / 1e9:.3f} GB (int32 label)")

    X = np.empty((n_total, n_features), dtype=np.float32)
    y = np.empty(n_total, dtype=np.int32)

    query = f"SELECT {cols_sql} FROM read_parquet('{feature_path}') WHERE {SPLIT_COLUMN}='{split_value}'"
    reader = con.sql(query).fetch_arrow_reader(batch_size=batch_size)

    offset = 0
    t0 = time.time()
    for i, batch in enumerate(reader):
        n_batch = batch.num_rows
        for j in range(n_features):
            col = pc.cast(batch.column(j), pa.float32())
            X[offset:offset + n_batch, j] = col.to_numpy(zero_copy_only=False)
        y[offset:offset + n_batch] = batch.column(n_features).to_numpy(zero_copy_only=False).astype(np.int32)
        offset += n_batch
        if verbose and (i % 10 == 0 or offset == n_total):
            elapsed = time.time() - t0
            print(f"    batch {i}: {offset:,}/{n_total:,} rows loaded ({elapsed:.1f}s, RSS={_mem_gb():.2f}GB)")

    assert offset == n_total
    return X, y


def verify_entity_split_safety(con, feature_path: Path) -> int:
    """Re-check (not assume) that no S1 entity spans both splits, directly
    on the stored feature file, before training."""
    n_overlap = con.execute(f"""
        SELECT count(*) FROM (
            SELECT source1_entity_id FROM read_parquet('{feature_path}')
            GROUP BY source1_entity_id HAVING count(DISTINCT {SPLIT_COLUMN}) > 1
        )
    """).fetchone()[0]
    return n_overlap


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, X_valid, y_valid, feature_names: List[str]):
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=False)

    evals_result = {}
    t0 = time.time()
    bst = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=25),
            lgb.record_evaluation(evals_result),
        ],
    )
    train_time = time.time() - t0
    return bst, evals_result, train_time


# ---------------------------------------------------------------------------
# Full (undownsampled) validation-entity candidate features, for faithful
# matching-specific evaluation -- reusing feature_select_expressions /
# build_provenance_table from matching_dataset.py rather than duplicating
# the SQL. Filters the by-rule table by ENTITY first (cheap single-key
# join) before grouping, instead of the tuple-IN semi-join used at training
# time, since that join was the dominant cost in the previous stage.
# ---------------------------------------------------------------------------
def build_validation_full_features(con, target_key: str) -> dict:
    prefix = PREFIX[target_key]
    s1_path = config.NORMALIZED_FILES["train_source1"]
    t_path = config.NORMALIZED_FILES[target_key]
    candidates_path = MATCHING_DIR / f"candidates_{target_key}.parquet"
    by_rule_path = MATCHING_DIR / f"candidates_by_rule_{target_key}.parquet"
    out_path = MATCHING_DIR / f"validation_full_features_{target_key}.parquet"

    build_entity_split(con)  # deterministic -- s1_split temp table
    build_gt_full(con)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE val_entities AS
        SELECT source1_entity_id FROM s1_split WHERE split='validation'
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE val_gt AS
        SELECT * FROM gt_exploded_full WHERE matched_id LIKE '{prefix}%'
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE val_gt_counts AS
        SELECT ve.source1_entity_id, count(g.matched_id) AS n_gt
        FROM val_entities ve
        LEFT JOIN val_gt g ON g.source1_entity_id = ve.source1_entity_id
        GROUP BY ve.source1_entity_id
    """)
    total_gt_validation = con.execute("SELECT sum(n_gt) FROM val_gt_counts").fetchone()[0]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE val_candidates AS
        SELECT c.source1_entity_id, c.candidate_id
        FROM read_parquet('{candidates_path}') c
        JOIN val_entities ve ON ve.source1_entity_id = c.source1_entity_id
    """)
    n_val_candidates = con.execute("SELECT count(*) FROM val_candidates").fetchone()[0]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE val_labeled AS
        SELECT vc.source1_entity_id, vc.candidate_id,
               CASE WHEN g.matched_id IS NOT NULL THEN 1 ELSE 0 END AS label
        FROM val_candidates vc
        LEFT JOIN val_gt g ON vc.source1_entity_id = g.source1_entity_id AND vc.candidate_id = g.matched_id
    """)
    gt_captured_validation = con.execute("SELECT sum(label) FROM val_labeled").fetchone()[0]

    # entity-first filter (fast) instead of the tuple-IN semi-join
    build_provenance_table(
        con,
        by_rule_source_sql=(
            f"(SELECT r.* FROM read_parquet('{by_rule_path}') r "
            f"JOIN val_entities ve ON ve.source1_entity_id = r.source1_entity_id)"
        ),
        out_table="val_provenance",
    )

    con.execute(f"""
        COPY (
            SELECT
                vl.source1_entity_id, vl.candidate_id, vl.label,
                {feature_select_expressions('s1', 't', 'p')}
            FROM val_labeled vl
            JOIN {io_utils.parquet_sql(s1_path)} s1 ON s1.entity_id = vl.source1_entity_id
            JOIN {io_utils.parquet_sql(t_path)} t ON t.entity_id = vl.candidate_id
            LEFT JOIN val_provenance p ON p.source1_entity_id = vl.source1_entity_id AND p.candidate_id = vl.candidate_id
        ) TO '{out_path}' (FORMAT PARQUET)
    """)

    return {
        "target": target_key,
        "validation_full_features_path": str(out_path),
        "n_validation_candidates": n_val_candidates,
        "total_gt_matches_validation": int(total_gt_validation),
        "gt_captured_by_blocking_validation": int(gt_captured_validation),
        "candidate_ceiling_recall_validation": round(gt_captured_validation / total_gt_validation, 4),
    }


# ---------------------------------------------------------------------------
# Scoring + compact prediction output
# ---------------------------------------------------------------------------
def score_validation(con, bst, val_features_path: Path, target_key: str) -> Path:
    out_path = MATCHING_DIR / f"validation_predictions_{target_key}.parquet"
    cols_sql = ", ".join(MODEL_FEATURE_COLUMNS)
    n_total = con.execute(f"SELECT count(*) FROM read_parquet('{val_features_path}')").fetchone()[0]

    query = f"SELECT source1_entity_id, candidate_id, label, {cols_sql} FROM read_parquet('{val_features_path}')"
    reader = con.sql(query).fetch_arrow_reader(batch_size=BATCH_SIZE)

    import pyarrow.parquet as pq
    writer = None
    t0 = time.time()
    n_done = 0
    for batch in reader:
        n_batch = batch.num_rows
        X = np.empty((n_batch, len(MODEL_FEATURE_COLUMNS)), dtype=np.float32)
        for j in range(len(MODEL_FEATURE_COLUMNS)):
            X[:, j] = pc.cast(batch.column(3 + j), pa.float32()).to_numpy(zero_copy_only=False)
        probs = bst.predict(X, num_iteration=bst.best_iteration)

        out_table = pa.table({
            "source1_entity_id": batch.column(0),
            "candidate_entity_id": batch.column(1),
            "label": batch.column(2),
            "prediction_probability": pa.array(probs.astype(np.float32)),
            "target_source": pa.array([target_key.replace("train_", "")] * n_batch),
        })
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), out_table.schema)
        writer.write_table(out_table)
        n_done += n_batch
    if writer is not None:
        writer.close()
    print(f"  scored {n_done:,} rows in {time.time()-t0:.1f}s -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Matching-specific evaluation
# ---------------------------------------------------------------------------
def pair_level_metrics(con, predictions_path: Path, thresholds: List[float]) -> list:
    rows = []
    for t in thresholds:
        r = con.execute(f"""
            SELECT
              sum(CASE WHEN label=1 AND prediction_probability>={t} THEN 1 ELSE 0 END) AS tp,
              sum(CASE WHEN label=0 AND prediction_probability>={t} THEN 1 ELSE 0 END) AS fp,
              sum(CASE WHEN label=1 AND prediction_probability<{t} THEN 1 ELSE 0 END) AS fn
            FROM read_parquet('{predictions_path}')
        """).fetchone()
        tp, fp, fn = r
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f05 = (1.25 * precision * recall / (0.25 * precision + recall)) if (precision + recall) else 0.0
        rows.append({"threshold": t, "pair_precision": round(precision, 4), "pair_recall": round(recall, 4),
                     "pair_f0.5": round(f05, 4), "tp": tp, "fp": fp, "fn": fn})
    return rows


def entity_level_metrics(con, predictions_path: Path, target_key: str, thresholds: List[float]) -> dict:
    """Macro-averaged entity-level precision/recall/F0.5 at each threshold,
    matching the challenge's own scoring convention (singleton correctly
    predicted empty = 1.0; singleton with any prediction = 0.0). Uses
    val_gt_counts (built in build_validation_full_features) so entities
    whose true matches were entirely missed by blocking are still counted
    with their real n_gt, not silently dropped."""
    results = {}
    pred_count_dist = {}
    segment_results = {}

    for t in thresholds:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE entity_stats AS
            SELECT ve.source1_entity_id, gt.n_gt,
                   coalesce(sum(CASE WHEN p.label=1 AND p.prediction_probability>={t} THEN 1 ELSE 0 END), 0) AS tp,
                   coalesce(sum(CASE WHEN p.label=0 AND p.prediction_probability>={t} THEN 1 ELSE 0 END), 0) AS fp
            FROM val_entities ve
            JOIN val_gt_counts gt ON gt.source1_entity_id = ve.source1_entity_id
            LEFT JOIN read_parquet('{predictions_path}') p ON p.source1_entity_id = ve.source1_entity_id
            GROUP BY ve.source1_entity_id, gt.n_gt
        """)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE entity_scores AS
            SELECT source1_entity_id, n_gt, tp, fp, (tp+fp) AS n_pred,
                   CASE WHEN n_gt = 0 THEN (CASE WHEN tp+fp = 0 THEN 1.0 ELSE 0.0 END)
                        ELSE (CASE WHEN tp+fp = 0 THEN 0.0 ELSE tp*1.0/(tp+fp) END) END AS precision,
                   CASE WHEN n_gt = 0 THEN (CASE WHEN tp+fp = 0 THEN 1.0 ELSE 0.0 END)
                        ELSE tp*1.0/n_gt END AS recall
            FROM entity_stats
        """)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE entity_f05 AS
            SELECT *, CASE WHEN precision+recall = 0 THEN 0.0
                           ELSE 1.25*precision*recall/(0.25*precision+recall) END AS f05,
                   CASE WHEN n_gt = 0 THEN 'singleton' WHEN n_gt = 1 THEN 'one_match' ELSE 'multi_match' END AS segment
            FROM entity_scores
        """)

        overall = con.execute("""
            SELECT count(*), avg(precision), avg(recall), avg(f05),
                   sum(CASE WHEN n_pred=0 THEN 1 ELSE 0 END),
                   sum(CASE WHEN n_pred>0 THEN 1 ELSE 0 END)
            FROM entity_f05
        """).fetchone()
        results[t] = {
            "n_entities": overall[0], "entity_precision": round(overall[1], 4),
            "entity_recall": round(overall[2], 4), "entity_f0.5": round(overall[3], 4),
            "n_zero_predictions": overall[4], "n_with_predictions": overall[5],
        }

        dist = con.execute("""
            SELECT CASE WHEN n_pred=0 THEN '0' WHEN n_pred=1 THEN '1' WHEN n_pred=2 THEN '2' ELSE '3+' END AS bucket,
                   count(*) FROM entity_f05 GROUP BY bucket
        """).fetchdf()
        pred_count_dist[t] = dict(zip(dist["bucket"], dist[dist.columns[1]]))

        seg = con.execute("""
            SELECT segment, count(*), avg(precision), avg(recall), avg(f05)
            FROM entity_f05 GROUP BY segment
        """).fetchdf()
        segment_results[t] = {row["segment"]: {"n": row[seg.columns[1]], "precision": round(row[seg.columns[2]], 4),
                                                 "recall": round(row[seg.columns[3]], 4), "f0.5": round(row[seg.columns[4]], 4)}
                               for _, row in seg.iterrows()}

    return {"overall": results, "prediction_count_distribution": pred_count_dist, "by_segment": segment_results}


def false_positive_and_missed_entities(con, predictions_path: Path, threshold: float, limit: int = 5) -> dict:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fp_entities AS
        SELECT DISTINCT source1_entity_id FROM read_parquet('{predictions_path}')
        WHERE label=0 AND prediction_probability >= {threshold}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE missed_entities AS
        SELECT DISTINCT source1_entity_id FROM read_parquet('{predictions_path}')
        WHERE label=1 AND prediction_probability < {threshold}
    """)
    n_fp = con.execute("SELECT count(*) FROM fp_entities").fetchone()[0]
    n_missed = con.execute("SELECT count(*) FROM missed_entities").fetchone()[0]
    return {"threshold": threshold, "n_entities_with_false_positive": n_fp, "n_entities_with_missed_match": n_missed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first LightGBM baseline matcher")
    parser.add_argument("--target", required=True, choices=["train_source2", "train_source3"])
    args = parser.parse_args()
    target_key = args.target

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = MATCHING_DIR / f"features_{target_key}.parquet"

    con = io_utils.get_configured_connection()
    t_start = time.time()

    print("=" * 100)
    print(f"STEP 4: entity-level split re-verification -- {target_key}")
    print("=" * 100)
    n_overlap = verify_entity_split_safety(con, feature_path)
    print(f"entity overlap between train/validation (must be 0): {n_overlap}")
    assert n_overlap == 0, "REFUSING TO TRAIN: train/validation entity split is not safe."

    n_total = con.execute(f"SELECT count(*) FROM read_parquet('{feature_path}')").fetchone()[0]
    n_pos = con.execute(f"SELECT count(*) FROM read_parquet('{feature_path}') WHERE label=1").fetchone()[0]
    n_neg = n_total - n_pos
    neg_pos_ratio = n_neg / n_pos

    print("\n" + "=" * 100)
    print(f"STEP 5: pre-training summary -- {target_key}")
    print("=" * 100)
    print(f"row count: {n_total:,}")
    print(f"positive count: {n_pos:,}")
    print(f"negative count: {n_neg:,}")
    print(f"positive rate: {n_pos/n_total:.4%}")
    print(f"negative:positive ratio: {neg_pos_ratio:.3f} (computed, NOT blindly fed to scale_pos_weight -- see README.md)")
    print(f"number of features: {len(MODEL_FEATURE_COLUMNS)}")
    print(f"feature names: {MODEL_FEATURE_COLUMNS}")
    print(f"LightGBM config: {LGB_PARAMS}")
    print(f"num_boost_round={NUM_BOOST_ROUND}, early_stopping_rounds={EARLY_STOPPING_ROUNDS}")
    print(f"RSS before loading: {_mem_gb():.2f} GB")

    print("\nLoading train split...")
    X_train, y_train = load_split_arrays(con, feature_path, "train", MODEL_FEATURE_COLUMNS)
    print("\nLoading validation split...")
    X_valid, y_valid = load_split_arrays(con, feature_path, "validation", MODEL_FEATURE_COLUMNS)
    load_time = time.time() - t_start
    print(f"\nRSS after loading: {_mem_gb():.2f} GB, load time: {load_time:.1f}s")

    print("\n" + "=" * 100)
    print("TRAINING")
    print("=" * 100)
    bst, evals_result, train_time = train_model(X_train, y_train, X_valid, y_valid, MODEL_FEATURE_COLUMNS)
    print(f"\nRSS after training: {_mem_gb():.2f} GB, train time: {train_time:.1f}s")
    print(f"best_iteration: {bst.best_iteration}")

    model_path = MODELS_DIR / f"lightgbm_{target_key.replace('train_', '')}_baseline.txt"
    bst.save_model(str(model_path))
    print(f"model saved: {model_path}")

    train_auc = evals_result["train"]["auc"][bst.best_iteration - 1]
    valid_auc = evals_result["valid"]["auc"][bst.best_iteration - 1]
    train_logloss = evals_result["train"]["binary_logloss"][bst.best_iteration - 1]
    valid_logloss = evals_result["valid"]["binary_logloss"][bst.best_iteration - 1]
    print(f"train AUC: {train_auc:.4f}, valid AUC: {valid_auc:.4f}")
    print(f"train logloss: {train_logloss:.4f}, valid logloss: {valid_logloss:.4f}")

    # Free the training arrays before the (separate, larger) full-candidate
    # validation scoring pass -- no need to hold both in memory at once.
    del X_train, y_train, X_valid, y_valid
    print(f"RSS after freeing train/valid arrays: {_mem_gb():.2f} GB")

    print("\n" + "=" * 100)
    print("STEP 7 (setup): building FULL (undownsampled) validation-entity candidate features")
    print("=" * 100)
    t0 = time.time()
    val_build_info = build_validation_full_features(con, target_key)
    for k, v in val_build_info.items():
        print(f"  {k}: {v}")
    print(f"  build time: {time.time()-t0:.1f}s")

    print("\n" + "=" * 100)
    print("Scoring validation candidates")
    print("=" * 100)
    predictions_path = score_validation(con, bst, Path(val_build_info["validation_full_features_path"]), target_key)

    print("\n" + "=" * 100)
    print("STEP 6: pair-level threshold table")
    print("=" * 100)
    pair_table = pair_level_metrics(con, predictions_path, THRESHOLDS)
    for row in pair_table:
        print(f"  {row}")

    print("\n" + "=" * 100)
    print("STEP 6/8: entity-level metrics, singleton/segment analysis, prediction-count distribution")
    print("=" * 100)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE val_entities AS SELECT source1_entity_id FROM s1_split WHERE split='validation'
    """)
    entity_results = entity_level_metrics(con, predictions_path, target_key, THRESHOLDS)
    for t in THRESHOLDS:
        print(f"\n  threshold={t}: {entity_results['overall'][t]}")
        print(f"    prediction-count distribution: {entity_results['prediction_count_distribution'][t]}")
        print(f"    by segment: {entity_results['by_segment'][t]}")

    print("\n" + "=" * 100)
    print("STEP 7: candidate-ceiling vs model recall vs final matching recall")
    print("=" * 100)
    total_gt = val_build_info["total_gt_matches_validation"]
    gt_captured = val_build_info["gt_captured_by_blocking_validation"]
    print(f"  BLOCKING RECALL (candidate ceiling)  = {gt_captured}/{total_gt} = {gt_captured/total_gt:.4f}")
    for t in THRESHOLDS:
        tp = pair_table[THRESHOLDS.index(t)]["tp"]
        model_recall_within_candidates = tp / gt_captured if gt_captured else None
        final_matching_recall = tp / total_gt if total_gt else None
        print(f"  threshold={t}: MODEL RECALL (within candidates) = {tp}/{gt_captured} = {model_recall_within_candidates:.4f}  |  "
              f"FINAL MATCHING RECALL (vs all true matches) = {tp}/{total_gt} = {final_matching_recall:.4f}")

    print("\n" + "=" * 100)
    print("False-positive / missed-match entity counts at a few thresholds")
    print("=" * 100)
    for t in [0.3, 0.5, 0.7]:
        print(f"  {false_positive_and_missed_entities(con, predictions_path, t)}")

    total_time = time.time() - t_start
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"model_path: {model_path}")
    print(f"validation_predictions_path: {predictions_path}")
    print(f"total runtime: {total_time:.1f}s")
    print(f"final RSS: {_mem_gb():.2f} GB")


if __name__ == "__main__":
    main()
