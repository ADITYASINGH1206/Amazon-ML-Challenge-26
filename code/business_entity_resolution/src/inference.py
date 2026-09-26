"""
inference.py — Full inference pipeline (Stages 1-4 on test data).

Architecture: Two-stage cascade
1. LightGBM scores ALL blocking candidates (fast: millions/second)
2. Cross-Encoder re-ranks only the top-K from LightGBM (expensive but precise)
3. Ensemble combines both scores
4. Margin-based thresholding + global deduplication

Key design decisions:
- NO top-1-per-source constraint (allows multiple matches per source)
- Margin-based threshold protects precision on ambiguous candidates
- Global dedup: each S2/S3 entity maps to at most one S1 entity
"""

import gc
import csv
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from src import config
from src.utils import log, timed, log_memory, save_tsv
from src.preprocess import load_preprocessed
from src.blocking import run_blocking, load_candidates
from src.features import (
    extract_features_for_pairs, get_feature_columns, load_embeddings_for_features,
    _safe_str,
)
from src.train_lgbm import load_lgbm_model
from src.train_cross_encoder import CrossEncoderScorer


# ─────────────────────────────────────────────────────────────
# STAGE 3a: LIGHTGBM CASCADE FILTER
# ─────────────────────────────────────────────────────────────

@timed
def lgbm_cascade_filter(
    df_features: pd.DataFrame,
    model,
    top_k: int = None,
) -> pd.DataFrame:
    """
    Score all candidates with LightGBM, keep top-K per S1 entity.

    This is the "fast filter" in the cascade — reduces ~50 candidates
    to ~5 for the expensive cross-encoder.
    """
    if top_k is None:
        top_k = config.LGBM_CASCADE_TOP_K

    if "lgbm_prob" not in df_features.columns:
        feature_cols = get_feature_columns()
        X = df_features[feature_cols].to_numpy(dtype=np.float32, copy=False)
        log.info(f"  Scoring {len(X):,} pairs with LightGBM in batches...")
        probs = np.zeros(len(X), dtype=np.float32)
        batch_sz = 5_000_000
        for i in range(0, len(X), batch_sz):
            end_i = min(i + batch_sz, len(X))
            probs[i:end_i] = model.predict_proba(X[i:end_i])[:, 1]
        df_features = df_features[["s1_id", "s2s3_id"]].copy()
        df_features["lgbm_prob"] = probs
        del X, probs
        gc.collect()

    # Ultra-fast vectorized top-k per S1 entity using pandas C implementation
    log.info(f"  Filtering to top-{top_k} per S1 entity...")
    df_filtered = (
        df_features[["s1_id", "s2s3_id", "lgbm_prob"]]
        .sort_values(["s1_id", "lgbm_prob"], ascending=[True, False])
        .groupby("s1_id", as_index=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    log.info(f"  After cascade: {len(df_filtered):,} pairs (from {len(df_features):,})")

    return df_filtered


# ─────────────────────────────────────────────────────────────
# STAGE 3b: CROSS-ENCODER RE-RANKING
# ─────────────────────────────────────────────────────────────

@timed
def cross_encoder_rerank(
    df_filtered: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    scorer: CrossEncoderScorer,
) -> pd.DataFrame:
    """
    Re-rank the LightGBM top-K candidates with the cross-encoder.

    Input: pairs filtered by LightGBM cascade
    Output: same DataFrame with added 'ce_prob' column
    """
    log.info("Building text lookup for cross-encoder re-ranking...")
    s1_text = dict(zip(df_s1["entity_id"], df_s1["name_addr"].fillna("")))
    target_text = dict(zip(df_targets["entity_id"], df_targets["name_addr"].fillna("")))

    s1_ids = df_filtered["s1_id"].values
    t_ids = df_filtered["s2s3_id"].values
    lgbm_probs = df_filtered["lgbm_prob"].values

    # Pre-filter: only send candidates with lgbm_prob >= 0.05 to Cross-Encoder
    # (Candidates below 0.05 cannot cross threshold >= 0.55 anyway)
    ce_candidate_mask = lgbm_probs >= 0.05

    text_pairs = []
    pair_indices = []
    for idx, (s1, t, eligible) in enumerate(zip(s1_ids, t_ids, ce_candidate_mask)):
        if eligible:
            ta = s1_text.get(s1, "")
            tb = target_text.get(t, "")
            if ta and tb:
                text_pairs.append((ta, tb))
                pair_indices.append(idx)

    log.info(f"  Scoring {len(text_pairs):,} promising pairs with cross-encoder (skipped {len(df_filtered) - len(text_pairs):,} low-probability candidates)...")

    ce_probs_all = np.zeros(len(df_filtered), dtype=np.float32)
    if len(text_pairs) > 0:
        ce_scores = scorer.score_pairs(text_pairs)
        ce_probs_all[pair_indices] = ce_scores
        del text_pairs, ce_scores

    df_filtered = df_filtered.copy()
    df_filtered["ce_prob"] = ce_probs_all

    del s1_text, target_text, ce_probs_all
    gc.collect()

    return df_filtered


# ─────────────────────────────────────────────────────────────
# STAGE 4: ENSEMBLE + MARGIN-BASED THRESHOLDING
# ─────────────────────────────────────────────────────────────

def compute_ensemble_score(lgbm_prob: float, ce_prob: float) -> float:
    """Weighted average ensemble of LightGBM and Cross-Encoder scores."""
    return (config.ENSEMBLE_WEIGHT_LGBM * lgbm_prob +
            config.ENSEMBLE_WEIGHT_CE * ce_prob)


def apply_thresholding(
    df_scored: pd.DataFrame,
    threshold: float,
    all_s1_ids: np.ndarray,
    use_margin: bool = True,
    margin_delta: float = None,
    global_dedup: bool = None,
) -> Dict[str, Set[str]]:
    """
    Apply precision-biased thresholding with margin protection and global dedup.

    Steps:
    1. Compute ensemble scores
    2. For each S1 entity, collect all candidates above threshold
    3. Apply margin-based filtering (reject if too close to next-best conflicting)
    4. Global deduplication: each S2/S3 assigned to at most one S1

    NO top-1-per-source constraint: all matches above τ are included.
    """
    if margin_delta is None:
        margin_delta = config.MARGIN_DELTA
    if global_dedup is None:
        global_dedup = config.GLOBAL_DEDUP

    # Step 1: Vectorized ensemble scores
    df_scored = df_scored.copy()
    if "ce_prob" in df_scored.columns:
        df_scored["ensemble_score"] = (
            config.ENSEMBLE_WEIGHT_LGBM * df_scored["lgbm_prob"].to_numpy(dtype=np.float32) +
            config.ENSEMBLE_WEIGHT_CE * df_scored["ce_prob"].to_numpy(dtype=np.float32)
        )
    else:
        df_scored["ensemble_score"] = df_scored["lgbm_prob"].to_numpy(dtype=np.float32)

    # Step 2: Vectorized filter of pairs above threshold
    above_mask = df_scored["ensemble_score"].values >= threshold
    df_above = df_scored[above_mask]

    raw_pred_map = defaultdict(set)
    for s1, s2 in zip(df_above["s1_id"].values, df_above["s2s3_id"].values):
        raw_pred_map[s1].add(s2)
    raw_predictions = {s1: raw_pred_map.get(s1, set()) for s1 in all_s1_ids}

    # Step 3: Global deduplication (each S2/S3 → at most one S1)
    if global_dedup:
        predictions = _global_dedup_with_margin(
            df_above, raw_predictions, margin_delta
        )
    else:
        predictions = raw_predictions

    # Ensure all S1 entities are present
    for s1_id in all_s1_ids:
        if s1_id not in predictions:
            predictions[s1_id] = set()

    return predictions


def _global_dedup_with_margin(
    df_above: pd.DataFrame,
    raw_predictions: Dict[str, Set[str]],
    margin_delta: float,
) -> Dict[str, Set[str]]:
    """
    Global deduplication with margin-based confidence in O(N) time.

    For each S2/S3 entity claimed by multiple S1 entities:
    - Assign to the S1 with the highest ensemble score
    - Only if margin over the second-highest S1 score > margin_delta
    - Otherwise, don't assign to any S1 (too ambiguous)
    """
    # Build reverse index directly from df_above (already above threshold!)
    reverse_index = defaultdict(list)
    for s1, s2, score in zip(df_above["s1_id"].values, df_above["s2s3_id"].values, df_above["ensemble_score"].values):
        reverse_index[s2].append((s1, float(score)))

    # Resolve conflicts
    assignments = {}  # s2s3_id → s1_id (or None)
    for s2s3_id, claimants in reverse_index.items():
        if len(claimants) == 1:
            assignments[s2s3_id] = claimants[0][0]  # Uncontested
        else:
            # Sort by score descending
            claimants.sort(key=lambda x: x[1], reverse=True)
            best_s1, best_score = claimants[0]
            second_score = claimants[1][1]

            if (best_score - second_score) >= margin_delta:
                assignments[s2s3_id] = best_s1  # Clear winner
            else:
                assignments[s2s3_id] = None  # Too ambiguous, skip

    # Rebuild predictions from assignments
    predictions = {s1_id: set() for s1_id in raw_predictions}
    for s2s3_id, s1_id in assignments.items():
        if s1_id is not None:
            predictions[s1_id].add(s2s3_id)

    n_contested = sum(1 for v in reverse_index.values() if len(v) > 1)
    n_rejected = sum(1 for v in assignments.values() if v is None)
    log.info(f"  Global dedup: {n_contested:,} contested S2/S3 entities, "
             f"{n_rejected:,} rejected (insufficient margin)")

    return predictions


@timed
def optimize_threshold(
    df_scored: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: np.ndarray,
) -> float:
    """
    Grid-search the optimal threshold τ maximizing macro F₀.₅ on validation.
    """
    from src.utils import macro_f05

    best_f05 = 0.0
    best_tau = 0.5

    thresholds = np.arange(
        config.THRESHOLD_SEARCH_MIN,
        config.THRESHOLD_SEARCH_MAX + config.THRESHOLD_SEARCH_STEP,
        config.THRESHOLD_SEARCH_STEP,
    )

    for tau in tqdm(thresholds, desc="Threshold search"):
        predictions = apply_thresholding(
            df_scored, tau, all_s1_ids,
            use_margin=True, global_dedup=True,
        )
        score = macro_f05(predictions, ground_truth)

        if score > best_f05:
            best_f05 = score
            best_tau = tau

    log.info(f"  Optimal threshold: τ = {best_tau:.3f}, F₀.₅ = {best_f05:.4f}")

    # Save
    with open(config.MODELS_DIR / "optimal_threshold.pkl", "wb") as f:
        pickle.dump(best_tau, f)

    return best_tau


# ─────────────────────────────────────────────────────────────
# OUTPUT GENERATION
# ─────────────────────────────────────────────────────────────

def format_output(predictions: Dict[str, Set[str]],
                  all_s1_ids: np.ndarray) -> pd.DataFrame:
    """
    Format predictions as the submission DataFrame.

    Columns: source1_entity_id, matched_entity_ids
    Singletons have empty string in matched_entity_ids.
    """
    rows = []
    for s1_id in all_s1_ids:
        matches = predictions.get(s1_id, set())
        matched_str = ",".join(sorted(matches)) if matches else ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})

    return pd.DataFrame(rows)


def format_candidate_pairs(candidates: Dict[str, Set[str]],
                           all_s1_ids: np.ndarray) -> pd.DataFrame:
    """
    Format candidate pairs for submission.

    Columns: source1_entity_id, candidate_entity_ids
    """
    rows = []
    for s1_id in all_s1_ids:
        cands = candidates.get(s1_id, set())
        cand_str = ",".join(sorted(cands)) if cands else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# FULL INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────

@timed
def run_inference(split: str = "test"):
    """
    End-to-end inference pipeline on test data.

    1. Load blocking candidates (or run blocking)
    2. Extract features
    3. LightGBM cascade filter (50 → 5)
    4. Cross-encoder re-ranking (5 → scored)
    5. Ensemble + threshold
    6. Output matching_results.tsv + candidate_pairs.tsv
    """

    # ── Load data ───────────────────────────────────────
    log.info("Loading preprocessed test data...")
    df_s1 = load_preprocessed(split, "s1")
    df_s2 = load_preprocessed(split, "s2")
    df_s3 = load_preprocessed(split, "s3")
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    all_s1_ids = df_s1["entity_id"].values
    log.info(f"S1 entities: {len(all_s1_ids):,}")
    log.info(f"Targets: {len(df_targets):,}")

    # ── Load / generate blocking candidates ─────────────
    cand_path = config.BLOCKING_DIR / f"{split}_candidates.pkl"
    if cand_path.exists():
        log.info("Loading cached blocking candidates...")
        candidates = load_candidates(split)
    else:
        log.info("Running blocking...")
        candidates = run_blocking(split, save=True)

    # Save candidate_pairs.tsv
    df_cand = format_candidate_pairs(candidates, all_s1_ids)
    save_tsv(df_cand, config.OUTPUT_DIR / "candidate_pairs.tsv")
    del df_cand

    # ── Extract features ────────────────────────────────
    features_path = config.FEATURES_DIR / f"{split}_features.parquet"
    if features_path.exists():
        log.info(f"Loading cached features from {features_path.name}...")
        df_features = pd.read_parquet(features_path)
    else:
        log.info("Loading embeddings for features...")
        emb_s1, emb_targets, s1_map, t_map = load_embeddings_for_features(
            split, df_s1, df_targets
        )

        log.info("Extracting features for all candidates...")
        extract_features_for_pairs(
            candidates, df_s1, df_targets,
            emb_s1, emb_targets, s1_map, t_map,
            output_path=features_path,
        )
        log.info(f"Features saved to {features_path.name}")
        df_features = pd.read_parquet(features_path)

        del emb_s1, emb_targets, s1_map, t_map
        gc.collect()

    if len(df_features) == 0:
        log.warning("No features extracted! Outputting all singletons.")
        predictions = {s1_id: set() for s1_id in all_s1_ids}
        df_output = format_output(predictions, all_s1_ids)
        save_tsv(df_output, config.OUTPUT_DIR / "matching_results.tsv")
        return

    # ── LightGBM cascade ────────────────────────────────
    log.info("Running LightGBM cascade filter...")
    lgbm_model = load_lgbm_model()
    df_filtered = lgbm_cascade_filter(df_features, lgbm_model)
    del df_features
    gc.collect()

    # ── Cross-encoder re-ranking ────────────────────────
    ce_dir = config.MODELS_DIR / "cross_encoder"
    if ce_dir.exists():
        log.info("Running cross-encoder re-ranking...")
        scorer = CrossEncoderScorer(str(ce_dir))
        df_scored = cross_encoder_rerank(df_filtered, df_s1, df_targets, scorer)
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        log.warning("Cross-encoder not found. Using LightGBM scores only.")
        df_scored = df_filtered.copy()
        df_scored["ce_prob"] = df_scored["lgbm_prob"]  # Fallback

    # ── Load optimal threshold ──────────────────────────
    threshold_path = config.MODELS_DIR / "optimal_threshold.pkl"
    if threshold_path.exists():
        with open(threshold_path, "rb") as f:
            threshold = pickle.load(f)
        log.info(f"Using saved threshold: τ = {threshold:.3f}")
    else:
        threshold = 0.75  # Conservative default
        log.warning(f"No saved threshold found. Using default: τ = {threshold:.3f}")

    # ── Apply thresholding ──────────────────────────────
    log.info("Applying margin-based thresholding + global dedup...")
    predictions = apply_thresholding(
        df_scored, threshold, all_s1_ids,
        use_margin=True,
        global_dedup=True,
    )

    # Stats
    n_matched = sum(1 for v in predictions.values() if len(v) > 0)
    n_singletons = sum(1 for v in predictions.values() if len(v) == 0)
    total_links = sum(len(v) for v in predictions.values())
    log.info(f"  Matched: {n_matched:,}, Singletons: {n_singletons:,}, "
             f"Total links: {total_links:,}")

    # ── Save outputs ────────────────────────────────────
    df_output = format_output(predictions, all_s1_ids)
    save_tsv(df_output, config.OUTPUT_DIR / "matching_results.tsv")

    log.info("═══ Inference complete! ═══")
    log.info(f"  matching_results.tsv: {config.OUTPUT_DIR / 'matching_results.tsv'}")
    log.info(f"  candidate_pairs.tsv:  {config.OUTPUT_DIR / 'candidate_pairs.tsv'}")


# ─────────────────────────────────────────────────────────────
# VALIDATION INFERENCE (for threshold optimization)
# ─────────────────────────────────────────────────────────────

@timed
def run_validation_inference():
    """
    Run inference on validation split of training data to optimize threshold.

    Uses the saved val features from LightGBM training.
    """
    from src.utils import load_ground_truth, evaluate_predictions

    val_path = config.FEATURES_DIR / "val_features.parquet"
    if not val_path.exists():
        log.error("Run LightGBM training first to generate validation features")
        return

    df_val = pd.read_parquet(val_path)
    gt = load_ground_truth()

    # Load models
    lgbm_model = load_lgbm_model()

    # LightGBM cascade
    df_filtered = lgbm_cascade_filter(df_val, lgbm_model)

    # Cross-encoder (if available)
    ce_dir = config.MODELS_DIR / "cross_encoder"
    if ce_dir.exists():
        log.info("Loading preprocessed training data for text...")
        df_s1 = load_preprocessed("train", "s1")
        df_s2 = load_preprocessed("train", "s2")
        df_s3 = load_preprocessed("train", "s3")
        df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

        scorer = CrossEncoderScorer(str(ce_dir))
        df_scored = cross_encoder_rerank(df_filtered, df_s1, df_targets, scorer)
        del scorer
    else:
        df_scored = df_filtered.copy()
        df_scored["ce_prob"] = df_scored["lgbm_prob"]

    # Get val S1 IDs
    val_s1_ids = df_scored["s1_id"].unique()
    val_gt = {s1: gt.get(s1, set()) for s1 in val_s1_ids}

    # Optimize threshold
    threshold = optimize_threshold(df_scored, val_gt, val_s1_ids)

    # Final evaluation
    predictions = apply_thresholding(
        df_scored, threshold, val_s1_ids,
        use_margin=True, global_dedup=True,
    )
    results = evaluate_predictions(predictions, val_gt)
    log.info("  Validation results:")
    for k, v in results.items():
        log.info(f"    {k}: {v}")

    return threshold


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        run_validation_inference()
    else:
        run_inference("test")
