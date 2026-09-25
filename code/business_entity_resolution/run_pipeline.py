"""
run_pipeline.py — End-to-end orchestrator for the Entity Resolution pipeline.

Usage:
    python run_pipeline.py all          # Run everything end-to-end
    python run_pipeline.py preprocess   # Stage 0 only
    python run_pipeline.py block        # Stage 1 only
    python run_pipeline.py features     # Stage 2 only
    python run_pipeline.py train_lgbm   # Stage 3a only
    python run_pipeline.py train_ce     # Stage 3b only
    python run_pipeline.py validate     # Threshold optimization on val split
    python run_pipeline.py infer        # Stage 3+4 on test data
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import log, timed


@timed
def stage_preprocess():
    """Stage 0: Preprocess and normalize all data."""
    from src.preprocess import run_preprocessing
    run_preprocessing("train")
    run_preprocessing("test")


@timed
def stage_blocking():
    """Stage 1: Generate candidate pairs via multi-strategy blocking."""
    from src.blocking import run_blocking, evaluate_blocking_recall
    from src.utils import load_ground_truth

    candidates = run_blocking("train")

    # Evaluate blocking recall on training data
    gt = load_ground_truth()
    evaluate_blocking_recall(candidates, gt)

    # Also run blocking on test data
    run_blocking("test")


@timed
def stage_features():
    """Stage 2: Extract features for all candidate pairs."""
    import pandas as pd
    from src.preprocess import load_preprocessed
    from src.blocking import load_candidates
    from src.features import extract_features_for_pairs, load_embeddings_for_features
    from src import config

    # Training features
    log.info("═══ Extracting TRAINING features ═══")
    candidates = load_candidates("train")
    df_s1 = load_preprocessed("train", "s1")
    df_s2 = load_preprocessed("train", "s2")
    df_s3 = load_preprocessed("train", "s3")
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    emb_s1, emb_targets, s1_map, t_map = load_embeddings_for_features(
        "train", df_s1, df_targets
    )

    df_features = extract_features_for_pairs(
        candidates, df_s1, df_targets,
        emb_s1, emb_targets, s1_map, t_map,
    )
    df_features.to_parquet(config.FEATURES_DIR / "train_features.parquet", index=False)
    log.info(f"Training features saved: {len(df_features):,} pairs")


@timed
def stage_train_lgbm():
    """Stage 3a: Train LightGBM pairwise classifier."""
    from src.train_lgbm import run_lgbm_training
    run_lgbm_training()


@timed
def stage_train_cross_encoder():
    """Stage 3b: Fine-tune cross-encoder on hard negatives."""
    from src.train_cross_encoder import run_cross_encoder_training
    run_cross_encoder_training()


@timed
def stage_validate():
    """Optimize threshold on validation split."""
    from src.inference import run_validation_inference
    run_validation_inference()


@timed
def stage_infer():
    """Stages 3+4: Full inference on test data."""
    from src.inference import run_inference
    run_inference("test")


@timed
def run_all():
    """Run the complete pipeline end-to-end."""
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  Amazon ML Challenge 2026                   ║")
    log.info("║  Business Entity Resolution Pipeline        ║")
    log.info("╚══════════════════════════════════════════════╝")

    t0 = time.perf_counter()

    log.info("\n" + "="*60)
    log.info("STAGE 0: PREPROCESSING")
    log.info("="*60)
    stage_preprocess()

    log.info("\n" + "="*60)
    log.info("STAGE 1: BLOCKING (Candidate Generation)")
    log.info("="*60)
    stage_blocking()

    log.info("\n" + "="*60)
    log.info("STAGE 2: FEATURE ENGINEERING")
    log.info("="*60)
    stage_features()

    log.info("\n" + "="*60)
    log.info("STAGE 3a: LIGHTGBM TRAINING")
    log.info("="*60)
    stage_train_lgbm()

    log.info("\n" + "="*60)
    log.info("STAGE 3b: CROSS-ENCODER TRAINING")
    log.info("="*60)
    stage_train_cross_encoder()

    log.info("\n" + "="*60)
    log.info("THRESHOLD OPTIMIZATION")
    log.info("="*60)
    stage_validate()

    log.info("\n" + "="*60)
    log.info("STAGE 4: TEST INFERENCE")
    log.info("="*60)
    stage_infer()

    elapsed = time.perf_counter() - t0
    h, remainder = divmod(elapsed, 3600)
    m, s = divmod(remainder, 60)
    log.info(f"\n{'='*60}")
    log.info(f"PIPELINE COMPLETE — Total time: {int(h)}h {int(m)}m {s:.0f}s")
    log.info(f"{'='*60}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable stages:")
        print("  all           Run everything end-to-end")
        print("  preprocess    Stage 0: Text normalization")
        print("  block         Stage 1: Blocking / candidate generation")
        print("  features      Stage 2: Feature extraction")
        print("  train_lgbm    Stage 3a: LightGBM training")
        print("  train_ce      Stage 3b: Cross-encoder fine-tuning")
        print("  validate      Threshold optimization on validation")
        print("  infer         Full inference on test data")
        sys.exit(0)

    stage = sys.argv[1].lower()

    dispatch = {
        "all": run_all,
        "preprocess": stage_preprocess,
        "block": stage_blocking,
        "features": stage_features,
        "train_lgbm": stage_train_lgbm,
        "train_ce": stage_train_cross_encoder,
        "validate": stage_validate,
        "infer": stage_infer,
    }

    if stage not in dispatch:
        print(f"Unknown stage: {stage}")
        print(f"Available stages: {', '.join(dispatch.keys())}")
        sys.exit(1)

    dispatch[stage]()


if __name__ == "__main__":
    main()
