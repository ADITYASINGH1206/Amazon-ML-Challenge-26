"""
config.py — Central configuration for all paths, hyperparameters, and model settings.

All tunable knobs live here so nothing is scattered across files.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Auto-detect project root (two levels up from src/)
_THIS_DIR = Path(__file__).resolve().parent                       # src/
CODE_DIR = _THIS_DIR.parent                                       # code/business_entity_resolution/
PROJECT_ROOT = CODE_DIR.parent.parent                             # Amazon-ML-Challenge-26/

# Override via environment variable if running on a different machine
STUDENT_RESOURCE_DIR = Path(os.environ.get(
    "STUDENT_RESOURCE_DIR",
    PROJECT_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource",
))

TRAIN_DIR  = STUDENT_RESOURCE_DIR / "dataset" / "train"
TEST_DIR   = STUDENT_RESOURCE_DIR / "dataset" / "test"

# Input files
TRAIN_S1   = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2   = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3   = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT   = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1    = TEST_DIR / "test_source1.tsv"
TEST_S2    = TEST_DIR / "test_source2.tsv"
TEST_S3    = TEST_DIR / "test_source3.tsv"

# Working directories (intermediate artifacts)
WORK_DIR         = CODE_DIR / "workdir"
PREPROCESSED_DIR = WORK_DIR / "preprocessed"
BLOCKING_DIR     = WORK_DIR / "blocking"
FEATURES_DIR     = WORK_DIR / "features"
MODELS_DIR       = WORK_DIR / "models"
EMBEDDINGS_DIR   = WORK_DIR / "embeddings"

# Final output
OUTPUT_DIR = STUDENT_RESOURCE_DIR / "output"

# Ensure all working directories exist
for d in [WORK_DIR, PREPROCESSED_DIR, BLOCKING_DIR, FEATURES_DIR,
          MODELS_DIR, EMBEDDINGS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Sentence-Transformer for dense blocking & cosine features
# Multilingual model — critical for French zero-shot generalization
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM        = 384

# Cross-Encoder for Stage 3b re-ranking
# Multilingual DeBERTa — handles EN/HI/FR
CROSS_ENCODER_MODEL  = "microsoft/mdeberta-v3-base"
CROSS_ENCODER_MAX_LEN = 256


# ─────────────────────────────────────────────────────────────
# BLOCKING HYPERPARAMETERS (Stage 1)
# ─────────────────────────────────────────────────────────────

# Maximum candidates per S1 entity from blocking
MAX_CANDIDATES_PER_ENTITY = 50

# FAISS: number of nearest neighbors to retrieve
FAISS_TOP_K = 30

# FAISS IVF: number of Voronoi cells (rule of thumb: sqrt(n))
FAISS_NLIST = 4096

# FAISS IVF: number of cells to probe at query time
FAISS_NPROBE = 64

# Inverted index: skip tokens appearing in more than this many docs
TOKEN_MAX_DF = 50000

# Minimum number of shared blocking tokens to consider a candidate
MIN_SHARED_TOKENS = 1


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (Stage 2)
# ─────────────────────────────────────────────────────────────

# TF-IDF settings for character n-gram features
TFIDF_ANALYZER    = "char_wb"
TFIDF_NGRAM_RANGE = (3, 4)
TFIDF_MAX_FEATURES = 100_000


# ─────────────────────────────────────────────────────────────
# LIGHTGBM HYPERPARAMETERS (Stage 3a)
# ─────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.04,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,   # will be overridden dynamically
    "n_jobs": -1,
    "verbose": -1,
    "early_stopping_rounds": 150,
}

# LightGBM cascade: keep top-K candidates per S1 for cross-encoder
LGBM_CASCADE_TOP_K = 5


# ─────────────────────────────────────────────────────────────
# CROSS-ENCODER TRAINING (Stage 3b)
# ─────────────────────────────────────────────────────────────

CE_BATCH_SIZE      = 64
CE_LEARNING_RATE   = 2e-5
CE_EPOCHS          = 2
CE_WARMUP_RATIO    = 0.1
CE_MAX_TRAIN_PAIRS = 500_000   # cap training pairs for speed
CE_FP16            = True


# ─────────────────────────────────────────────────────────────
# ENSEMBLE & THRESHOLDING (Stage 4)
# ─────────────────────────────────────────────────────────────

# Ensemble weights: LightGBM vs Cross-Encoder
ENSEMBLE_WEIGHT_LGBM = 0.35
ENSEMBLE_WEIGHT_CE   = 0.65

# Threshold search range for τ
THRESHOLD_SEARCH_MIN  = 0.60
THRESHOLD_SEARCH_MAX  = 0.98
THRESHOLD_SEARCH_STEP = 0.005

# Margin-based thresholding: minimum gap between best match score
# and next-best *conflicting* candidate score to accept a match
MARGIN_DELTA = 0.08

# Global deduplication: each S2/S3 entity assigned to at most one S1
GLOBAL_DEDUP = True


# ─────────────────────────────────────────────────────────────
# VALIDATION SPLIT
# ─────────────────────────────────────────────────────────────

VAL_FRACTION = 0.15
RANDOM_SEED  = 42


# ─────────────────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────────────────

# Batch sizes for various processing steps
EMBEDDING_BATCH_SIZE   = 512
FEATURE_BATCH_SIZE     = 50_000
INFERENCE_BATCH_SIZE   = 256
