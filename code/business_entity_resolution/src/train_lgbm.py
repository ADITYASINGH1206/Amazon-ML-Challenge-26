"""
train_lgbm.py — LightGBM pairwise classifier training (Stage 3a).

Trains a gradient boosted tree on the 40 features extracted in Stage 2.
Used as a fast cascade filter: scores all ~50 candidates per entity,
keeps top-K for the expensive cross-encoder re-ranking.
"""

import gc
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Set, Tuple
from sklearn.model_selection import train_test_split

from src import config
from src.utils import log, timed, load_ground_truth, macro_f05, evaluate_predictions, log_memory
from src.features import get_feature_columns, FEATURE_NAMES


@timed
def prepare_training_data(
    df_features: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Label candidate pairs using ground truth and split into train/val.

    Labels:
    - 1: (s1_id, s2s3_id) is a true match in ground truth
    - 0: (s1_id, s2s3_id) is NOT a true match

    Splits by S1 entity (not by pair) to avoid data leakage.
    """
    log.info(f"Labeling {len(df_features):,} candidate pairs...")

    # Label pairs
    labels = []
    for _, row in df_features.iterrows():
        s1_id = row["s1_id"]
        s2s3_id = row["s2s3_id"]
        true_matches = ground_truth.get(s1_id, set())
        labels.append(1 if s2s3_id in true_matches else 0)

    df_features = df_features.copy()
    df_features["label"] = labels

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    log.info(f"  Positives: {n_pos:,} ({100*n_pos/len(labels):.1f}%)")
    log.info(f"  Negatives: {n_neg:,} ({100*n_neg/len(labels):.1f}%)")

    # Split by S1 entity to avoid leakage
    all_s1_ids = df_features["s1_id"].unique()
    train_s1, val_s1 = train_test_split(
        all_s1_ids,
        test_size=config.VAL_FRACTION,
        random_state=config.RANDOM_SEED,
    )
    train_s1_set = set(train_s1)
    val_s1_set = set(val_s1)

    df_train = df_features[df_features["s1_id"].isin(train_s1_set)].reset_index(drop=True)
    df_val = df_features[df_features["s1_id"].isin(val_s1_set)].reset_index(drop=True)

    log.info(f"  Train: {len(df_train):,} pairs ({df_train['label'].sum():,} pos)")
    log.info(f"  Val:   {len(df_val):,} pairs ({df_val['label'].sum():,} pos)")

    return df_train, df_val


@timed
def train_lightgbm(df_train: pd.DataFrame, df_val: pd.DataFrame) -> lgb.LGBMClassifier:
    """
    Train LightGBM binary classifier on candidate pair features.
    """
    feature_cols = get_feature_columns()

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["label"].values

    # Dynamic positive weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    log.info(f"  scale_pos_weight: {scale_pos:.2f}")

    params = config.LGBM_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos

    # Extract early_stopping_rounds from params (sklearn API changed)
    early_stopping = params.pop("early_stopping_rounds", 100)

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    log.info(f"  Best iteration: {model.best_iteration_}")

    # Feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    log.info("  Top 15 features:")
    for i in sorted_idx[:15]:
        log.info(f"    {FEATURE_NAMES[i]:35s}  {importance[i]:>8.0f}")

    # Save model
    model_path = config.MODELS_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"  Model saved to {model_path.name}")

    return model


@timed
def evaluate_lgbm_on_val(
    model: lgb.LGBMClassifier,
    df_val: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
) -> float:
    """
    Evaluate LightGBM predictions on validation set using macro F0.5.

    Grid-searches the optimal threshold τ on the validation set.
    """
    feature_cols = get_feature_columns()
    X_val = df_val[feature_cols].values

    # Get prediction probabilities
    probs = model.predict_proba(X_val)[:, 1]
    df_val = df_val.copy()
    df_val["lgbm_prob"] = probs

    # Grid search threshold
    best_f05 = 0.0
    best_threshold = 0.5

    thresholds = np.arange(
        config.THRESHOLD_SEARCH_MIN,
        config.THRESHOLD_SEARCH_MAX + config.THRESHOLD_SEARCH_STEP,
        config.THRESHOLD_SEARCH_STEP,
    )

    val_s1_ids = df_val["s1_id"].unique()
    val_gt = {s1: ground_truth.get(s1, set()) for s1 in val_s1_ids}

    for tau in thresholds:
        predictions = {}
        for s1_id in val_s1_ids:
            mask = (df_val["s1_id"] == s1_id) & (df_val["lgbm_prob"] >= tau)
            matched = set(df_val.loc[mask, "s2s3_id"].values)
            predictions[s1_id] = matched

        score = macro_f05(predictions, val_gt)
        if score > best_f05:
            best_f05 = score
            best_threshold = tau

    log.info(f"  Best LightGBM threshold: {best_threshold:.3f}")
    log.info(f"  Best LightGBM macro F0.5: {best_f05:.4f}")

    # Detailed evaluation at best threshold
    predictions = {}
    for s1_id in val_s1_ids:
        mask = (df_val["s1_id"] == s1_id) & (df_val["lgbm_prob"] >= best_threshold)
        matched = set(df_val.loc[mask, "s2s3_id"].values)
        predictions[s1_id] = matched

    eval_results = evaluate_predictions(predictions, val_gt)
    for k, v in eval_results.items():
        log.info(f"    {k}: {v}")

    # Save threshold
    with open(config.MODELS_DIR / "lgbm_threshold.pkl", "wb") as f:
        pickle.dump(best_threshold, f)

    return best_f05


def load_lgbm_model() -> lgb.LGBMClassifier:
    """Load trained LightGBM model."""
    model_path = config.MODELS_DIR / "lgbm_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


@timed
def run_lgbm_training():
    """Full LightGBM training pipeline."""
    # Load features
    features_path = config.FEATURES_DIR / "train_features.parquet"
    log.info(f"Loading features from {features_path.name}...")
    df_features = pd.read_parquet(features_path)
    log.info(f"  Loaded {len(df_features):,} pairs")

    # Load ground truth
    gt = load_ground_truth()

    # Prepare train/val split
    df_train, df_val = prepare_training_data(df_features, gt)

    # Save val split info for cross-encoder training
    df_val.to_parquet(config.FEATURES_DIR / "val_features.parquet", index=False)
    df_train.to_parquet(config.FEATURES_DIR / "train_features_labeled.parquet", index=False)

    # Train
    model = train_lightgbm(df_train, df_val)

    # Evaluate
    evaluate_lgbm_on_val(model, df_val, gt)

    return model


if __name__ == "__main__":
    run_lgbm_training()
