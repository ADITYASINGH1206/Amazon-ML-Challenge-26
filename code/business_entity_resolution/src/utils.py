"""
utils.py — Shared I/O helpers, timing, memory monitoring, and F0.5 computation.
"""

import time
import functools
import logging
try:
    import psutil
except ImportError:
    psutil = None
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional

from src import config

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("entity_resolution")


# ─────────────────────────────────────────────────────────────
# TIMING DECORATOR
# ─────────────────────────────────────────────────────────────

def timed(func):
    """Decorator that logs wall-clock time for a function call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log.info(f"▶ {func.__name__} started")
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        m, s = divmod(elapsed, 60)
        log.info(f"✓ {func.__name__} finished in {int(m)}m {s:.1f}s")
        return result
    return wrapper


def mem_usage_gb():
    """Return current process RSS in GB."""
    if psutil is None:
        return 0.0
    return psutil.Process().memory_info().rss / (1024 ** 3)


def log_memory():
    """Log current memory usage."""
    if psutil is not None:
        log.info(f"  Memory: {mem_usage_gb():.2f} GB")


# ─────────────────────────────────────────────────────────────
# DATA I/O
# ─────────────────────────────────────────────────────────────

def load_source(path: Path, usecols=None) -> pd.DataFrame:
    """Load a tab-separated source file."""
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=usecols,
                     na_values=[""], keep_default_na=True)
    log.info(f"  Loaded {path.name}: {len(df):,} rows")
    return df


def load_ground_truth(path: Path = None) -> Dict[str, Set[str]]:
    """
    Load ground truth as {source1_entity_id: set(matched_entity_ids)}.
    Singletons map to an empty set.
    """
    if path is None:
        path = config.TRAIN_GT
    df = pd.read_csv(path, sep="\t", dtype=str)
    gt = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if pd.isna(matched) or matched.strip() == "":
            gt[s1_id] = set()
        else:
            gt[s1_id] = set(matched.strip().split(","))
    log.info(f"  Ground truth loaded: {len(gt):,} S1 entities, "
             f"{sum(1 for v in gt.values() if v):,} with matches")
    return gt


def save_tsv(df: pd.DataFrame, path: Path):
    """Save DataFrame as tab-separated file with no quoting."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    log.info(f"  Saved {path.name}: {len(df):,} rows")


# ─────────────────────────────────────────────────────────────
# F0.5 SCORING
# ─────────────────────────────────────────────────────────────

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def per_entity_f05(predicted: Set[str], truth: Set[str]) -> float:
    """
    Compute F0.5 for a single S1 entity.

    Singleton rule: if truth is empty, score is 1.0 if predicted is also empty,
    else 0.0.
    """
    if len(truth) == 0:
        return 1.0 if len(predicted) == 0 else 0.0
    if len(predicted) == 0:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted) if len(predicted) > 0 else 0.0
    recall = tp / len(truth) if len(truth) > 0 else 0.0
    return f_beta(precision, recall, beta=0.5)


def macro_f05(predictions: Dict[str, Set[str]],
              ground_truth: Dict[str, Set[str]]) -> float:
    """
    Compute macro-averaged F0.5 across all S1 entities.

    Both dicts must have the same set of S1 keys.
    """
    scores = []
    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        truth = ground_truth[s1_id]
        scores.append(per_entity_f05(pred, truth))
    return float(np.mean(scores))


def evaluate_predictions(predictions: Dict[str, Set[str]],
                         ground_truth: Dict[str, Set[str]]) -> Dict:
    """
    Comprehensive evaluation with breakdown statistics.
    """
    scores = []
    n_singleton_correct = 0
    n_singleton_wrong = 0
    n_matched_scores = []

    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        truth = ground_truth[s1_id]
        score = per_entity_f05(pred, truth)
        scores.append(score)

        if len(truth) == 0:
            if len(pred) == 0:
                n_singleton_correct += 1
            else:
                n_singleton_wrong += 1
        else:
            n_matched_scores.append(score)

    n_singletons = sum(1 for v in ground_truth.values() if len(v) == 0)
    n_predicted_empty = sum(1 for v in predictions.values() if len(v) == 0)

    total_pred = sum(len(v) for v in predictions.values())
    total_truth = sum(len(v) for v in ground_truth.values())

    return {
        "macro_f05": float(np.mean(scores)),
        "n_entities": len(ground_truth),
        "n_singletons_truth": n_singletons,
        "n_singletons_correct": n_singleton_correct,
        "n_singletons_wrong": n_singleton_wrong,
        "n_predicted_empty": n_predicted_empty,
        "mean_f05_matched": float(np.mean(n_matched_scores)) if n_matched_scores else 0.0,
        "total_predicted_links": total_pred,
        "total_true_links": total_truth,
    }
