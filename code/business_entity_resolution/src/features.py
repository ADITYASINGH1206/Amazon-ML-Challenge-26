"""
features.py — Feature engineering for candidate pairs (Stage 2).

Extracts 44 features per (S1, S2/S3) candidate pair:
- String similarity (name & address): Jaro-Winkler, Levenshtein, token ratios
- Component matching: street number, postal code, city
- Semantic: TF-IDF cosine, dense embedding cosine
- IDF-weighted token overlap
- Meta: source indicator, length ratios
"""

import gc
import math
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Set, List, Optional, Iterable
from pathlib import Path
from tqdm import tqdm

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from src import config
from src.utils import log, timed, log_memory
from src.preprocess import load_preprocessed


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _safe_str(x) -> str:
    """Convert to string, handling NaN."""
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


def _jaccard_tokens(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    if not a or not b:
        return 0.0
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not a or not b:
        return 0.0
    if len(a) < n or len(b) < n:
        return 0.0
    ngrams_a = set(a[i:i+n] for i in range(len(a) - n + 1))
    ngrams_b = set(b[i:i+n] for i in range(len(b) - n + 1))
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union if union > 0 else 0.0


def _lcs_ratio(a: str, b: str) -> float:
    """Longest Common Subsequence ratio."""
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    if m > 500 or n > 500:
        # Truncate very long strings for performance
        a, b = a[:500], b[:500]
        m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
            else:
                dp[i % 2][j] = max(dp[(i-1) % 2][j], dp[i % 2][j-1])
    lcs_len = dp[m % 2][n]
    return 2.0 * lcs_len / (m + n) if (m + n) > 0 else 0.0


def _numeric_overlap(a: str, b: str) -> float:
    """Fraction of shared numeric tokens."""
    import re
    nums_a = set(re.findall(r"\b\d+\b", a))
    nums_b = set(re.findall(r"\b\d+\b", b))
    if not nums_a or not nums_b:
        return 0.0
    return len(nums_a & nums_b) / len(nums_a | nums_b)


# ─────────────────────────────────────────────────────────────
# IDF COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_idf(documents: Iterable[str]) -> Dict[str, float]:
    """Compute IDF scores for tokens across a corpus."""
    n_docs = 0
    doc_freq = Counter()
    for doc in documents:
        n_docs += 1
        tokens = set(doc.split()) if isinstance(doc, str) else set()
        doc_freq.update(tokens)

    idf = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((n_docs + 1) / (df + 1)) + 1  # smoothed IDF
    return idf


def _idf_weighted_overlap(a: str, b: str, idf: Dict[str, float]) -> float:
    """IDF-weighted token overlap score."""
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    shared = tokens_a & tokens_b
    if not shared:
        return 0.0

    score = sum(idf.get(t, 1.0) for t in shared)
    max_possible = max(
        sum(idf.get(t, 1.0) for t in tokens_a),
        sum(idf.get(t, 1.0) for t in tokens_b),
    )
    return score / max_possible if max_possible > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Name string features (12)
    "jaro_winkler_name",
    "levenshtein_ratio_name",
    "token_sort_ratio_name",
    "token_set_ratio_name",
    "partial_ratio_name",
    "lcs_ratio_name",
    "jaccard_3gram_name",
    "jaccard_tokens_name",
    "exact_match_name",
    "prefix_match_5_name",
    "name_length_diff",
    "name_token_count_diff",
    # Address string features (13)
    "jaro_winkler_addr",
    "levenshtein_ratio_addr",
    "token_sort_ratio_addr",
    "token_set_ratio_addr",
    "partial_ratio_addr",
    "lcs_ratio_addr",
    "jaccard_3gram_addr",
    "jaccard_tokens_addr",
    "exact_match_addr",
    "addr_length_diff",
    "addr_token_count_diff",
    "addr_missing_either",
    "addr_missing_both",
    # Component features (5)
    "street_number_match",
    "postal_code_match",
    "numeric_overlap",
    "postal_code_edit_dist",
    "country_match",
    # Combined features (5)
    "combined_token_sort_ratio",
    "combined_jaro_winkler",
    "idf_overlap_name",
    "idf_overlap_addr",
    "idf_overlap_combined",
    # Embedding feature (1)
    "embedding_cosine",
    # Meta features (4)
    "source_indicator",
    "name_length_ratio",
    "addr_length_ratio",
    "name_addr_length_ratio",
]

assert len(FEATURE_NAMES) == 40, f"Expected 40 features, got {len(FEATURE_NAMES)}"


def extract_pair_features(
    name_s1: str, addr_s1: str, name_addr_s1: str,
    street_num_s1: str, postal_s1: str, country_s1: str,
    name_s2: str, addr_s2: str, name_addr_s2: str,
    street_num_s2: str, postal_s2: str, country_s2: str,
    entity_id_s2: str,
    idf_name: Dict[str, float],
    idf_addr: Dict[str, float],
    idf_combined: Dict[str, float],
    emb_cosine: float = 0.0,
) -> np.ndarray:
    """
    Extract all features for a single (S1, S2/S3) candidate pair.

    Returns: 1D array of shape (40,)
    """
    feats = np.zeros(40, dtype=np.float32)

    n1 = _safe_str(name_s1)
    n2 = _safe_str(name_s2)
    a1 = _safe_str(addr_s1)
    a2 = _safe_str(addr_s2)
    na1 = _safe_str(name_addr_s1)
    na2 = _safe_str(name_addr_s2)

    # ── Name features ───────────────────────────────────
    if n1 and n2:
        feats[0] = JaroWinkler.normalized_similarity(n1, n2)
        feats[1] = Levenshtein.normalized_similarity(n1, n2)
        feats[2] = fuzz.token_sort_ratio(n1, n2) / 100.0
        feats[3] = fuzz.token_set_ratio(n1, n2) / 100.0
        feats[4] = fuzz.partial_ratio(n1, n2) / 100.0
        feats[5] = _lcs_ratio(n1, n2)
        feats[6] = _jaccard_ngrams(n1, n2, 3)
        feats[7] = _jaccard_tokens(n1, n2)
        feats[8] = 1.0 if n1 == n2 else 0.0
        feats[9] = 1.0 if n1[:5] == n2[:5] and len(n1) >= 5 and len(n2) >= 5 else 0.0
    feats[10] = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
    feats[11] = abs(len(n1.split()) - len(n2.split()))

    # ── Address features ────────────────────────────────
    a1_present = len(a1) > 0
    a2_present = len(a2) > 0

    if a1_present and a2_present:
        feats[12] = JaroWinkler.normalized_similarity(a1, a2)
        feats[13] = Levenshtein.normalized_similarity(a1, a2)
        feats[14] = fuzz.token_sort_ratio(a1, a2) / 100.0
        feats[15] = fuzz.token_set_ratio(a1, a2) / 100.0
        feats[16] = fuzz.partial_ratio(a1, a2) / 100.0
        feats[17] = _lcs_ratio(a1, a2)
        feats[18] = _jaccard_ngrams(a1, a2, 3)
        feats[19] = _jaccard_tokens(a1, a2)
        feats[20] = 1.0 if a1 == a2 else 0.0
    feats[21] = abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1) if (a1_present or a2_present) else 0.0
    feats[22] = abs(len(a1.split()) - len(a2.split())) if (a1_present and a2_present) else 0.0
    feats[23] = 1.0 if not a1_present or not a2_present else 0.0
    feats[24] = 1.0 if not a1_present and not a2_present else 0.0

    # ── Component features ──────────────────────────────
    sn1 = _safe_str(street_num_s1)
    sn2 = _safe_str(street_num_s2)
    pc1 = _safe_str(postal_s1)
    pc2 = _safe_str(postal_s2)

    feats[25] = 1.0 if sn1 and sn2 and sn1 == sn2 else (0.0 if sn1 and sn2 else -1.0)
    feats[26] = 1.0 if pc1 and pc2 and pc1 == pc2 else (0.0 if pc1 and pc2 else -1.0)
    feats[27] = _numeric_overlap(a1, a2) if a1_present and a2_present else 0.0

    if pc1 and pc2:
        feats[28] = Levenshtein.normalized_similarity(pc1, pc2)
    else:
        feats[28] = -1.0

    feats[29] = 1.0 if country_s1 == country_s2 else 0.0

    # ── Combined features ───────────────────────────────
    if na1 and na2:
        feats[30] = fuzz.token_sort_ratio(na1, na2) / 100.0
        feats[31] = JaroWinkler.normalized_similarity(na1, na2)

    feats[32] = _idf_weighted_overlap(n1, n2, idf_name) if n1 and n2 else 0.0
    feats[33] = _idf_weighted_overlap(a1, a2, idf_addr) if a1 and a2 else 0.0
    feats[34] = _idf_weighted_overlap(na1, na2, idf_combined) if na1 and na2 else 0.0

    # ── Embedding cosine ────────────────────────────────
    feats[35] = emb_cosine

    # ── Meta features ───────────────────────────────────
    feats[36] = 1.0 if entity_id_s2.startswith("S2-") else 0.0  # source indicator
    feats[37] = min(len(n1), len(n2)) / max(len(n1), len(n2), 1)  # name length ratio
    feats[38] = min(len(a1), len(a2)) / max(len(a1), len(a2), 1) if (a1_present and a2_present) else 0.0
    feats[39] = min(len(na1), len(na2)) / max(len(na1), len(na2), 1)

    return feats


# ─────────────────────────────────────────────────────────────
# BATCH FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

@timed
def extract_features_for_pairs(
    candidates: Dict[str, Set[str]],
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    embeddings_s1: Optional[np.ndarray] = None,
    embeddings_targets: Optional[np.ndarray] = None,
    s1_eid_to_idx: Optional[Dict[str, int]] = None,
    target_eid_to_idx: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Extract features for all candidate pairs.

    Returns DataFrame with columns: s1_id, s2s3_id, feature_0, ..., feature_39
    """
    # Build lookup dictionaries for fast access
    log.info("Building lookup indices...")
    s1_lookup = {}
    for idx, row in df_s1.iterrows():
        s1_lookup[row["entity_id"]] = row

    target_lookup = {}
    for idx, row in df_targets.iterrows():
        target_lookup[row["entity_id"]] = row

    # Compute IDF
    log.info("Computing IDF scores...")
    from itertools import chain

    all_names = chain(
        (text for text in df_s1["name_clean"].dropna()),
        (text for text in df_targets["name_clean"].dropna())
    )
    all_addrs = chain(
        (text for text in df_s1["addr_clean"].dropna()),
        (text for text in df_targets["addr_clean"].dropna())
    )
    all_combined = chain(
        (text for text in df_s1["name_addr"].dropna()),
        (text for text in df_targets["name_addr"].dropna())
    )

    idf_name = compute_idf(all_names)
    idf_addr = compute_idf(all_addrs)
    idf_combined = compute_idf(all_combined)

    del all_names, all_addrs, all_combined
    gc.collect()

    # Count total pairs
    total_pairs = sum(len(v) for v in candidates.items())
    log.info(f"Extracting features for {total_pairs:,} pairs...")

    import joblib

    def process_chunk(chunk):
        batch_s1_ids = []
        batch_s2s3_ids = []
        batch_features = []
        for s1_eid, cand_set in chunk:
            if s1_eid not in s1_lookup:
                continue

            row_s1 = s1_lookup[s1_eid]
            n1 = _safe_str(row_s1.get("name_clean", ""))
            a1 = _safe_str(row_s1.get("addr_clean", ""))
            na1 = _safe_str(row_s1.get("name_addr", ""))
            sn1 = _safe_str(row_s1.get("street_num", ""))
            pc1 = _safe_str(row_s1.get("postal", ""))
            c1 = _safe_str(row_s1.get("country_clean", ""))

            for cand_eid in cand_set:
                if cand_eid not in target_lookup:
                    continue

                row_t = target_lookup[cand_eid]
                n2 = _safe_str(row_t.get("name_clean", ""))
                a2 = _safe_str(row_t.get("addr_clean", ""))
                na2 = _safe_str(row_t.get("name_addr", ""))
                sn2 = _safe_str(row_t.get("street_num", ""))
                pc2 = _safe_str(row_t.get("postal", ""))
                c2 = _safe_str(row_t.get("country_clean", ""))

                # Embedding cosine
                emb_cos = 0.0
                if (embeddings_s1 is not None and embeddings_targets is not None
                        and s1_eid_to_idx is not None and target_eid_to_idx is not None):
                    s1_idx = s1_eid_to_idx.get(s1_eid)
                    t_idx = target_eid_to_idx.get(cand_eid)
                    if s1_idx is not None and t_idx is not None:
                        emb_cos = float(np.dot(
                            embeddings_s1[s1_idx],
                            embeddings_targets[t_idx]
                        ))

                feats = extract_pair_features(
                    n1, a1, na1, sn1, pc1, c1,
                    n2, a2, na2, sn2, pc2, c2,
                    cand_eid,
                    idf_name, idf_addr, idf_combined,
                    emb_cos,
                )

                batch_s1_ids.append(s1_eid)
                batch_s2s3_ids.append(cand_eid)
                batch_features.append(feats)

        if not batch_features:
            return pd.DataFrame()

        return pd.DataFrame({
            "s1_id": batch_s1_ids,
            "s2s3_id": batch_s2s3_ids,
            **{f"f_{i}": [f[i] for f in batch_features]
               for i in range(len(FEATURE_NAMES))},
        })

    candidate_items = list(candidates.items())
    chunk_size = 5000
    chunks = [candidate_items[i:i + chunk_size] for i in range(0, len(candidate_items), chunk_size)]

    log.info(f"Processing {len(chunks)} chunks using joblib threading...")
    results = joblib.Parallel(n_jobs=-1, backend="threading")(
        joblib.delayed(process_chunk)(chunk) 
        for chunk in tqdm(chunks, desc="Feature extraction chunks", mininterval=5)
    )

    df_features = pd.concat([r for r in results if not r.empty], ignore_index=True) if results else pd.DataFrame()
    log.info(f"  Feature matrix shape: {df_features.shape}")
    log_memory()

    return df_features


def get_feature_columns() -> List[str]:
    """Return list of feature column names in the feature DataFrame."""
    return [f"f_{i}" for i in range(len(FEATURE_NAMES))]


def load_embeddings_for_features(split: str, df_s1: pd.DataFrame,
                                  df_targets: pd.DataFrame):
    """
    Load pre-computed embeddings and build entity→index mappings.

    Returns: (emb_s1, emb_targets, s1_eid_to_idx, target_eid_to_idx)
    """
    import os
    target_cache = f'output/target_embeddings_{split}.npy'
    query_cache = f'output/query_embeddings_{split}.npy'

    if not os.path.exists(target_cache) or not os.path.exists(query_cache):
        log.warning(f"Global embeddings for {split} not found! Computing them now...")
        from src.blocking import compute_embeddings
        t_texts = df_targets["name_addr"].fillna("").values
        q_texts = df_s1["name_addr"].fillna("").values
        
        emb_targets = compute_embeddings(t_texts, save_path=Path(target_cache))
        emb_s1 = compute_embeddings(q_texts, save_path=Path(query_cache))
    else:
        emb_s1 = np.load(query_cache)
        emb_targets = np.load(target_cache)

    s1_eid_to_idx = {eid: idx for idx, eid in enumerate(df_s1["entity_id"].values)}
    target_eid_to_idx = {eid: idx for idx, eid in enumerate(df_targets["entity_id"].values)}

    return emb_s1, emb_targets, s1_eid_to_idx, target_eid_to_idx


if __name__ == "__main__":
    from src.blocking import load_candidates

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
