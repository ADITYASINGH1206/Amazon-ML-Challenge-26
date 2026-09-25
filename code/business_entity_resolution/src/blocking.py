"""
blocking.py — Multi-strategy candidate generation (Stage 1).

Combines four blocking strategies to achieve >95% recall:
1. Composite-key inverted index (city + name prefix, postal code, etc.)
2. Character n-gram TF-IDF sparse retrieval
3. Phonetic (Soundex) key blocking
4. Dense PyTorch ANN retrieval (multilingual embeddings)

All strategies are partitioned by country to reduce search space.
The union of all candidates becomes the final blocking set.
"""

import os
import joblib

os.makedirs('output', exist_ok=True)

import gc
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.utils import log, timed, log_memory
from src.preprocess import load_preprocessed

HAS_JELLYFISH = True


# ─────────────────────────────────────────────────────────────
# STRATEGY 1: COMPOSITE-KEY INVERTED INDEX
# ─────────────────────────────────────────────────────────────

def _get_blocking_keys(name_tokens: list, addr_tokens: list,
                       postal: str, country: str) -> List[str]:
    """
    Generate multiple composite blocking keys for one entity.

    Keys are designed to be selective (small buckets) but overlapping
    so the union catches most true pairs.
    """
    keys = []
    name_sig = [t for t in name_tokens if len(t) > 2]  # skip short tokens

    # Key type 1: country + first significant name word
    if name_sig:
        keys.append(f"n1:{country}:{name_sig[0]}")
        if len(name_sig) > 1:
            keys.append(f"n2:{country}:{name_sig[1]}")

    # Key type 2: country + postal code
    if postal:
        keys.append(f"pc:{country}:{postal}")

    # Key type 3: country + first name word + first addr word
    addr_sig = [t for t in addr_tokens if len(t) > 2 and not t.isdigit()]
    if name_sig and addr_sig:
        keys.append(f"na:{country}:{name_sig[0]}:{addr_sig[0]}")

    # Key type 4: country + street number + first addr word
    street_nums = [t for t in addr_tokens if t.isdigit()]
    if street_nums and addr_sig:
        keys.append(f"sn:{country}:{street_nums[0]}:{addr_sig[0]}")

    # Key type 5: phonetic key (Soundex of first name word + country)
    if HAS_JELLYFISH and name_sig:
        try:
            import jellyfish
            sdx = jellyfish.soundex(name_sig[0])
            keys.append(f"sx:{country}:{sdx}")
            if len(name_sig) > 1:
                sdx2 = jellyfish.soundex(name_sig[1])
                keys.append(f"sx2:{country}:{sdx}:{sdx2}")
        except Exception:
            pass

    # Key type 6: first 4 chars of name + country (handles short names)
    name_joined = "".join(name_sig)
    if len(name_joined) >= 4:
        keys.append(f"pfx:{country}:{name_joined[:4]}")

    return keys


import os
import concurrent.futures
import joblib
import uuid

def _build_index_chunk(df_chunk: pd.DataFrame, start_idx: int, temp_dir: str) -> str:
    index = defaultdict(list)
    for i in range(len(df_chunk)):
        row = df_chunk.iloc[i]
        name_tokens = row["name_tokens"] if isinstance(row["name_tokens"], list) else []
        addr_tokens = row["addr_tokens"] if isinstance(row["addr_tokens"], list) else []
        postal = str(row.get("postal", "")) if pd.notna(row.get("postal", "")) else ""
        country = str(row.get("country_clean", ""))

        keys = _get_blocking_keys(name_tokens, addr_tokens, postal, country)
        for key in keys:
            index[key].append(start_idx + i)
            
    file_path = os.path.join(temp_dir, f"chunk_{uuid.uuid4().hex}.joblib")
    joblib.dump(dict(index), file_path)
    return file_path

@timed
def build_inverted_index(df_targets: pd.DataFrame, split: str = "train") -> Dict[str, List[int]]:
    """
    Build an inverted index: blocking_key → list of target DataFrame indices.
    """
    cache_path = f"output/inverted_index_{split}.joblib"
    if os.path.exists(cache_path):
        log.info("  Loading inverted index from cache...")
        return joblib.load(cache_path)

    max_bucket = config.TOKEN_MAX_DF
    n_workers = os.cpu_count() or 4
    chunk_size = max(1, len(df_targets) // n_workers)
    
    temp_dir = "output/temp_indexes"
    os.makedirs(temp_dir, exist_ok=True)
    
    chunks = []
    for i in range(0, len(df_targets), chunk_size):
        chunks.append((df_targets.iloc[i:i+chunk_size], i))
        
    global_index = defaultdict(list)
    
    log.info(f"  Building inverted index using {n_workers} processes in {len(chunks)} chunks...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_build_index_chunk, chunk, start, temp_dir) for chunk, start in chunks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Merging index chunks"):
            file_path = future.result()
            partial_idx = joblib.load(file_path)
            for key, val in partial_idx.items():
                global_index[key].extend(val)
            os.remove(file_path)

    # Prune overly common keys
    pruned = 0
    for key in list(global_index.keys()):
        if len(global_index[key]) > max_bucket:
            del global_index[key]
            pruned += 1

    final_index = dict(global_index)
    joblib.dump(final_index, cache_path)
    log.info(f"  Inverted index: {len(final_index):,} keys, {pruned:,} pruned (>{max_bucket})")
    return final_index


def query_inverted_index(df_queries: pd.DataFrame,
                         inv_index: Dict[str, List[int]],
                         target_entity_ids: np.ndarray,
                         max_candidates: int = 50) -> Dict[str, Set[str]]:
    """
    Query the inverted index and return candidate sets per query entity.

    Returns: {query_entity_id: set(target_entity_ids)}
    """
    candidates = {}
    target_eids_arr = np.array(target_entity_ids)

    for idx in tqdm(range(len(df_queries)), desc="Querying inverted index",
                    mininterval=10):
        row = df_queries.iloc[idx]
        q_eid = row["entity_id"]
        name_tokens = row["name_tokens"] if isinstance(row["name_tokens"], list) else []
        addr_tokens = row["addr_tokens"] if isinstance(row["addr_tokens"], list) else []
        postal = str(row.get("postal", "")) if pd.notna(row.get("postal", "")) else ""
        country = str(row.get("country_clean", ""))

        keys = _get_blocking_keys(name_tokens, addr_tokens, postal, country)

        # Count how many keys each target shares with this query
        cand_counts = Counter()
        for key in keys:
            if key in inv_index:
                for target_idx in inv_index[key]:
                    cand_counts[target_idx] += 1

        # Keep candidates with at least MIN_SHARED_TOKENS shared keys
        top = cand_counts.most_common(max_candidates)
        cand_eids = set()
        for target_idx, count in top:
            if count >= config.MIN_SHARED_TOKENS:
                if target_idx < len(target_eids_arr):
                    cand_eids.add(target_eids_arr[target_idx])

        candidates[q_eid] = cand_eids

    return candidates


# ─────────────────────────────────────────────────────────────
# STRATEGY 2: TF-IDF CHARACTER N-GRAM RETRIEVAL
# ─────────────────────────────────────────────────────────────

@timed
def tfidf_blocking_by_country(
    df_queries: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_k: int = 30,
    batch_size: int = 100,
) -> Dict[str, Set[str]]:
    """
    TF-IDF character n-gram blocking, processed per-country partition.

    Uses dense matrix multiplication in tiny batches to prevent OOM.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    candidates = {}
    countries = df_queries["country_clean"].unique()

    for country in countries:
        log.info(f"  TF-IDF blocking for country: {country}")

        q_mask = df_queries["country_clean"] == country
        t_mask = df_targets["country_clean"] == country

        q_df = df_queries[q_mask].reset_index(drop=True)
        t_df = df_targets[t_mask].reset_index(drop=True)

        if len(q_df) == 0 or len(t_df) == 0:
            continue

        log.info(f"    Queries: {len(q_df):,}, Targets: {len(t_df):,}")

        # Fit TF-IDF on targets
        vectorizer = TfidfVectorizer(
            analyzer=config.TFIDF_ANALYZER,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            max_features=config.TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            dtype=np.float32,
        )

        t_texts = t_df["name_addr"].fillna("").values
        q_texts = q_df["name_addr"].fillna("").values

        log.info("    Fitting TF-IDF vectorizer...")
        tfidf_targets = vectorizer.fit_transform(t_texts)
        log.info(f"    Target TF-IDF shape: {tfidf_targets.shape}")

        # Use sparse_dot_topn for massive speedup and zero OOM risk
        from sparse_dot_topn import sp_matmul_topn
        import multiprocessing
        
        tfidf_queries = vectorizer.transform(q_texts)
        
        log.info(f"    Computing sparse dot product (top_{top_k})...")
        # Ensure CSR format for fast C++ processing
        A = tfidf_queries.tocsr()
        B_T = tfidf_targets.transpose().tocsr()
        
        # This executes in C++, keeps only top_k per row, and never instantiates the dense matrix!
        # Enable multi-threading to use all CPU cores
        n_jobs = multiprocessing.cpu_count()
        sim_sparse = sp_matmul_topn(A, B_T, top_n=top_k, n_threads=n_jobs)
        
        q_eids = q_df["entity_id"].values
        
        log.info("    Extracting candidates...")
        # sim_sparse is a CSR matrix
        for i in range(sim_sparse.shape[0]):
            q_eid = q_eids[i]
            if q_eid not in candidates:
                candidates[q_eid] = set()
                
            # CSR format: indices for row i are stored in indices[indptr[i]:indptr[i+1]]
            start_idx = sim_sparse.indptr[i]
            end_idx = sim_sparse.indptr[i+1]
            
            for ptr in range(start_idx, end_idx):
                ti = sim_sparse.indices[ptr]
                score = sim_sparse.data[ptr]
                if score > 0:
                    candidates[q_eid].add(t_eids[ti])
                    
        del tfidf_targets, tfidf_queries, sim_sparse, vectorizer
        gc.collect()

    return candidates


# ─────────────────────────────────────────────────────────────
# STRATEGY 3: DENSE PYTORCH ANN RETRIEVAL
# ─────────────────────────────────────────────────────────────

@timed
def compute_embeddings(texts: np.ndarray, model_name: str = None,
                       batch_size: int = None,
                       save_path: Optional[Path] = None) -> np.ndarray:
    """
    Compute sentence embeddings using a multilingual model.

    """
    if save_path is not None and Path(save_path).exists():
        log.info(f"  Loading embeddings from {save_path}...")
        return np.load(save_path)

    from sentence_transformers import SentenceTransformer

    if model_name is None:
        model_name = config.EMBEDDING_MODEL_NAME
    if batch_size is None:
        batch_size = config.EMBEDDING_BATCH_SIZE

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  Device: {device}")

    log.info(f"  Computing embeddings with {model_name} for {len(texts):,} texts...")
    model = SentenceTransformer(model_name, device=device)

    embeddings = model.encode(
        texts.tolist(),
        batch_size=4096,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine → inner product
        device=device,
    )

    if save_path is not None:
        np.save(save_path, embeddings)
        log.info(f"  Embeddings saved to {save_path.name}")

    return embeddings.astype(np.float32)


@timed
def dense_blocking_by_country(
    df_queries: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_k: int = None,
    split: str = "train"
) -> Dict[str, Set[str]]:
    """
    Dense retrieval per country partition using native PyTorch matrix multiplication.
    """

    if top_k is None:
        top_k = config.FAISS_TOP_K

    import torch

    target_cache = f'output/target_embeddings_{split}.npy'
    query_cache = f'output/query_embeddings_{split}.npy'

    if os.path.exists(target_cache) and os.path.exists(query_cache):
        log.info("  Loading global embeddings from cache...")
        global_emb_targets = np.load(target_cache)
        global_emb_queries = np.load(query_cache)
    else:
        log.info("  Computing global embeddings...")
        t_texts = df_targets["name_addr"].fillna("").values
        q_texts = df_queries["name_addr"].fillna("").values

        global_emb_targets = compute_embeddings(t_texts, save_path=Path(target_cache))
        global_emb_queries = compute_embeddings(q_texts, save_path=Path(query_cache))

    candidates = {}
    countries = df_queries["country_clean"].unique()

    for country in countries:
        log.info(f"  Dense blocking for country: {country}")

        q_mask = df_queries["country_clean"] == country
        t_mask = df_targets["country_clean"] == country

        q_df = df_queries[q_mask].reset_index(drop=True)
        t_df = df_targets[t_mask].reset_index(drop=True)

        if len(q_df) == 0 or len(t_df) == 0:
            continue

        log.info(f"    Queries: {len(q_df):,}, Targets: {len(t_df):,}")

        emb_targets = global_emb_targets[t_mask]
        emb_queries = global_emb_queries[q_mask]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        log.info(f"    Moving target embeddings to {device}...")
        target_tensor = torch.tensor(emb_targets, dtype=torch.float16, device=device)

        t_eids = t_df["entity_id"].values
        q_eids = q_df["entity_id"].values

        search_batch = 500
        for start in tqdm(range(0, len(emb_queries), search_batch),
                          desc=f"  Dense search {country}", mininterval=5):
            end = min(start + search_batch, len(emb_queries))
            batch_q = emb_queries[start:end]

            query_tensor = torch.tensor(batch_q, dtype=torch.float16, device=device)
            sim_matrix = torch.matmul(query_tensor, target_tensor.T)

            k = min(top_k, sim_matrix.shape[1])
            scores, indices = torch.topk(sim_matrix, k=k, dim=1)
            indices_np = indices.cpu().numpy()

            for i in range(len(batch_q)):
                q_eid = q_eids[start + i]
                if q_eid not in candidates:
                    candidates[q_eid] = set()

                for j in range(k):
                    t_idx = indices_np[i, j]
                    candidates[q_eid].add(t_eids[t_idx])

        # Cleanup
        del target_tensor, emb_targets, emb_queries
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    return candidates


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR: UNION ALL BLOCKING STRATEGIES
# ─────────────────────────────────────────────────────────────

@timed
def run_blocking(split: str = "train",
                 save: bool = True) -> Dict[str, Set[str]]:
    """
    Run all blocking strategies and union their candidate sets.

    Returns: {s1_entity_id: set(candidate_entity_ids)}
    """
    # Load preprocessed data
    log.info(f"Loading preprocessed {split} data...")
    df_s1 = load_preprocessed(split, "s1")
    df_s2 = load_preprocessed(split, "s2")
    df_s3 = load_preprocessed(split, "s3")

    # Combine S2 + S3 as target set
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    log.info(f"Total targets (S2+S3): {len(df_targets):,}")
    log_memory()

    target_eids = df_targets["entity_id"].values

    # ── Strategy 1: Inverted Index ──────────────────────────
    log.info("═══ Strategy 1: Inverted Index Blocking ═══")
    inv_cands_path = f"output/inv_candidates_{split}.joblib"
    if os.path.exists(inv_cands_path):
        log.info("  Loading inverted index candidates from cache...")
        cands_inv = joblib.load(inv_cands_path)
    else:
        inv_index = build_inverted_index(df_targets, split=split)
        cands_inv = query_inverted_index(
            df_s1, inv_index, target_eids,
            max_candidates=config.MAX_CANDIDATES_PER_ENTITY
        )
        del inv_index
        joblib.dump(cands_inv, inv_cands_path)
        
    gc.collect()
    log.info(f"  Inverted index candidates: "
             f"{sum(len(v) for v in cands_inv.values()):,} pairs")
    log_memory()

    # ── Strategy 2: PyTorch Dense Blocking ────────────────────
    log.info("═══ Strategy 2: PyTorch Dense Blocking ═══")
    dense_cands_path = f"output/dense_candidates_{split}.joblib"
    if os.path.exists(dense_cands_path):
        log.info("  Loading dense candidates from cache...")
        cands_dense = joblib.load(dense_cands_path)
    else:
        cands_dense = dense_blocking_by_country(df_s1, df_targets, top_k=config.FAISS_TOP_K, split=split)
        joblib.dump(cands_dense, dense_cands_path)
        
    log.info(f"  Dense candidates: "
             f"{sum(len(v) for v in cands_dense.values()):,} pairs")
    log_memory()

    # ── Union all strategies ────────────────────────────────
    log.info("═══ Merging all blocking strategies ═══")
    all_s1_eids = df_s1["entity_id"].values
    final_candidates = {}

    for s1_eid in tqdm(all_s1_eids, desc="Merging candidates", mininterval=10):
        merged = set()
        merged.update(cands_inv.get(s1_eid, set()))
        merged.update(cands_dense.get(s1_eid, set()))

        # Cap at max candidates (keep all if under limit)
        if len(merged) > config.MAX_CANDIDATES_PER_ENTITY * 2:
            # If too many, we can't just randomly drop — keep all for now
            # The LightGBM cascade will filter anyway
            pass

        final_candidates[s1_eid] = merged

    # Stats
    n_pairs = sum(len(v) for v in final_candidates.values())
    n_empty = sum(1 for v in final_candidates.values() if len(v) == 0)
    avg_cands = n_pairs / max(len(final_candidates), 1)

    log.info(f"  Final blocking: {len(final_candidates):,} S1 entities, "
             f"{n_pairs:,} total pairs, avg {avg_cands:.1f} candidates/entity, "
             f"{n_empty:,} entities with no candidates")

    # Save
    if save:
        save_path = config.BLOCKING_DIR / f"{split}_candidates.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(final_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"  Saved to {save_path.name}")
        
        # Output TSV for candidates
        tsv_path = Path(f"output/candidate_pairs_{split}.tsv")
        tsv_path.parent.mkdir(exist_ok=True, parents=True)
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for q_eid, c_eids in final_candidates.items():
                c_str = str(list(c_eids))
                f.write(f"{q_eid}\t{c_str}\n")
        log.info(f"  Saved TSV to {tsv_path}")

    return final_candidates


def load_candidates(split: str) -> Dict[str, Set[str]]:
    """Load saved candidate pairs."""
    path = config.BLOCKING_DIR / f"{split}_candidates.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@timed
def evaluate_blocking_recall(candidates: Dict[str, Set[str]],
                              ground_truth: Dict[str, Set[str]]) -> float:
    """
    Compute blocking recall = fraction of true pairs captured by blocking.
    This sets the upper bound on model recall.
    """
    total_true = 0
    captured = 0

    for s1_eid, true_matches in ground_truth.items():
        if not true_matches:
            continue
        total_true += len(true_matches)
        cands = candidates.get(s1_eid, set())
        captured += len(true_matches & cands)

    recall = captured / total_true if total_true > 0 else 0.0
    log.info(f"  Blocking recall: {recall:.4f} ({captured:,}/{total_true:,})")
    return recall


if __name__ == "__main__":
    from src.utils import load_ground_truth

    # Run blocking on training data
    candidates = run_blocking("train")

    # Evaluate blocking recall
    gt = load_ground_truth()
    evaluate_blocking_recall(candidates, gt)
