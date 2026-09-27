"""Nearest-neighbour candidate generation: sparse TF-IDF exact top-k (sparse_dot_topn when available),
forward (S1 -> pool) and reverse (pool -> S1) search, per-country grouping, and an approximate dense
variant (TruncatedSVD + FAISS HNSW) to measure what we lose by going approximate at full scale."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sparse_dot_topn import sp_matmul_topn
    HAVE_SDT = True
except ImportError:
    HAVE_SDT = False
try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

VECTORIZERS = {
    "char_wb24": dict(analyzer="char_wb", ngram_range=(2, 4)),
    "char_wb33": dict(analyzer="char_wb", ngram_range=(3, 3)),
    "char_wb35": dict(analyzer="char_wb", ngram_range=(3, 5)),
    "word": dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 1)),
}


def make_vectorizer(kind="char_wb24", min_df=2, max_features=None):
    return TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=min_df, max_features=max_features,
                           **VECTORIZERS[kind])


def _csr_topk(C, k):
    n = C.shape[0]
    cnt = np.diff(C.indptr)
    rows = np.repeat(np.arange(n), cnt)
    order = np.lexsort((-C.data, rows))
    rows, cols, data = rows[order], C.indices[order], C.data[order]
    pos = np.arange(len(rows)) - np.repeat(np.cumsum(np.r_[0, cnt[:-1]]), cnt)
    keep = pos < k
    idx = np.full((n, k), -1, dtype=np.int64)
    sc = np.zeros((n, k), dtype=np.float32)
    idx[rows[keep], pos[keep]] = cols[keep]
    sc[rows[keep], pos[keep]] = data[keep]
    return idx, sc


def _topk_dense_fallback(Q, P, k):
    n, m = Q.shape[0], P.shape[0]
    PT = P.T.tocsr()
    ch = max(1, int(2e7 // max(m, 1)))
    idx = np.full((n, k), -1, dtype=np.int64)
    sc = np.zeros((n, k), dtype=np.float32)
    for i in range(0, n, ch):
        S = (Q[i:i + ch] @ PT).toarray()
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        o = np.argsort(-ps, axis=1)
        idx[i:i + ch] = np.take_along_axis(part, o, 1)
        sc[i:i + ch] = np.take_along_axis(ps, o, 1)
    idx[sc <= 0] = -1
    return idx, sc


def topk_sparse(Q, P, k, n_jobs=1, chunk=50_000):
    k = int(min(k, P.shape[0]))
    if k <= 0 or Q.shape[0] == 0:
        return np.full((Q.shape[0], max(k, 0)), -1, np.int64), np.zeros((Q.shape[0], max(k, 0)), np.float32)
    if not HAVE_SDT:
        return _topk_dense_fallback(Q, P, k)
    PT = P.T.tocsr()
    outs_i, outs_s = [], []
    for i in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[i:i + chunk], PT, top_n=k, sort=True, n_threads=(n_jobs if n_jobs > 1 else None))
        a, b = _csr_topk(C.tocsr(), k)
        outs_i.append(a)
        outs_s.append(b)
    return np.vstack(outs_i), np.vstack(outs_s)


def knn_grouped(Q, P, qg, pg, k, n_jobs=1, fn=None):
    """Top-k of every Q row among P rows of the same group (country). Returns global P indices (-1 = none)."""
    fn = fn or topk_sparse
    n = Q.shape[0]
    idx = np.full((n, k), -1, dtype=np.int64)
    sc = np.zeros((n, k), dtype=np.float32)
    for g in pd.unique(qg):
        qi = np.where(qg == g)[0]
        pi = np.where(pg == g)[0]
        if len(pi) == 0 or len(qi) == 0:
            continue
        a, b = fn(Q[qi], P[pi], k, n_jobs)
        kk = a.shape[1]
        ok = a >= 0
        glob = np.where(ok, pi[np.clip(a, 0, None)], -1)
        idx[qi, :kk] = glob
        sc[qi, :kk] = np.where(ok, b, 0)
    return idx, sc


def pair_cos(Q, P, qi, pj, chunk=500_000):
    out = np.zeros(len(qi), dtype=np.float32)
    for s in range(0, len(qi), chunk):
        a = Q[qi[s:s + chunk]]
        b = P[pj[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def build_candidates(Q, P, qg, pg, k_fwd, m_rev, n_jobs=1, fn=None):
    """Union of forward top-k_fwd (S1 -> pool) and reverse top-m_rev (pool -> S1) neighbours.
    Returns (cands df[qi, pj, cos, fwd_rank, rev_rank], q_top1 score per S1, p_top1 score per pool)."""
    parts = []
    q_top1 = np.zeros(Q.shape[0], np.float32)
    p_top1 = np.zeros(P.shape[0], np.float32)
    if k_fwd > 0:
        fi, fs = knn_grouped(Q, P, qg, pg, k_fwd, n_jobs, fn)
        q_top1 = fs[:, 0].copy()
        r = np.tile(np.arange(fi.shape[1]), fi.shape[0])
        d = pd.DataFrame({"qi": np.repeat(np.arange(fi.shape[0]), fi.shape[1]), "pj": fi.ravel(),
                          "fwd_rank": r.astype(np.float32), "cos": fs.ravel()})
        parts.append(d[d.pj >= 0])
    if m_rev > 0:
        ri, rs = knn_grouped(P, Q, pg, qg, m_rev, n_jobs, fn)
        p_top1 = rs[:, 0].copy()
        r = np.tile(np.arange(ri.shape[1]), ri.shape[0])
        d = pd.DataFrame({"pj": np.repeat(np.arange(ri.shape[0]), ri.shape[1]), "qi": ri.ravel(),
                          "rev_rank": r.astype(np.float32), "cos_r": rs.ravel()})
        parts.append(d[d.qi >= 0])
    if len(parts) == 2:
        c = parts[0].merge(parts[1], on=["qi", "pj"], how="outer")
        c["cos"] = c["cos"].fillna(c["cos_r"])
        c = c.drop(columns="cos_r")
    elif k_fwd > 0:
        c = parts[0].assign(rev_rank=np.nan)
    else:
        c = parts[0].rename(columns={"cos_r": "cos"}).assign(fwd_rank=np.nan)
    c = c.reset_index(drop=True)
    if k_fwd <= 0:
        q_top1 = c.groupby("qi").cos.max().reindex(range(Q.shape[0])).fillna(0).values.astype(np.float32)
    return c[["qi", "pj", "cos", "fwd_rank", "rev_rank"]], q_top1, p_top1


def svd_dense(Q, P, dim=128, fit_rows=200_000, seed=0):
    from sklearn.decomposition import TruncatedSVD
    rng = np.random.default_rng(seed)
    rows = rng.choice(P.shape[0], size=min(fit_rows, P.shape[0]), replace=False)
    svd = TruncatedSVD(n_components=min(dim, P.shape[1] - 1), random_state=seed).fit(P[rows])
    q = svd.transform(Q).astype(np.float32)
    p = svd.transform(P).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    return q, p


def _topk_dense_torch(q, p, k, block_bytes=1 << 30):
    """Exact fp32 inner-product top-k on the GPU (same result as the numpy path, ~100x faster). None if no CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    P = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32)).cuda()
    ch = max(1, int(block_bytes // (4 * max(p.shape[0], 1))))
    idx = np.zeros((q.shape[0], k), np.int64)
    sc = np.zeros((q.shape[0], k), np.float32)
    for i in range(0, q.shape[0], ch):
        Q = torch.from_numpy(np.ascontiguousarray(q[i:i + ch], dtype=np.float32)).cuda()
        v, j = torch.topk(Q @ P.T, k, dim=1)
        idx[i:i + ch] = j.cpu().numpy()
        sc[i:i + ch] = v.cpu().numpy()
    del P
    torch.cuda.empty_cache()
    return idx, sc


def topk_dense(q, p, k, n_jobs=1, hnsw=True):
    """Approximate (FAISS HNSW) or exact (numpy) inner-product top-k on dense normalised vectors."""
    k = int(min(k, p.shape[0]))
    if HAVE_FAISS and hnsw:
        faiss.omp_set_num_threads(max(1, n_jobs))
        index = faiss.IndexHNSWFlat(p.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.add(np.ascontiguousarray(p))
        index.hnsw.efSearch = max(64, 2 * k)
        sc, idx = index.search(np.ascontiguousarray(q), k)
        return idx.astype(np.int64), sc.astype(np.float32)
    gpu = _topk_dense_torch(q, p, k)
    if gpu is not None:
        return gpu
    idx = np.zeros((q.shape[0], k), np.int64)
    sc = np.zeros((q.shape[0], k), np.float32)
    ch = max(1, int(5e7 // max(p.shape[0], 1)))
    for i in range(0, q.shape[0], ch):
        S = q[i:i + ch] @ p.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        o = np.argsort(-ps, axis=1)
        idx[i:i + ch] = np.take_along_axis(part, o, 1)
        sc[i:i + ch] = np.take_along_axis(ps, o, 1)
    return idx, sc
