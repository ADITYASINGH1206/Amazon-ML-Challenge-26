"""Blocking: exact nearest neighbours on the GPU inside each country, forward (S1 -> pool) and reverse (pool -> S1).

Scores are computed in fp16 on tensor cores to find the top (k + extra) per query, then re-scored exactly in fp32
and re-ranked, so the result equals an exact fp32 search up to ties. No approximate index is involved."""
import numpy as np
import pandas as pd

from .util import log


def topk_gpu(Q, P, k, extra=8, q_chunk=None, p_block=None):
    """Q (n x d), P (m x d): L2-normalised float16 arrays. Returns (idx n x k into P, -1 = none; scores fp32)."""
    import torch
    n, m = len(Q), len(P)
    k = min(k, m)
    idx = np.full((n, max(k, 0)), -1, np.int64)
    sc = np.zeros((n, max(k, 0)), np.float32)
    if n == 0 or m == 0:
        return idx, sc
    
    # Adaptive sizing: scale for high-memory GPUs (e.g. RTX 5090 32GB)
    if q_chunk is None or p_block is None:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 24:  # RTX 5090, RTX 4090, A100
                default_q, default_p = 4096, 1_000_000
            elif vram_gb >= 14:
                default_q, default_p = 2048, 500_000
            else:
                default_q, default_p = 1024, 250_000
        else:
            default_q, default_p = 512, 100_000
        q_chunk = q_chunk or default_q
        p_block = p_block or default_p

    kk = min(k + extra, m)
    dev = "cuda"
    Pg = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float16)).to(dev)
    blocks = [(s, min(s + p_block, m)) for s in range(0, m, p_block)]
    for s in range(0, n, q_chunk):
        q = torch.from_numpy(np.ascontiguousarray(Q[s:s + q_chunk], dtype=np.float16)).to(dev)
        vs, js = [], []
        for a, b in blocks:
            v, j = torch.topk(q @ Pg[a:b].T, min(kk, b - a), dim=1)
            vs.append(v)
            js.append(j + a)
        v, o = torch.topk(torch.cat(vs, 1), kk, dim=1)
        j = torch.gather(torch.cat(js, 1), 1, o)
        exact = torch.einsum("nd,nkd->nk", q.float(), Pg[j].float())       # exact fp32 re-scoring
        v, o = torch.topk(exact, k, dim=1)
        idx[s:s + len(q)] = torch.gather(j, 1, o).cpu().numpy()
        sc[s:s + len(q)] = v.cpu().numpy()
    del Pg
    torch.cuda.empty_cache()
    return idx, sc


def _assign(X, C, chunk=65536):
    """Nearest centroid (inner product) of each row of X. X, C: torch fp16 on the GPU."""
    import torch
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):
        out[s:s + chunk] = (X[s:s + chunk] @ C.T).argmax(1)
    return out


def _kmeans(X, nlist, iters=12):
    """Spherical k-means on the GPU (X: normalised fp16 rows). Returns normalised fp16 centroids."""
    import torch
    n = X.shape[0]
    C = X[torch.randperm(n, device=X.device)[:nlist]].float()
    for _ in range(iters):
        a = _assign(X, C.half())
        S = torch.zeros_like(C).index_add_(0, a, X.float())
        empty = torch.bincount(a, minlength=nlist) == 0
        if bool(empty.any()):                               # re-seed empty lists from random points
            S[empty] = X[torch.randint(0, n, (int(empty.sum()),), device=X.device)].float()
        C = torch.nn.functional.normalize(S, dim=1)
    return C.half()


def topk_ivf(Q, P, k, list_size=None, nprobe=None, extra=8, q_chunk=65536):
    """Inverted-file (IVF) nearest-neighbour search on the GPU, the same design as FAISS IndexIVFFlat.
    P is partitioned into ~len(P)/list_size k-means lists; each query scans only its nprobe nearest lists, so the
    work per query is O(n_lists + nprobe * list_size) instead of O(len(P)) -- the part that scales to billions of
    records (with product quantisation of the lists on top when memory is the limit). The kept neighbours are
    re-scored exactly in fp32. Returns (idx n x k into P, -1 = none; scores fp32) like topk_gpu."""
    import torch
    from . import config as C
    list_size = list_size or C.IVF_LIST_SIZE
    nprobe = nprobe or C.IVF_NPROBE
    n, m = len(Q), len(P)
    k = min(k, m)
    nlist = m // list_size
    if n == 0 or m == 0 or nlist < 4 * nprobe:              # tiny collection: scanning it all is already cheap
        return topk_gpu(Q, P, k, extra)
    torch.manual_seed(C.SEED)
    dev = "cuda"
    kk = k + extra
    Pg = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float16)).to(dev)
    cent = _kmeans(Pg[torch.randperm(m, device=dev)[:min(m, nlist * 64)]], nlist)
    lists = _assign(Pg, cent)
    order = torch.argsort(lists)                            # pool rows grouped by list
    off = np.concatenate([[0], np.cumsum(torch.bincount(lists, minlength=nlist).cpu().numpy())])
    del lists
    idx = np.full((n, k), -1, np.int64)
    sc = np.zeros((n, k), np.float32)
    for s in range(0, n, q_chunk):
        q = torch.from_numpy(np.ascontiguousarray(Q[s:s + q_chunk], dtype=np.float16)).to(dev)
        nq = len(q)
        probe = torch.topk(q @ cent.T, nprobe, dim=1).indices.reshape(-1)       # (nq * nprobe) list ids
        qid = torch.arange(nq, device=dev).repeat_interleave(nprobe)
        o = torch.argsort(probe)
        probe, qid = probe[o], qid[o]
        ul, cnt = torch.unique_consecutive(probe, return_counts=True)
        ul, bnd = ul.cpu().numpy(), np.concatenate([[0], np.cumsum(cnt.cpu().numpy())])
        best_v = torch.full((nq, kk), -1e4, dtype=torch.float16, device=dev)
        best_i = torch.full((nq, kk), -1, dtype=torch.long, device=dev)
        for t, l in enumerate(ul):                          # list-major: one dense block per probed list
            a, b = int(off[l]), int(off[l + 1])
            if b == a:
                continue
            qs = qid[bnd[t]:bnd[t + 1]]
            mem = order[a:b]
            v, j = torch.topk(q[qs] @ Pg[mem].T, min(kk, b - a), dim=1)
            v = torch.cat([best_v[qs], v], 1)
            j = torch.cat([best_i[qs], mem[j]], 1)
            v, o2 = torch.topk(v, kk, dim=1)
            best_v[qs], best_i[qs] = v, torch.gather(j, 1, o2)
        ok = best_i >= 0
        safe = torch.where(ok, best_i, torch.zeros_like(best_i))
        exact = torch.einsum("nd,nkd->nk", q.float(), Pg[safe].float())         # exact fp32 re-scoring
        exact = torch.where(ok, exact, torch.full_like(exact, -1e4))
        v, o3 = torch.topk(exact, k, dim=1)
        j = torch.gather(safe, 1, o3)
        j = torch.where(v > -1e3, j, torch.full_like(j, -1))
        idx[s:s + nq] = j.cpu().numpy()
        sc[s:s + nq] = v.cpu().numpy()
    del Pg, cent, order
    torch.cuda.empty_cache()
    return idx, sc


def _search(Q, P, k):
    from . import config as C
    return topk_ivf(Q, P, k) if C.BLOCK_INDEX == "ivf" else topk_gpu(Q, P, k)


def block(E1, Ep, s1_country, pool_country, k_fwd, m_rev, fwd_rows=None, rev_mask=None):
    """Union of forward top-k_fwd and reverse top-m_rev neighbours inside each country.
    fwd_rows: S1 rows that need candidates (default all).
    rev_mask: S1 rows that exist as reverse-search targets (default all). Training excludes a slice of S1 so that
    their S2/S3 records become ownerless, matching the test set's higher share of pool records without an S1.
    Returns (cands[qi, pj, cos, fwd_rank, rev_rank], q_top1, p_top1)."""
    n1, n2 = len(s1_country), len(pool_country)
    need = np.ones(n1, bool) if fwd_rows is None else np.isin(np.arange(n1), fwd_rows)
    present = np.ones(n1, bool) if rev_mask is None else np.asarray(rev_mask, bool)
    q_top1 = np.zeros(n1, np.float32)
    p_top1 = np.zeros(n2, np.float32)
    fwd, rev = [], []
    for c in pd.unique(s1_country):
        qa = np.where((s1_country == c) & present)[0]
        qn = qa[need[qa]]
        pi = np.where(pool_country == c)[0]
        if len(pi) == 0 or len(qn) == 0:
            continue
        log(f"  [{c}] forward {len(qn):,} x {len(pi):,}")
        fi, fs = _search(E1[qn], Ep[pi], k_fwd)
        q_top1[qn] = fs[:, 0]
        ok = fi >= 0
        fwd.append(pd.DataFrame({"qi": np.repeat(qn, fi.shape[1])[ok.ravel()], "pj": pi[fi[ok]],
                                 "fwd_rank": np.tile(np.arange(fi.shape[1]), len(qn))[ok.ravel()].astype(np.float32),
                                 "cos": fs[ok]}))
        log(f"  [{c}] reverse {len(pi):,} x {len(qa):,}")
        ri, rs = _search(Ep[pi], E1[qa], m_rev)
        p_top1[pi] = rs[:, 0]
        ok = ri >= 0
        d = pd.DataFrame({"pj": np.repeat(pi, ri.shape[1])[ok.ravel()], "qi": qa[ri[ok]],
                          "rev_rank": np.tile(np.arange(ri.shape[1]), len(pi))[ok.ravel()].astype(np.float32),
                          "cos_r": rs[ok]})
        rev.append(d[need[d.qi.values]])
    f = pd.concat(fwd, ignore_index=True) if fwd else pd.DataFrame(columns=["qi", "pj", "fwd_rank", "cos"])
    r = pd.concat(rev, ignore_index=True) if rev else pd.DataFrame(columns=["pj", "qi", "rev_rank", "cos_r"])
    c = f.merge(r, on=["qi", "pj"], how="outer")
    c["cos"] = c["cos"].fillna(c["cos_r"]).astype(np.float32)
    c = c.drop(columns="cos_r").astype({"qi": np.int64, "pj": np.int64})
    return c[["qi", "pj", "cos", "fwd_rank", "rev_rank"]].reset_index(drop=True), q_top1, p_top1


def add_context(c, q_top1, p_top1, pool_src):
    c["gap_q"] = (q_top1[c.qi.values] - c.cos.values).astype(np.float32)
    c["rev_margin"] = (p_top1[c.pj.values] - c.cos.values).astype(np.float32)
    c["mutual"] = ((c.fwd_rank == 0) & (c.rev_rank == 0)).astype(np.float32)
    c["n_cand_q"] = c.groupby("qi").pj.transform("size").astype(np.float32)
    c["is_s3"] = (pool_src[c.pj.values] == "S3").astype(np.float32)
    return c
