"""Matchers (LightGBM), stage-2 group features, decision layer and the challenge's macro F0.5."""
import numpy as np
import pandas as pd

from .config import LGB_PARAMS, LGB_ROUNDS, LGB_STAGE2


# ----------------------------------------------------------------------------- scoring (exact challenge definition)
def _dedupe(qi, pj):
    if len(qi) == 0:
        return np.asarray(qi, np.int64), np.asarray(pj, np.int64)
    key = np.asarray(qi, np.int64) * (np.int64(1) << 32) + np.asarray(pj, np.int64)
    _, first = np.unique(key, return_index=True)
    return np.asarray(qi)[first], np.asarray(pj)[first]


def per_entity_f05(qi, pj, own, k):
    qi, pj = _dedupe(qi, pj)
    n = len(k)
    npred = np.bincount(qi, minlength=n)
    tp = np.bincount(qi[own[pj] == qi], minlength=n) if len(qi) else np.zeros(n, np.int64)
    f = np.where((k == 0) & (npred == 0), 1.0,
                 np.where((k == 0) | (npred == 0), 0.0, 1.25 * tp / (0.25 * k + np.maximum(npred, 1))))
    return f, npred, tp


def macro_f05(qi, pj, own, k, mask, countries=None):
    f, npred, tp = per_entity_f05(qi, pj, own, k)
    m = mask
    res = {"f05": float(f[m].mean()),
           "f05_singletons": float(f[m & (k == 0)].mean()) if (m & (k == 0)).any() else None,
           "f05_nonsingletons": float(f[m & (k > 0)].mean()) if (m & (k > 0)).any() else None,
           "micro_precision": float(tp[m].sum() / max(npred[m].sum(), 1)),
           "micro_recall": float(tp[m].sum() / max(k[m].sum(), 1)),
           "avg_pred_per_s1": float(npred[m].mean()),
           "singleton_false_merge_rate": float((npred[m & (k == 0)] > 0).mean()) if (m & (k == 0)).any() else None,
           "n_s1": int(m.sum())}
    if countries is not None:
        for c in pd.unique(countries[m]):
            res[f"f05_{c}"] = float(f[m & (countries == c)].mean())
    return res


def blocking_stats(qi, pj, own, k, mask):
    qi, pj = _dedupe(qi, pj)
    n = len(k)
    size = np.bincount(qi, minlength=n)
    tp = np.bincount(qi[own[pj] == qi], minlength=n) if len(qi) else np.zeros(n, np.int64)
    ceil = np.where(k == 0, 1.0, np.where(tp == 0, 0.0, 1.25 * tp / (0.25 * k + np.maximum(tp, 1))))
    m = mask
    return {"pair_recall": float(tp[m].sum() / max(k[m].sum(), 1)), "ceiling_f05": float(ceil[m].mean()),
            "cand_mean": float(size[m].mean()), "cand_p90": float(np.percentile(size[m], 90)),
            "cand_max": int(size[m].max()), "no_cand_rate": float((size[m] == 0).mean())}


# ----------------------------------------------------------------------------- LightGBM
def fit_lgb(X, y, groups, params, n_jobs, rounds=LGB_ROUNDS):
    """Early stopping on the S1 entities with group % 10 == 0 (never split an S1 across fit / validation)."""
    import lightgbm as lgb
    inner = (groups % 10) == 0
    p = dict(params, num_threads=n_jobs)
    m = lgb.train(p, lgb.Dataset(X[~inner], y[~inner]), rounds, valid_sets=[lgb.Dataset(X[inner], y[inner])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def predict(m, X):
    return m.predict(X, num_iteration=m.best_iteration).astype(np.float32)


# ----------------------------------------------------------------------------- stage 2 (group-aware stacking)
def _tri(s):
    s = f" {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def add_pj_features(d):
    """Competition for the same pool record across all S1: number of claims, rank of this claim, best other p."""
    s = d[["pj", "p"]]                       # slim copy: never sort / copy the wide frame
    gp = s.groupby("pj")
    d["s2_nclaim_pj"] = gp.p.transform("size").values.astype(np.float32)
    d["s2_rank_pj"] = gp.p.rank(ascending=False, method="first").values.astype(np.float32)
    pmax_pj = gp.p.transform("max").values
    srt = s.sort_values(["pj", "p"], ascending=[True, False])
    second = srt.groupby("pj").p.nth(1)
    top2 = s.pj.map(pd.Series(second.values, index=srt.loc[second.index, "pj"].values)).fillna(0.0).values
    d["s2_pmax_other_pj"] = np.where(s.p.values >= pmax_pj, top2, pmax_pj).astype(np.float32)
    return d


def stage2_features(d, pool_nname, pool_naddr, pool_nums, pool_src):
    """d: qi, pj, p (+ anything). Adds s2_* features: rank / share of the S1's best, competing strong candidates,
    competition for the same pool record, and similarity to the S1's strongest other candidate (sibling)."""
    d = d.sort_values(["qi", "p"], ascending=[True, False]).reset_index(drop=True)
    g = d.groupby("qi")
    d["s2_rank_q"] = g.cumcount().astype(np.float32)
    d["s2_pmax_q"] = g.p.transform("max")
    d["s2_prel"] = d.p / (d.s2_pmax_q + 1e-9)
    d["s2_psum_q"] = g.p.transform("sum")
    d["_hi"] = (d.p > 0.5).astype(np.float32)
    d["s2_nhi_q"] = d.groupby("qi")._hi.transform("sum")
    src = pool_src[d.pj.values]
    d["_hi_s3"] = ((d.p > 0.5) & (src == "S3")).astype(np.float32)
    d["_hi_s2"] = ((d.p > 0.5) & (src == "S2")).astype(np.float32)
    d["s2_nhi_same_src"] = np.where(src == "S3", d.groupby("qi")._hi_s3.transform("sum"),
                                    d.groupby("qi")._hi_s2.transform("sum"))
    if "s2_nclaim_pj" not in d:            # pool-record features need every S1's claims; chunked callers
        d = add_pj_features(d)             # compute them once up front with add_pj_features
    first = d[d.s2_rank_q == 0].set_index("qi").pj
    sec = d[d.s2_rank_q == 1].set_index("qi").pj
    ref = np.where(d.s2_rank_q.values == 0, d.qi.map(sec).fillna(-1).values, d.qi.map(first).values).astype(np.int64)
    sn, sa, su = [], [], []
    for pj, r in zip(d.pj.values, ref):
        if r < 0:
            sn.append(np.nan)
            sa.append(np.nan)
            su.append(np.nan)
            continue
        sn.append(_jac(_tri(pool_nname[pj]), _tri(pool_nname[r])))
        sa.append(_jac(_tri(pool_naddr[pj]), _tri(pool_naddr[r])) if pool_naddr[pj] and pool_naddr[r] else np.nan)
        a, b = pool_nums[pj], pool_nums[r]
        su.append(float(bool(a & b)) if a and b else np.nan)
    d["s2_sib_name_j"] = np.array(sn, np.float32)
    d["s2_sib_addr_j"] = np.array(sa, np.float32)
    d["s2_sib_num"] = np.array(su, np.float32)
    return d.drop(columns=["_hi", "_hi_s3", "_hi_s2"])


S2_COLS = ["p", "s2_rank_q", "s2_pmax_q", "s2_prel", "s2_psum_q", "s2_nhi_q", "s2_nhi_same_src", "s2_nclaim_pj",
           "s2_rank_pj", "s2_pmax_other_pj", "s2_sib_name_j", "s2_sib_addr_j", "s2_sib_num"]


REL_COLS = ["cos", "name_jw", "name_tsr", "addr_tsr", "core_idf_q", "addr_idf_c", "num_j", "name_tri_idfcos",
            "addr_tri_idfcos"]
S2X_COLS = ["x_emb_wmean", "x_emb_max_hi", "x_emb_n_close_hi", "x_emb_max_hi_other", "x_emb_sib_top"] + \
           [f"rel_{c}" for c in REL_COLS]


def _group_pairs(qi):
    """All ordered (i, j) row pairs, i != j, inside each run of equal qi (rows must be sorted by qi)."""
    n = len(qi)
    starts = np.flatnonzero(np.r_[True, qi[1:] != qi[:-1]])
    sizes = np.diff(np.r_[starts, n])
    gs = np.repeat(sizes, sizes)
    gst = np.repeat(starts, sizes)
    I = np.repeat(np.arange(n), gs)
    off = np.arange(len(I)) - np.repeat(np.r_[0, np.cumsum(gs)[:-1]], gs)
    J = gst[I] + off
    k = I != J
    return I[k], J[k], np.arange(n) - gst


def stage2_extra(d, pool_emb, pool_src, chunk_rows=1_500_000, pool_core=None):
    """Cluster features from the fine-tuned embeddings of the S1's candidates (true matches of one S1 are the
    S2/S3 copies of one business and sit close together; look-alikes form other clusters) + each raw signal
    relative to the best candidate of the same S1. d must be sorted by (qi, p desc) with a 0..n-1 index."""
    import torch
    for c in REL_COLS:
        if c in d:
            d[f"rel_{c}"] = (d[c] - d.groupby("qi")[c].transform("max")).astype(np.float32)
    n = len(d)
    qi, pj, p = d.qi.values, d.pj.values, d.p.values.astype(np.float32)
    src = pool_src[pj]
    out = {c: np.full(n, np.nan, np.float32) for c in S2X_COLS[:5] + (["x_sib_core_hi"] if pool_core is not None else [])}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    starts = np.flatnonzero(np.r_[True, qi[1:] != qi[:-1]])
    bounds = [0]
    for s in starts[1:]:                                   # chunk on group boundaries
        if s - bounds[-1] >= chunk_rows:
            bounds.append(s)
    bounds.append(n)
    for a, b in zip(bounds[:-1], bounds[1:]):
        I, J, rank = _group_pairs(qi[a:b])
        up, inv = np.unique(pj[a:b], return_inverse=True)
        E = torch.from_numpy(np.ascontiguousarray(pool_emb[up], dtype=np.float16)).to(dev)
        cos = np.empty(len(I), np.float32)
        for s in range(0, len(I), 4_000_000):
            ii = torch.from_numpy(inv[I[s:s + 4_000_000]]).to(dev)
            jj = torch.from_numpy(inv[J[s:s + 4_000_000]]).to(dev)
            cos[s:s + len(ii)] = (E[ii].float() * E[jj].float()).sum(1).cpu().numpy()
        del E
        pl, sl, m = p[a:b], src[a:b], b - a
        w = pl[J]
        ws = np.bincount(I, weights=w, minlength=m)
        wc = np.bincount(I, weights=w * cos, minlength=m)
        out["x_emb_wmean"][a:b] = np.where(ws > 0, wc / np.maximum(ws, 1e-9), np.nan)
        hi = w > 0.5
        mx = pd.Series(cos[hi]).groupby(I[hi]).max()
        out["x_emb_max_hi"][a + mx.index.values] = mx.values
        out["x_emb_n_close_hi"][a:b] = np.bincount(I[hi & (cos > 0.9)], minlength=m)
        ot = hi & (sl[J] != sl[I])
        mo = pd.Series(cos[ot]).groupby(I[ot]).max()
        out["x_emb_max_hi_other"][a + mo.index.values] = mo.values
        sib = rank[J] == np.where(rank[I] == 0, 1, 0)
        out["x_emb_sib_top"][a + I[sib]] = cos[sib]
        if pool_core is not None:      # high-p siblings carrying exactly the same core name (empty-address copies)
            cc = pool_core[pj[a:b]]
            same = hi & (cc[I] == cc[J])
            out["x_sib_core_hi"][a:b] = np.bincount(I[same], minlength=m)
    for c, v in out.items():
        d[c] = v
    return d


# ----------------------------------------------------------------------------- decision
def one_to_one(d):
    """Each pool record goes to at most one S1 (its highest-probability claimant); 0 violations in train truth."""
    return d.sort_values(["pj", "p"], ascending=[True, False]).drop_duplicates("pj", keep="first")


def decide(d, t):
    return one_to_one(d[d.p >= t])


def prep_rules(d):
    """One-to-one, then sort by (qi, p desc) once and precompute rank / best p / expected-F helpers."""
    s = one_to_one(d)[["qi", "pj", "p"]].sort_values(["qi", "p"], ascending=[True, False]).reset_index(drop=True)
    g = s.groupby("qi")
    s["_r"] = g.cumcount().values
    s["_mx"] = g.p.transform("max").values
    s["_S"] = g.p.cumsum().values
    s["_T"] = g.p.transform("sum").values
    s["_l"] = np.log1p(-np.clip(s.p.values, 0, 1 - 1e-6))
    s["_L"] = s.groupby("qi")._l.transform("sum").values
    return s


def rule_threshold(s, t):
    return s[s.p.values >= t]


def rule_top1_plus(s, t1, t2):
    """Best candidate of an S1 if p >= t1 (a lower bar: predicting nothing for a non-singleton scores 0),
    further candidates if p >= t2."""
    r, p = s._r.values, s.p.values
    return s[((r == 0) & (p >= t1)) | ((r > 0) & (p >= t2))]


def rule_relative(s, t, r):
    p = s.p.values
    return s[(p >= t) & (p >= r * s._mx.values)]


def rule_expected_f(s, c_miss):
    """Per S1 keep the top-m candidates maximising the expected F0.5 (singleton option = predict nothing)."""
    m = s._r.values + 1
    ef = 1.25 * s._S.values / (0.25 * (s._T.values + c_miss) + m)
    ef0 = np.exp(s._L.values - c_miss)
    t = pd.DataFrame({"qi": s.qi.values, "ef": ef, "m": m})
    best = t.groupby("qi").ef.transform("max").values
    bm = t[ef == best].groupby("qi").m.min()
    keep = (m <= t.qi.map(bm).values) & (best > ef0)
    return s[keep]


def rule_gated(s, tE, t1, t2):
    """top1_plus, but only for S1 whose has-a-match probability _e >= tE (protects singletons)."""
    r, p, e = s._r.values, s.p.values, s._e.values
    return s[(e >= tE) & (((r == 0) & (p >= t1)) | ((r > 0) & (p >= t2)))]


ENT_TOP = ["p", "cos", "gap_q", "rev_rank", "rev_margin", "name_jw", "name_tri_idfcos", "addr_tri_idfcos", "num_share",
           "num_conflict", "core_idf_q", "addr_tsr", "s2_nclaim_pj", "s2_pmax_other_pj", "unshared_max_idf_q",
           "addr_unshared_max_idf_q"]


def entity_features(d, pcol="p2"):
    """One row per S1 (index qi): stage-2 score profile of its candidates + features of its best candidate."""
    g = d.sort_values(["qi", pcol], ascending=[True, False])
    g = g.assign(_hi=(g[pcol] > 0.5).astype(np.float32), _mid=(g[pcol] > 0.2).astype(np.float32))
    gb = g.groupby("qi")
    rank = gb.cumcount().values
    ent = pd.DataFrame({"e_pmax": gb[pcol].max(), "e_psum": gb[pcol].sum(), "e_ncand": gb.size().astype(np.float32),
                        "e_nhi": gb._hi.sum(), "e_nmid": gb._mid.sum(),
                        "e_p2nd": g[rank == 1].set_index("qi")[pcol], "e_cosmax": gb.cos.max()})
    top = g[rank == 0].set_index("qi")
    for c in ENT_TOP:
        if c in top:
            ent[f"e_top_{c}"] = top[c]
    return ent.astype(np.float32)


ENT_COLS = ["e_pmax", "e_psum", "e_ncand", "e_nhi", "e_nmid", "e_p2nd", "e_cosmax"] + [f"e_top_{c}" for c in ENT_TOP]


RULES = {
    "gated": (rule_gated, [(tE, t1, t2) for tE in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
                           for t1 in (0.2, 0.3, 0.4, 0.5, 0.6) for t2 in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85)]),
    "threshold": (rule_threshold, [(t,) for t in np.round(np.arange(0.3, 0.96, 0.025), 3)]),
    "top1_plus": (rule_top1_plus, [(t1, t2) for t1 in np.round(np.arange(0.1, 0.8, 0.1), 1)
                                   for t2 in np.round(np.arange(0.5, 0.96, 0.05), 2) if t2 >= t1]),
    "relative": (rule_relative, [(t, r) for t in np.round(np.arange(0.3, 0.9, 0.05), 2) for r in (0.5, 0.7, 0.8)]),
    "expected_f": (rule_expected_f, [(c,) for c in (0.0, 0.05, 0.1, 0.2)]),
}


def tune_rule(d, own, k, mask, rule, prepared=None):
    """Best parameters of a decision rule on the S1 rows in mask."""
    fn, grid = RULES[rule]
    s = prep_rules(d) if prepared is None else prepared
    best, best_f = None, -1.0
    for params in grid:
        kept = fn(s, *params)
        f = macro_f05(kept.qi.values, kept.pj.values, own, k, mask)["f05"]
        if f > best_f:
            best, best_f = tuple(float(x) for x in params), f
    return best, best_f


def apply_rule(d, rule, params, prepared=None):
    s = prep_rules(d) if prepared is None else prepared
    return RULES[rule][0](s, *params)


def tune_threshold(d, own, k, mask, grid=None):
    grid = np.round(np.arange(0.1, 0.97, 0.025), 3) if grid is None else grid
    best_t, best_f = None, -1.0
    o = one_to_one(d)
    for t in grid:
        kept = o[o.p >= t]
        f = macro_f05(kept.qi.values, kept.pj.values, own, k, mask)["f05"]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f
