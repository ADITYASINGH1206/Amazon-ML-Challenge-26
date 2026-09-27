"""Scoring: exact macro F0.5 (as defined by the challenge) and blocking quality statistics.
Everything works on integer indices: qi = row in the universe S1 table, pj = row in the universe pool table."""
import numpy as np
import pandas as pd


def owner_index(s1, pool):
    """For every pool row, the S1 row index of its true owner (-1 for distractors)."""
    m = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    return pool.owner.map(m).fillna(-1).astype(np.int64).values


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


def macro_f05(qi, pj, own, k, countries=None, mask=None):
    f, npred, tp = per_entity_f05(qi, pj, own, k)
    m = np.ones(len(k), bool) if mask is None else mask
    res = {
        "f05": float(f[m].mean()),
        "f05_singletons": float(f[m & (k == 0)].mean()) if (m & (k == 0)).any() else np.nan,
        "f05_nonsingletons": float(f[m & (k > 0)].mean()) if (m & (k > 0)).any() else np.nan,
        "micro_precision": float(tp[m].sum() / max(npred[m].sum(), 1)),
        "micro_recall": float(tp[m].sum() / max(k[m].sum(), 1)),
        "avg_pred_per_s1": float(npred[m].mean()),
        "singleton_false_merge_rate": float((npred[m & (k == 0)] > 0).mean()) if (m & (k == 0)).any() else np.nan,
    }
    if countries is not None:
        for c in pd.unique(countries[m]):
            mc = m & (countries == c)
            res[f"f05_{c}"] = float(f[mc].mean())
    return res


def blocking_stats(qi, pj, own, k, countries=None, mask=None):
    qi, pj = _dedupe(qi, pj)
    n = len(k)
    m = np.ones(n, bool) if mask is None else mask
    size = np.bincount(qi, minlength=n)
    tp = np.bincount(qi[own[pj] == qi], minlength=n) if len(qi) else np.zeros(n, np.int64)
    ceil = np.where(k == 0, 1.0, np.where(tp == 0, 0.0, 1.25 * tp / (0.25 * k + np.maximum(tp, 1))))
    s = size[m]
    res = {
        "pair_recall": float(tp[m].sum() / max(k[m].sum(), 1)),
        "entity_full_recall": float((tp[m & (k > 0)] == k[m & (k > 0)]).mean()) if (m & (k > 0)).any() else np.nan,
        "ceiling_f05": float(ceil[m].mean()),
        "cand_mean": float(s.mean()), "cand_p50": float(np.median(s)), "cand_p90": float(np.percentile(s, 90)),
        "cand_p99": float(np.percentile(s, 99)), "cand_max": int(s.max()) if len(s) else 0,
        "no_cand_rate": float((s == 0).mean()),
    }
    if countries is not None:
        for c in pd.unique(countries[m]):
            mc = m & (countries == c)
            res[f"recall_{c}"] = float(tp[mc].sum() / max(k[mc].sum(), 1))
            res[f"cand_mean_{c}"] = float(size[mc].mean())
    return res


def recall_from_idx(idx, own, k, ks, countries=None, pool_mask=None, prefix=""):
    """Recall@k of a (n_s1 x K) neighbour matrix (-1 = empty). pool_mask restricts to a subset of true pairs
    (e.g. matches whose name is in a native Indic script)."""
    n, K = idx.shape
    qi = np.repeat(np.arange(n), K)
    pj = idx.ravel()
    r = np.tile(np.arange(K), n)
    ok = pj >= 0
    qi, pj, r = qi[ok], pj[ok], r[ok]
    hit = own[pj] == qi
    rq, qq, pp = r[hit], qi[hit], pj[hit]
    tot = max(int(k.sum()), 1)
    res = {f"{prefix}R@{kk}": float((rq < kk).sum() / tot) for kk in ks if kk <= K}
    if countries is not None:
        for c in pd.unique(countries):
            mc = countries[qq] == c
            tc = max(int(k[countries == c].sum()), 1)
            res[f"{prefix}R@{ks[-1]}_{c}"] = float((rq[mc] < ks[-1]).sum() / tc)
            if 10 <= K:
                res[f"{prefix}R@10_{c}"] = float((rq[mc] < 10).sum() / tc)
    if pool_mask is not None:
        tm = max(int((pool_mask & (own >= 0)).sum()), 1)
        mm = pool_mask[pp]
        for kk in (1, 10, ks[-1]):
            if kk <= K:
                res[f"{prefix}native_R@{kk}"] = float((rq[mm] < kk).sum() / tm)
    return res


def recall_at_k_curve(df, own, k, score_col, ks=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50), countries=None):
    """Keep the top-k candidates of each S1 by score_col -> recall / size / ceiling for every k."""
    d = df.sort_values(["qi", score_col], ascending=[True, False])
    d = d.assign(_r=d.groupby("qi").cumcount())
    rows = []
    for kk in ks:
        s = d[d._r < kk]
        st = blocking_stats(s.qi.values, s.pj.values, own, k, countries)
        rows.append({"k": kk, **st})
    return rows
