"""Decision layer: turn per-pair probabilities into final per-S1 match lists.
All strategies take a DataFrame with columns qi, pj, p and return the kept rows."""
import numpy as np
import pandas as pd

from .metrics import macro_f05


def thr(d, t):
    return d[d.p >= t]


def one_to_one(d):
    """Each pool record goes to at most one S1 (its highest-probability claimant)."""
    s = d.sort_values(["pj", "p"], ascending=[True, False])
    return s.drop_duplicates("pj", keep="first")


def relative(d, t, r):
    mx = d.groupby("qi").p.transform("max")
    return d[(d.p >= t) & (d.p >= r * mx)]


def expected_f05(d, c_miss=0.0):
    """Per S1, keep the top-m candidates (by p) that maximise expected F0.5:
         E[F](m) ~= 1.25 * sum_{i<=m} p_i / (0.25 * E|T| + m),  E|T| = sum_all p + c_miss
         E[F](0)  = P(no true match) ~= prod(1 - p_i) * exp(-c_miss)
    c_miss = expected number of true matches the candidate list missed (from blocking recall)."""
    if len(d) == 0:
        return d
    s = d.sort_values(["qi", "p"], ascending=[True, False]).copy()
    g = s.groupby("qi")
    s["_m"] = g.cumcount() + 1
    s["_S"] = g.p.cumsum()
    ET = g.p.transform("sum") + c_miss
    s["_ef"] = 1.25 * s._S / (0.25 * ET + s._m)
    s["_l"] = np.log1p(-np.clip(s.p.values, 0, 1 - 1e-6))
    ef0 = np.exp(s.groupby("qi")._l.transform("sum") - c_miss)
    best = s.groupby("qi")._ef.transform("max")
    best_m = s[s._ef == best].groupby("qi")._m.min()
    s["_bm"] = s.qi.map(best_m)
    keep = (s._m <= s._bm) & (best > ef0)
    return s[keep].drop(columns=["_m", "_S", "_ef", "_bm", "_l"])


def top1_plus(d, t1, t2):
    """Take the best candidate if p >= t1; add further candidates only if p >= t2 (t2 >= t1 typically)."""
    s = d.sort_values(["qi", "p"], ascending=[True, False])
    r = s.groupby("qi").cumcount()
    return s[((r == 0) & (s.p >= t1)) | ((r > 0) & (s.p >= t2))]


def tune_and_eval(d, fn, grid, own, k, q_fold, countries, tune_folds=(0, 1, 2)):
    """Pick the parameters maximising macro F0.5 on S1 entities of tune_folds, report on the other folds
    (honest) and on all folds. Returns (best_params, report_dict)."""
    tune_mask = np.isin(q_fold, tune_folds)
    hold_mask = ~tune_mask
    best, best_f = None, -1
    for params in grid:
        kept = fn(d, *params)
        f = macro_f05(kept.qi.values, kept.pj.values, own, k, mask=tune_mask)["f05"]
        if f > best_f:
            best, best_f = params, f
    kept = fn(d, *best)
    hold = macro_f05(kept.qi.values, kept.pj.values, own, k, countries, mask=hold_mask)
    allr = macro_f05(kept.qi.values, kept.pj.values, own, k, countries)
    rep = {"params": list(best), "f05_tune": best_f, "f05_holdout": hold["f05"], "f05_all": allr["f05"],
           **{f"hold_{kk}": v for kk, v in hold.items() if kk != "f05"}}
    return best, rep


def isotonic_oof(d, q_fold):
    """Calibrate p with isotonic regression fitted on the other folds (per fold)."""
    from sklearn.isotonic import IsotonicRegression
    out = d.p.values.copy()
    f = q_fold[d.qi.values]
    for k in np.unique(f):
        tr = f != k
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(d.p.values[tr], d.y.values[tr])
        out[~tr] = ir.predict(d.p.values[~tr])
    return out
