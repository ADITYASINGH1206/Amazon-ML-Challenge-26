#!/usr/bin/env python3
"""Final-F0.5 cost of replacing the exact search by the IVF index, on the held-out slice B.
Runs the IVF search (forward top-K_FWD for B's S1, reverse top-M_REV for every pool record of the country, visible S1
only), keeps the v4 B candidates that IVF also finds (pairs only IVF finds are ignored, so this is a slight lower
bound), then: cheap filter -> saved full matcher -> stage 2 retrained out of fold -> rules tuned on folds 0-2,
reported on folds 3-4. Same procedure as size_curve.py, whose t_cheap=0.005 row is the exact-search reference."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from ber import config as C
from ber.knn import topk_ivf
from ber.util import Tee, load_json, log, save_json

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def ivf_pairs(E1, Ep, cty, pcty, qrows, present, nprobe):
    """(qi, pj) keys found by the IVF search: forward for qrows, reverse for every pool record."""
    keys = []
    for c in pd.unique(cty[qrows]):
        pi = np.where(pcty == c)[0]
        qn = qrows[cty[qrows] == c]
        qa = np.where((cty == c) & present)[0]
        fi, _ = topk_ivf(np.asarray(E1[qn]), np.asarray(Ep[pi]), C.K_FWD, nprobe=nprobe)
        ok = fi >= 0
        keys.append(np.repeat(qn, fi.shape[1])[ok.ravel()] * 100_000_000 + pi[fi[ok]])
        ri, _ = topk_ivf(np.asarray(Ep[pi]), np.asarray(E1[qa]), C.M_REV, nprobe=nprobe)
        ok = ri >= 0
        q = qa[ri[ok]]
        p = np.repeat(pi, ri.shape[1])[ok.ravel()]
        m = np.isin(q, qn)
        keys.append(q[m] * 100_000_000 + p[m])
        log(f"  [{c}] IVF nprobe {nprobe}: forward {len(qn):,} x {len(pi):,}, reverse {len(pi):,} x {len(qa):,}")
    return np.unique(np.concatenate(keys))


def main():
    import lightgbm as lgb
    import run_pipeline as R
    from ber.data import load_split, owner_index
    from ber.model import add_pj_features, apply_rule, fit_lgb, macro_f05, predict, tune_rule
    from ber.prep import load_norm
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--nprobe", default="64,256")
    ap.add_argument("--t-cheap", default="0.005,0.02")
    ap.add_argument("--n-jobs", type=int, default=24)
    a = ap.parse_args()
    W = a.work
    sys.stdout = Tee(os.path.join(W, "tune", "ann_impact.log"))
    dec = load_json(os.path.join(W, "models", "decision.json"))
    s1, pool = load_split(W, "train")
    own, k = owner_index(s1, pool), s1.k.values.astype(np.int64)
    fold, prio, cty, pcty = s1.fold.values, s1.prio.values, s1.country.values, pool.country.values
    inB = (prio >= C.TRAIN_B[0]) & (prio < C.TRAIN_B[1])
    lo, hi = C.TRAIN_HIDDEN_S1
    present = ~((prio >= lo) & (prio < hi))
    tune, hold = inB & (fold < 3), inB & (fold >= 3)
    E1 = np.load(os.path.join(W, "emb", "train_s1.npy"), mmap_mode="r")
    Ep = np.load(os.path.join(W, "emb", "train_pool.npy"), mmap_mode="r")
    _, npl, _ = load_norm(W, "train")
    emb = Ep
    F = pd.read_parquet(os.path.join(W, "feats", "train.parquet"))
    FB = F[inB[F.qi.values]].reset_index(drop=True)
    del F
    cheap = lgb.Booster(model_file=os.path.join(W, "models", "cheap.lgb"))
    full = lgb.Booster(model_file=os.path.join(W, "models", "full.lgb"))
    FB["pc"] = cheap.predict(FB[C.CHEAP].values.astype(np.float32)).astype(np.float32)
    FB["p"] = full.predict(FB[dec["cols1"]].values.astype(np.float32)).astype(np.float32)
    fkey = FB.qi.values * 100_000_000 + FB.pj.values
    n_b, n_true = int(inB.sum()), int(k[inB].sum())
    out = []
    for npb in [int(x) for x in a.nprobe.split(",")]:
        t0 = time.time()
        found = np.isin(fkey, ivf_pairs(E1, Ep, cty, pcty, np.where(inB)[0], present, npb))
        log(f"nprobe {npb}: IVF finds {found.mean():.4f} of the exact B candidates, "
            f"true-pair recall {(FB.label.values[found] == 1).sum() / n_true:.5f} "
            f"(exact {(FB.label.values == 1).sum() / n_true:.5f}), search {time.time() - t0:.0f}s")
        for t in [float(x) for x in a.t_cheap.split(",")]:
            sv = FB[found & (FB.pc.values >= t)].copy()
            d = R.stage2_frame(add_pj_features(sv), pool, npl.nname.values, npl.naddr.values, emb,
                               dec.get("use_s2x", False))
            X, y, q = d[dec["cols2"]].values.astype(np.float32), d.label.values, d.qi.values
            p2 = np.zeros(len(d), np.float32)
            for f in range(5):
                te = fold[q] == f
                p2[te] = predict(fit_lgb(X[~te], y[~te], q[~te], C.LGB_STAGE2, a.n_jobs), X[te])
            dd = d[["qi", "pj"]].assign(p=p2)
            params, _ = tune_rule(dd, own, k, tune, "top1_plus")
            rep = macro_f05(*apply_rule(dd, "top1_plus", params)[["qi", "pj"]].values.T, own, k, hold, cty)
            row = {"nprobe": npb, "t_cheap": t, "cand_per_S1": len(sv) / n_b, "top1_plus": round(rep["f05"], 5),
                   "US": round(rep["f05_US"], 5), "India": round(rep["f05_India"], 5)}
            log(f"ann impact: {row}")
            out.append(row)
    save_json(out, os.path.join(W, "tune", "ann_impact.json"))
    log("ANN IMPACT DONE")


if __name__ == "__main__":
    main()
