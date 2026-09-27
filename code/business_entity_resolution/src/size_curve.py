#!/usr/bin/env python3
"""Candidate-set size vs final F0.5 on the held-out slice B. A smaller candidate set per S1 ranks higher in the
final evaluation, so this finds how far the cheap re-ranker's threshold can be raised before the score drops.
For each threshold: B survivors -> full matcher (the saved model) -> stage-2 features on exactly that set ->
5-fold out-of-fold stage 2 -> decision rules tuned on folds 0-2, reported on folds 3-4."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from ber import config as C
from ber.util import Tee, load_json, log, save_json

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def main():
    import lightgbm as lgb
    import run_pipeline as R
    from ber.data import load_split, owner_index
    from ber.model import add_pj_features, apply_rule, fit_lgb, macro_f05, predict, tune_rule
    from ber.prep import load_norm
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--thresholds", default="0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--rules", default="top1_plus,expected_f")
    ap.add_argument("--n-jobs", type=int, default=24)
    a = ap.parse_args()
    W = a.work
    sys.stdout = Tee(os.path.join(W, "tune", "size_curve.log"))
    dec = load_json(os.path.join(W, "models", "decision.json"))
    s1, pool = load_split(W, "train")
    own, k = owner_index(s1, pool), s1.k.values.astype(np.int64)
    fold, prio, cty = s1.fold.values, s1.prio.values, s1.country.values
    inB = (prio >= C.TRAIN_B[0]) & (prio < C.TRAIN_B[1])
    tune, hold = inB & (fold < 3), inB & (fold >= 3)
    _, npl, _ = load_norm(W, "train")
    emb = np.load(os.path.join(W, "emb", "train_pool.npy"), mmap_mode="r")
    F = pd.read_parquet(os.path.join(W, "feats", "train.parquet"))
    FB = F[inB[F.qi.values]].copy()
    del F
    cheap = lgb.Booster(model_file=os.path.join(W, "models", "cheap.lgb"))
    full = lgb.Booster(model_file=os.path.join(W, "models", "full.lgb"))
    FB["pc"] = cheap.predict(FB[C.CHEAP].values.astype(np.float32)).astype(np.float32)
    FB["p"] = full.predict(FB[dec["cols1"]].values.astype(np.float32)).astype(np.float32)
    n_b = int(inB.sum())
    out = []
    for t in [float(x) for x in a.thresholds.split(",")]:
        t0 = time.time()
        sv = FB[FB.pc >= t].copy()
        tp = int((own[sv.pj.values] == sv.qi.values).sum())
        d = R.stage2_frame(add_pj_features(sv), pool, npl.nname.values, npl.naddr.values, emb,
                           dec.get("use_s2x", False))
        X, y, q = d[dec["cols2"]].values.astype(np.float32), d.label.values, d.qi.values
        p2 = np.zeros(len(d), np.float32)
        for f in range(5):
            te = fold[q] == f
            p2[te] = predict(fit_lgb(X[~te], y[~te], q[~te], C.LGB_STAGE2, a.n_jobs), X[te])
        dd = d[["qi", "pj"]].assign(p=p2)
        row = {"t_cheap": t, "cand_per_S1": len(sv) / n_b,
               "cand_p90": float(np.quantile(np.bincount(sv.qi.values, minlength=len(k))[inB], 0.9)),
               "pair_recall": tp / int(k[inB].sum())}
        for r in a.rules.split(","):
            params, _ = tune_rule(dd, own, k, tune, r)
            rep = macro_f05(*apply_rule(dd, r, params)[["qi", "pj"]].values.T, own, k, hold, cty)
            row[r] = round(rep["f05"], 5)
            row[f"{r}_US"], row[f"{r}_India"] = round(rep["f05_US"], 5), round(rep["f05_India"], 5)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        log(f"size curve: {row}")
        out.append(row)
        del sv, d, X, dd
    save_json(out, os.path.join(W, "tune", "size_curve.json"))
    log("SIZE CURVE DONE")


if __name__ == "__main__":
    main()
