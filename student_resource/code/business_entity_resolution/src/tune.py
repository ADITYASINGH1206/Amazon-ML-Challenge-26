#!/usr/bin/env python3
"""Measure candidate improvements on the held-out slice B (tuned on folds 0-2, reported on folds 3-4), cheapest
first. Uses the training candidates/features produced by run_pipeline.py (block + feats stages).

  old    the previously submitted models, re-scored on the density-matched B (should mirror the leaderboard drop)
  base   cheap + full matcher retrained on the density-matched A, stage 2, threshold rule
  rules  decision rules on the base stage-2 probabilities
  s2x    stage 2 + embedding-cluster and relative-to-best features
  v2     full matcher + V2 pair features (trigram TF-IDF cosine, name specificity, rare-token mismatch)
  big    larger LightGBM trees for the best feature set
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from ber import config as C
from ber.util import Tee, log, save_json

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


class Ctx:
    def __init__(self, work, n_jobs):
        from ber.data import load_split, owner_index
        from ber.prep import load_norm
        self.work, self.n_jobs = work, n_jobs
        s1, pool = load_split(work, "train")
        self.own = owner_index(s1, pool)
        self.k = s1.k.values.astype(np.int64)
        self.fold, self.cty, prio = s1.fold.values, s1.country.values, s1.prio.values
        self.inA = (prio >= C.TRAIN_A[0]) & (prio < C.TRAIN_A[1])
        self.inB = (prio >= C.TRAIN_B[0]) & (prio < C.TRAIN_B[1])
        self.tune = self.inB & (self.fold < 3)
        self.hold = self.inB & (self.fold >= 3)
        self.pool_src = pool.src.values
        self.pool_addr = pool.addr.values
        _, npl, _ = load_norm(work, "train")
        self.pn, self.pa = npl.nname.values, npl.naddr.values
        self.emb = np.load(os.path.join(work, "emb", "train_pool.npy"), mmap_mode="r")
        F = pd.read_parquet(os.path.join(work, "feats", "train.parquet"))
        self.FA = F[self.inA[F.qi.values]].reset_index(drop=True)
        self.FB = F[self.inB[F.qi.values]].reset_index(drop=True)
        del F
        log(f"A {len(self.FA):,} pairs, B {len(self.FB):,} pairs")

    def f05(self, kept, mask):
        from ber.model import macro_f05
        return macro_f05(kept.qi.values, kept.pj.values, self.own, self.k, mask, self.cty)


def evaluate(ctx, d, rules=("threshold",)):
    """d: qi, pj, p. Tune each rule on folds 0-2 of B, report on folds 3-4."""
    from ber.model import apply_rule, tune_rule
    res = {}
    for r in rules:
        params, _ = tune_rule(d, ctx.own, ctx.k, ctx.tune, r)
        rep = ctx.f05(apply_rule(d, r, params), ctx.hold)
        res[r] = {"params": list(params), **{k: rep[k] for k in ("f05", "micro_precision", "micro_recall",
                                                                  "singleton_false_merge_rate", "f05_US", "f05_India")}}
    return res


def stage2_oof(ctx, sv, cols1, extra):
    """sv: B survivors with stage-1 p. Returns (qi, pj, p2) with 5-fold out-of-fold stage-2 predictions."""
    from ber.model import S2_COLS, S2X_COLS, fit_lgb, predict, stage2_extra, stage2_features
    from ber.text import numbers
    nums = {int(j): numbers(ctx.pool_addr[j]) for j in np.unique(sv.pj.values)}
    d = stage2_features(sv, ctx.pn, ctx.pa, nums, ctx.pool_src)
    cols = S2_COLS + cols1
    if extra:
        d = stage2_extra(d, ctx.emb, ctx.pool_src)
        cols = cols + [c for c in S2X_COLS if c in d]
    X, y, q = d[cols].values.astype(np.float32), d.label.values, d.qi.values
    fq = ctx.fold[q]
    p2 = np.zeros(len(d), np.float32)
    for f in range(5):
        te = fq == f
        m = fit_lgb(X[~te], y[~te], q[~te], C.LGB_STAGE2, ctx.n_jobs)
        p2[te] = predict(m, X[te])
    return d[["qi", "pj"]].assign(p=p2)


def run(ctx, cheap, cols1, params, extra, rules=("threshold",)):
    from ber.model import fit_lgb, predict
    t0 = time.time()
    full = fit_lgb(ctx.FA[cols1].values.astype(np.float32), ctx.FA.label.values, ctx.FA.qi.values, params,
                   ctx.n_jobs)
    sv = ctx.FB[ctx.FB.pc >= ctx.t_cheap].copy()
    sv["p"] = predict(full, sv[cols1].values.astype(np.float32))
    s1 = evaluate(ctx, sv[["qi", "pj", "p"]])
    d2 = stage2_oof(ctx, sv, cols1, extra)
    s2 = evaluate(ctx, d2, rules)
    return {"stage1": s1["threshold"], "stage2": s2, "minutes": (time.time() - t0) / 60,
            "best_iteration": full.best_iteration}, d2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--steps", default="old,base,rules,s2x,v2,big")
    ap.add_argument("--n-jobs", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.work, "tune"), exist_ok=True)
    sys.stdout = Tee(os.path.join(a.work, "tune", "tune.log"))
    steps = a.steps.split(",")
    from ber.model import blocking_stats, fit_lgb, predict
    import lightgbm as lgb
    ctx = Ctx(a.work, a.n_jobs)
    R = {}
    out = os.path.join(a.work, "tune", "results.json")

    def show(name, r):
        R[name] = r
        save_json(R, out)
        s2 = r["stage2"] if "stage2" in r else r
        best = max(s2.items(), key=lambda kv: kv[1]["f05"]) if isinstance(s2, dict) and "threshold" in s2 else None
        log(f"RESULT {name}: stage1 F0.5={r.get('stage1', {}).get('f05', float('nan')):.4f}  "
            + ("  ".join(f"{k}: {v['f05']:.4f}" for k, v in s2.items()) if best else ""))

    R["blocking_B"] = blocking_stats(ctx.FB.qi.values, ctx.FB.pj.values, ctx.own, ctx.k, ctx.inB)
    log(f"B blocking with hidden S1 (test-like density): {R['blocking_B']}")

    if "old" in steps:
        mdir = os.path.join(a.work, "models_v1")
        cheap = lgb.Booster(model_file=os.path.join(mdir, "cheap.lgb"))
        full = lgb.Booster(model_file=os.path.join(mdir, "full.lgb"))
        sv = ctx.FB[cheap.predict(ctx.FB[C.CHEAP].values.astype(np.float32)) >= 0.005].copy()
        sv["p"] = full.predict(sv[C.FULL].values.astype(np.float32)).astype(np.float32)
        r = evaluate(ctx, sv[["qi", "pj", "p"]])
        kept = sv[sv.p >= 0.7].sort_values(["pj", "p"], ascending=[True, False]).drop_duplicates("pj")
        r["v1_threshold_as_submitted"] = {"f05": ctx.f05(kept, ctx.hold)["f05"]}
        show("old_models_on_new_B", {"stage2": {}, "stage1": r["threshold"], "as_submitted": r["v1_threshold_as_submitted"]})
        log(f"  v1 stage-1 at its old threshold 0.7: {r['v1_threshold_as_submitted']['f05']:.4f}")

    log("cheap re-ranker on the density-matched A")
    cheap = fit_lgb(ctx.FA[C.CHEAP].values.astype(np.float32), ctx.FA.label.values, ctx.FA.qi.values, C.LGB_PARAMS,
                    a.n_jobs)
    ctx.FB["pc"] = predict(cheap, ctx.FB[C.CHEAP].values.astype(np.float32))
    ctx.t_cheap = C.CHEAP_THRESHOLDS[0]
    for t in C.CHEAP_THRESHOLDS:
        s = ctx.FB[ctx.FB.pc >= t]
        st = blocking_stats(s.qi.values, s.pj.values, ctx.own, ctx.k, ctx.inB)
        if st["pair_recall"] >= R["blocking_B"]["pair_recall"] - C.CHEAP_MAX_RECALL_LOSS:
            ctx.t_cheap, R["candidates_B"] = t, st
    log(f"cheap filter p >= {ctx.t_cheap}: {R['candidates_B']}")

    has_v2 = all(c in ctx.FA for c in C.PAIR_V2)
    extra = False
    if "base" in steps:
        r, d2 = run(ctx, cheap, C.FULL, C.LGB_PARAMS, extra=False, rules=tuple(
            ["threshold"] + (["top1_plus", "relative", "expected_f"] if "rules" in steps else [])))
        show("base", r)
    if "s2x" in steps:
        r, _ = run(ctx, cheap, C.FULL, C.LGB_PARAMS, extra=True)
        show("s2x", r)
        extra = r["stage2"]["threshold"]["f05"] > R["base"]["stage2"]["threshold"]["f05"]
    cols = C.FULL
    if "v2" in steps and has_v2:
        r, _ = run(ctx, cheap, C.FULL + C.PAIR_V2, C.LGB_PARAMS, extra=extra)
        show("v2" + ("+s2x" if extra else ""), r)
        best_prev = max(v["stage2"]["threshold"]["f05"] for k, v in R.items() if isinstance(v, dict) and "stage2" in v
                        and "threshold" in v["stage2"] and not k.startswith("v2"))
        if r["stage2"]["threshold"]["f05"] > best_prev:
            cols = C.FULL + C.PAIR_V2
    if "big" in steps:
        big = dict(C.LGB_PARAMS, num_leaves=255, min_data_in_leaf=100)
        r, _ = run(ctx, cheap, cols, big, extra=extra, rules=("threshold", "top1_plus", "relative", "expected_f"))
        show("big", r)
    log("DONE -- " + ", ".join(f"{k}={v['stage2']['threshold']['f05']:.4f}" for k, v in R.items()
                               if isinstance(v, dict) and "stage2" in v and "threshold" in v.get("stage2", {})))


if __name__ == "__main__":
    main()
