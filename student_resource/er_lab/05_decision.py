#!/usr/bin/env python3
"""05 -- Decision layer: from pair probabilities to final match lists (this is what F0.5 actually scores).

Strategies (parameters tuned on S1 folds 0-2, reported on folds 3-4 = honest holdout):
  S0 global threshold
  S1 one-to-one (each pool record -> its best S1 only) + threshold          <- uses the 0-violation finding
  S2 one-to-one + relative threshold (keep p >= t and p >= r * best p of that S1)
  S3 one-to-one + top1-plus (best candidate if p >= t1, others only if p >= t2)
  S4 one-to-one + expected-F0.5 optimal list per S1 (no threshold; picks the list size that maximises E[F0.5])
  S5/S6 the same with isotonic-calibrated probabilities
Stage 2 (stacking): group-aware features computed from stage-1 OOF probabilities -- rank within the S1,
  share of the best, how many other strong candidates, competition for the same pool record, and SIBLING
  similarity (does this candidate look like the S1's strongest candidate? = your S2<->S3 agreement idea).
Also prints an error analysis (false merges / misses) so we can see what still breaks.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lab.common import base_parser, log_result, setup, step, udir
from lab.data import load_universe
from lab.decide import expected_f05, isotonic_oof, one_to_one, relative, thr, top1_plus, tune_and_eval
from lab.feats import GROUPS
from lab.metrics import blocking_stats, macro_f05, owner_index
from lab.text import load_dict, normalize_series


def grid_thr():
    return [(t,) for t in np.round(np.arange(0.1, 0.97, 0.025), 3)]


def strategies(d, own, k, q_fold, countries, args, tag, calibrated=None):
    out = {}
    runs = [
        ("S0_threshold", d, thr, grid_thr()),
        ("S1_one2one+threshold", one_to_one(d), thr, grid_thr()),
        ("S2_one2one+relative", one_to_one(d), relative,
         [(t, r) for t in np.arange(0.2, 0.91, 0.1) for r in (0.3, 0.5, 0.7, 0.8, 0.9)]),
        ("S3_one2one+top1plus", one_to_one(d), top1_plus,
         [(t1, t2) for t1 in np.arange(0.1, 0.81, 0.1) for t2 in np.arange(0.3, 0.96, 0.05) if t2 >= t1]),
        ("S4_one2one+expectedF", one_to_one(d), expected_f05, [(c,) for c in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)]),
    ]
    if calibrated is not None:
        dc = d.assign(p=calibrated)
        runs += [("S5_isotonic+one2one+threshold", one_to_one(dc), thr, grid_thr()),
                 ("S6_isotonic+one2one+expectedF", one_to_one(dc), expected_f05,
                  [(c,) for c in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)])]
    for name, dd, fn, grid in runs:
        t0 = time.time()
        best, rep = tune_and_eval(dd, fn, grid, own, k, q_fold, countries)
        rep["seconds"] = time.time() - t0
        log_result(args, "05_decision", f"{tag}|{name}", **rep)
        out[name] = (rep["f05_holdout"], fn, best, dd)
    return out


def stage2_features(d, pool_names, pool_addrs, pool_nums, pool_src):
    d = d.sort_values(["qi", "p"], ascending=[True, False]).reset_index(drop=True)
    g = d.groupby("qi")
    d["s2_rank_q"] = g.cumcount().astype(np.float32)
    d["s2_pmax_q"] = g.p.transform("max")
    d["s2_prel"] = d.p / (d.s2_pmax_q + 1e-9)
    d["s2_psum_q"] = g.p.transform("sum")
    d["s2_nhi_q"] = g.p.transform(lambda x: (x > 0.5).sum()).astype(np.float32)
    src = pool_src[d.pj.values]
    d["_hi_s3"] = ((d.p > 0.5) & (src == "S3")).astype(np.float32)
    d["_hi_s2"] = ((d.p > 0.5) & (src == "S2")).astype(np.float32)
    d["s2_nhi_same_src"] = np.where(src == "S3", g._hi_s3.transform("sum"), g._hi_s2.transform("sum"))
    gp = d.groupby("pj")
    d["s2_nclaim_pj"] = gp.p.transform("size").astype(np.float32)
    d["s2_rank_pj"] = gp.p.rank(ascending=False, method="first").astype(np.float32)
    top2 = gp.p.transform(lambda x: np.sort(x.values)[-2] if len(x) > 1 else 0.0)
    pmax_pj = gp.p.transform("max")
    d["s2_pmax_other_pj"] = np.where(d.p >= pmax_pj, top2, pmax_pj)
    # sibling similarity: compare with the S1's strongest other candidate
    first = d[d.s2_rank_q == 0].set_index("qi").pj
    second = d[d.s2_rank_q == 1].set_index("qi").pj
    ref = np.where(d.s2_rank_q.values == 0, d.qi.map(second).fillna(-1).values, d.qi.map(first).values)
    ref = ref.astype(np.int64)

    def tri(s):
        s = f" {s} "
        return {s[i:i + 3] for i in range(len(s) - 2)}

    def jac(a, b):
        u = len(a | b)
        return len(a & b) / u if u else np.nan

    sn, sa, su = [], [], []
    for pj, r in zip(d.pj.values, ref):
        if r < 0:
            sn.append(np.nan)
            sa.append(np.nan)
            su.append(np.nan)
            continue
        sn.append(jac(tri(pool_names[pj]), tri(pool_names[r])))
        sa.append(jac(tri(pool_addrs[pj]), tri(pool_addrs[r])) if pool_addrs[pj] and pool_addrs[r] else np.nan)
        a, b = pool_nums[pj], pool_nums[r]
        su.append(float(bool(a & b)) if a and b else np.nan)
    d["s2_sib_name_j"] = np.array(sn, np.float32)
    d["s2_sib_addr_j"] = np.array(sa, np.float32)
    d["s2_sib_num"] = np.array(su, np.float32)
    return d.drop(columns=["_hi_s3", "_hi_s2"])


def error_analysis(kept, d, s1, pool, own, k, mask, n=15):
    pred = set(zip(kept.qi.values, kept.pj.values))
    fp = kept[(own[kept.pj.values] != kept.qi.values) & mask[kept.qi.values]]
    print(f"\n--- FALSE MERGES (holdout): {len(fp):,} pairs; sample ---")
    for r in fp.sample(min(n, len(fp)), random_state=0).itertuples():
        q, c = s1.iloc[r.qi], pool.iloc[r.pj]
        true_owner = s1.iloc[own[r.pj]].entity_id if own[r.pj] >= 0 else "(distractor)"
        print(f"  p={r.p:.3f} S1 {q.entity_id} [{q.country}] {q['name'][:50]} | {q.addr[:80]}")
        print(f"          -> {c.entity_id} {c['name'][:50]} | {c.addr[:80]}   (true owner: {true_owner})")
    truth = pool[(own >= 0) & mask[np.clip(own, 0, None)]]
    tp_idx = np.where((own >= 0) & mask[np.clip(own, 0, None)])[0]
    miss = [j for j in tp_idx if (own[j], j) not in pred]
    cand_set = set(zip(d.qi.values, d.pj.values))
    print(f"\n--- MISSED MATCHES (holdout): {len(miss):,}; of which not even in candidates: "
          f"{sum((own[j], j) not in cand_set for j in miss):,}; sample ---")
    rng = np.random.default_rng(0)
    pmap = dict(zip(zip(d.qi.values, d.pj.values), d.p.values))
    for j in rng.choice(miss, size=min(n, len(miss)), replace=False) if miss else []:
        q, c = s1.iloc[own[j]], pool.iloc[j]
        pv = pmap.get((own[j], j))
        print(f"  p={'not-a-candidate' if pv is None else f'{pv:.3f}'} S1 {q.entity_id} [{q.country}] "
              f"{q['name'][:50]} | {q.addr[:80]}")
        print(f"          <- {c.entity_id} {c['name'][:50]} | {c.addr[:80]}")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--oofs", default="", help="comma list of OOF names (default: FULL, CHEAP, best)")
    ap.add_argument("--feats", default="feats_L4")
    ap.add_argument("--no-stage2", action="store_true")
    args = ap.parse_args()
    setup(args, "05_decision")
    s1, pool = load_universe(args)
    own = owner_index(s1, pool)
    k, q_fold, countries = s1.k.values, s1.fold.values, s1.country.values
    d0 = udir(args)
    best_name = open(os.path.join(d0, "oof_best.txt")).read().strip()
    full_name = "A_CTX+NAME+ADDR+NUM+COMP"
    names = args.oofs.split(",") if args.oofs else list(dict.fromkeys([full_name, best_name, "C_CHEAP"]))
    cands = pd.read_parquet(os.path.join(d0, "cands.parquet"), columns=["qi", "pj"])
    ceil = blocking_stats(cands.qi.values, cands.pj.values, own, k, countries)
    log_result(args, "05_decision", "UPPER_BOUND|perfect matcher on these candidates", f05_holdout=ceil["ceiling_f05"])

    results = {}
    for nm in names:
        path = os.path.join(d0, f"oof_{nm}.parquet")
        if not os.path.exists(path):
            continue
        d = pd.read_parquet(path)
        step(f"decision strategies on {nm}")
        cal = isotonic_oof(d, q_fold)
        for s, v in strategies(d, own, k, q_fold, countries, args, nm, cal).items():
            results[f"{nm}|{s}"] = v

    if not args.no_stage2:
        base_name = best_name if os.path.exists(os.path.join(d0, f"oof_{best_name}.parquet")) else full_name
        step(f"stage 2 stacking on top of {base_name}")
        d = pd.read_parquet(os.path.join(d0, f"oof_{base_name}.parquet"))
        dct = load_dict(os.path.join(args.work, "models", "translit_dict.json"))
        pn = normalize_series(pool.name.values, pool.country.values, 4, "name", dct)
        pa = normalize_series(pool.addr.values, pool.country.values, 4, "addr", dct)
        from lab.text import numbers
        pu = [numbers(a) for a in pool.addr.values]
        d = stage2_features(d, pn, pa, pu, pool.src.values)
        F = pd.read_parquet(os.path.join(d0, f"{args.feats}.parquet"))
        base_cols = [c for g in ("CTX", "NAME", "ADDR", "NUM", "COMP") for c in GROUPS[g] if c in F]
        d = d.merge(F[["qi", "pj", "fold"] + base_cols], on=["qi", "pj"], how="left")
        s2_cols = ["p"] + [c for c in d.columns if c.startswith("s2_")]
        for tag, cols in (("stage2_groupfeats_only", s2_cols), ("stage2_full+groupfeats", s2_cols + base_cols)):
            X = d[cols].values.astype(np.float32)
            y = d.y.values
            p2 = np.zeros(len(d), np.float32)
            for f in range(5):
                te = (d.fold.values % 5) == f
                inner = (~te) & ((d.qi.values % 10) == 0)
                fit_ = (~te) & ~inner
                m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                                   feature_fraction=0.9, verbose=-1, num_threads=args.n_jobs, seed=args.seed),
                              lgb.Dataset(X[fit_], y[fit_]), 2000 if not args.quick else 300,
                              valid_sets=[lgb.Dataset(X[inner], y[inner])],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
                p2[te] = m.predict(X[te], num_iteration=m.best_iteration)
            log_result(args, "05_stage2_model", tag, auc=roc_auc_score(y, p2), auc_stage1=roc_auc_score(y, d.p.values))
            d2 = d[["qi", "pj", "y"]].assign(p=p2)
            pd.DataFrame(d2).to_parquet(os.path.join(d0, f"oof_S2_{tag}.parquet"), index=False)
            for s, v in strategies(d2, own, k, q_fold, countries, args, tag, None).items():
                results[f"{tag}|{s}"] = v

    best = max(results, key=lambda x: results[x][0])
    f, fn, params, dd = results[best]
    step(f"BEST: {best}  holdout F0.5={f:.4f} params={params}")
    kept = fn(dd, *params)
    hold = ~np.isin(q_fold, (0, 1, 2))
    rep = macro_f05(kept.qi.values, kept.pj.values, own, k, countries, mask=hold)
    log_result(args, "05_best", best, params=list(params), **rep)
    error_analysis(kept, dd, s1, pool, own, k, hold)


if __name__ == "__main__":
    main()
