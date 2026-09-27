#!/usr/bin/env python3
"""04 -- Pairwise matcher (LightGBM) experiments on the candidate pairs of U1, 5-fold CV grouped by S1 entity.

A  Feature-group ladder: CTX -> +NAME -> +ADDR -> +NUM -> +COMP(=FULL) -> +EMB -> +CE (if 06/07 were run)
B  Leave-one-group-out from FULL, and FULL without blocking context (pure pair model)
C  CHEAP model (set-based features only, no fuzzy matching) = the candidate-reduction stage
D  Candidate-size curve: keep top-k / p>=t per S1 using CHEAP and FULL OOF scores -> recall vs list size.
   This decides what goes into candidate_pairs.tsv (the smaller, the better for the ranking).
E  Country transfer (proxy for France): train on US -> test India and vice-versa, threshold chosen on the source
F  Feature importance (gain)
Every config is scored with a quick decision (one-to-one + threshold tuned on folds 0-2, reported on 3-4).
OOF predictions of FULL, CHEAP and the best extended model are saved for 05_decision.py.
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from lab.common import base_parser, log_result, setup, step, udir
from lab.data import load_universe
from lab.decide import one_to_one, thr, tune_and_eval
from lab.feats import GROUPS
from lab.metrics import blocking_stats, macro_f05, owner_index, recall_at_k_curve

CHEAP_EXTRA = ["core_tok_j", "name_tok_j", "skel_j", "core_idf_q", "addr_tok_j", "addr_contain"]


def params(args):
    return dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=200,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                num_threads=args.n_jobs, seed=args.seed)


def fit(X, y, qi, args, rounds=None):
    inner = (qi % 10) == 0
    dtr = lgb.Dataset(X[~inner], y[~inner], free_raw_data=False)
    dva = lgb.Dataset(X[inner], y[inner], free_raw_data=False)
    return lgb.train(params(args), dtr, rounds or args.rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def cv_oof(df, cols, args):
    X = df[cols].values.astype(np.float32)
    y = df.label.values
    qi = df.qi.values
    fold = df.fold.values % args.folds
    oof = np.zeros(len(df), np.float32)
    model = None
    for f in range(args.folds):
        te = fold == f
        model = fit(X[~te], y[~te], qi[~te], args)
        oof[te] = model.predict(X[te], num_iteration=model.best_iteration)
    return oof, model


def quick_decision(df, p, own, k, q_fold, countries):
    d = pd.DataFrame({"qi": df.qi.values, "pj": df.pj.values, "p": p})
    d = one_to_one(d)
    _, rep = tune_and_eval(d, thr, [(t,) for t in np.arange(0.2, 0.96, 0.05)], own, k, q_fold % 5, countries)
    return rep


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--feats", default="feats_L4")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--only", default="", help="comma list of blocks to run: A,B,C,D,E,F (default all)")
    ap.add_argument("--tag", default="", help="prefix for result names; a tagged run does not overwrite OOF files")
    args = ap.parse_args()
    if args.quick:
        args.rounds, args.lr = 400, 0.1
    setup(args, "04_matcher")
    blocks = set(args.only.split(",")) if args.only else set("ABCDEF")
    s1, pool = load_universe(args)
    own = owner_index(s1, pool)
    k, q_fold, countries = s1.k.values, s1.fold.values, s1.country.values
    d = udir(args)
    df = pd.read_parquet(os.path.join(d, f"{args.feats}.parquet"))
    groups = {g: [c for c in cols if c in df] for g, cols in GROUPS.items()}
    for path in sorted(glob.glob(os.path.join(d, "extra_*.parquet"))):
        ex = pd.read_parquet(path)
        df = df.merge(ex, on=["qi", "pj"], how="left")
        new = [c for c in ex.columns if c not in ("qi", "pj")]
        g = "CE" if os.path.basename(path).startswith("extra_ce") else "EMB"
        groups.setdefault(g, []).extend(new)
        step(f"merged extra features {os.path.basename(path)} -> group {g}: {new}")
    step(f"{len(df):,} pairs, positives {df.label.mean():.2%}, groups: { {g: len(c) for g, c in groups.items()} }")
    base = ["CTX", "NAME", "ADDR", "NUM", "COMP"]
    full = [c for g in base for c in groups[g]]
    cheap = groups["CTX"] + groups["NUM"] + [c for c in CHEAP_EXTRA if c in df]

    configs = []
    if "A" in blocks:
        for i in range(1, len(base) + 1):
            configs.append(("A_" + "+".join(base[:i]), [c for g in base[:i] for c in groups[g]]))
        if groups.get("EMB"):
            configs.append(("A_FULL+EMB", full + groups["EMB"]))
        if groups.get("CE"):
            configs.append(("A_FULL+CE", full + groups["CE"]))
        if groups.get("EMB") and groups.get("CE"):
            configs.append(("A_FULL+EMB+CE", full + groups["EMB"] + groups["CE"]))
    if "B" in blocks:
        for g in base:
            configs.append((f"B_FULL-{g}", [c for c in full if c not in groups[g]]))
    if "C" in blocks or "D" in blocks:
        configs.append(("C_CHEAP", cheap))
    names = [n for n, _ in configs]
    if "A_" + "+".join(base) not in names:
        configs.append(("A_" + "+".join(base), full))

    oofs, results, models = {}, {}, {}
    for name, cols in configs:
        t0 = time.time()
        p, model = cv_oof(df, cols, args)
        secs = time.time() - t0
        y = df.label.values
        rep = quick_decision(df, p, own, k, q_fold, countries)
        results[name] = rep["f05_holdout"]
        oofs[name] = p
        models[name] = (model, cols)
        log_result(args, "04_matcher", args.tag + name, n_features=len(cols), auc=roc_auc_score(y, p),
                   ap=average_precision_score(y, p), logloss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                   seconds=secs, **rep)

    full_name = "A_" + "+".join(base)
    best_name = max(results, key=results.get)
    step(f"best config by holdout F0.5: {best_name} ({results[best_name]:.4f})")
    if args.tag:
        return
    for name in {full_name, best_name, "C_CHEAP"} & set(oofs):
        pd.DataFrame({"qi": df.qi.values, "pj": df.pj.values, "y": df.label.values, "p": oofs[name]}) \
            .to_parquet(os.path.join(d, f"oof_{name}.parquet"), index=False)
    with open(os.path.join(d, "oof_best.txt"), "w") as f:
        f.write(best_name)

    if "D" in blocks:
        step("D: candidate-size curve (what goes into candidate_pairs.tsv)")
        for name in ("C_CHEAP", full_name):
            if name not in oofs:
                continue
            dd = pd.DataFrame({"qi": df.qi.values, "pj": df.pj.values, "p": oofs[name]})
            for row in recall_at_k_curve(dd, own, k, "p", countries=countries):
                kk = row.pop("k")
                log_result(args, "04_candsize", f"{name}|top{kk}", **row)
            for t in (0.0005, 0.001, 0.003, 0.01, 0.03, 0.1):
                s = dd[dd.p >= t]
                log_result(args, "04_candsize", f"{name}|p>={t}", **blocking_stats(s.qi.values, s.pj.values, own, k,
                                                                                  countries))
            for kk, t in ((8, 0.003), (12, 0.003), (8, 0.01), (12, 0.01)):  # threshold with a safety cap
                r = dd.sort_values(["qi", "p"], ascending=[True, False])
                r = r[(r.groupby("qi").cumcount() < kk) & (r.p >= t)]
                log_result(args, "04_candsize", f"{name}|p>={t}&cap{kk}",
                           **blocking_stats(r.qi.values, r.pj.values, own, k, countries))

    if "E" in blocks and df.country.nunique() >= 2:
        step("E: country transfer (train on one country, test on the other)")
        for cols_name, cols in (("FULL", full), ("FULL-CTX", [c for c in full if c not in groups["CTX"]])):
            for src in df.country.unique():
                for tgt in df.country.unique():
                    if src == tgt:
                        continue
                    tr = (df.country == src).values
                    te = (df.country == tgt).values
                    model = fit(df.loc[tr, cols].values.astype(np.float32), df.label.values[tr], df.qi.values[tr], args)
                    p_src = np.zeros(len(df), np.float32)
                    p_src[te] = model.predict(df.loc[te, cols].values.astype(np.float32),
                                              num_iteration=model.best_iteration)
                    # threshold tuned on the SOURCE (in-sample OOF of the source is not available -> use a
                    # CV'd source model's OOF from the full run restricted to src rows)
                    src_rows = df[tr]
                    d_src = one_to_one(pd.DataFrame({"qi": src_rows.qi.values, "pj": src_rows.pj.values,
                                                     "p": oofs[full_name][tr] if cols_name == "FULL" else
                                                     oofs.get("B_FULL-CTX", oofs[full_name])[tr]}))
                    src_mask = countries == src
                    best_t, bf = 0.5, -1
                    for t in np.arange(0.2, 0.96, 0.05):
                        kept = thr(d_src, t)
                        f_ = macro_f05(kept.qi.values, kept.pj.values, own, k, mask=src_mask)["f05"]
                        if f_ > bf:
                            best_t, bf = t, f_
                    dt = one_to_one(pd.DataFrame({"qi": df.qi.values[te], "pj": df.pj.values[te], "p": p_src[te]}))
                    tgt_mask = countries == tgt
                    kept = thr(dt, best_t)
                    f_transfer = macro_f05(kept.qi.values, kept.pj.values, own, k, mask=tgt_mask)["f05"]
                    f_oracle = max(macro_f05(thr(dt, t).qi.values, thr(dt, t).pj.values, own, k, mask=tgt_mask)["f05"]
                                   for t in np.arange(0.2, 0.96, 0.05))
                    din = one_to_one(pd.DataFrame({"qi": df.qi.values[te], "pj": df.pj.values[te],
                                                   "p": oofs[full_name][te]}))
                    f_in = max(macro_f05(thr(din, t).qi.values, thr(din, t).pj.values, own, k, mask=tgt_mask)["f05"]
                               for t in np.arange(0.2, 0.96, 0.05))
                    log_result(args, "04_transfer", f"{cols_name}|{src}->{tgt}", f05_transfer_src_threshold=f_transfer,
                               f05_transfer_oracle_threshold=f_oracle, f05_in_country_cv=f_in,
                               transfer_penalty=f_in - f_transfer, src_threshold=best_t)

    if "F" in blocks:
        model, cols = models[full_name]
        imp = model.feature_importance("gain")
        tot = imp.sum() or 1
        for c, v in sorted(zip(cols, imp), key=lambda x: -x[1])[:30]:
            log_result(args, "04_importance", c, gain_share=float(v / tot))


if __name__ == "__main__":
    main()
