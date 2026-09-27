#!/usr/bin/env python3
"""Diagnostics on the held-out slice B (rules tuned on folds 0-2, reported on folds 3-4):
  1. where the F0.5 loss comes from (singleton false merges, fully missed S1, partial misses, extra matches)
  2. the ceiling of perfect singleton detection (drop every singleton's predictions, re-tune the rule)
  3. an S1-level "has at least one match" model and a decision rule that uses it
  4. check that stage 2 computed in S1 chunks equals one pass (what predict does to save memory)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from ber import config as C
from ber.util import Tee, load_json, log, save_json

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    import run_pipeline as R
    from ber.data import load_split, owner_index
    from ber.model import (add_pj_features, apply_rule, fit_lgb, macro_f05, per_entity_f05, predict, prep_rules,
                           tune_rule)
    from ber.prep import load_norm
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--n-jobs", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.work, "tune"), exist_ok=True)
    sys.stdout = Tee(os.path.join(a.work, "tune", "analyze.log"))
    W = a.work
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
    cand_tp = np.bincount(FB.qi.values[own[FB.pj.values] == FB.qi.values], minlength=len(k))   # before cheap filter
    sv = FB[FB.pc >= dec["t_cheap"]].copy()
    del FB
    sv["p"] = full.predict(sv[dec["cols1"]].values.astype(np.float32)).astype(np.float32)
    surv_tp = np.bincount(sv.qi.values[own[sv.pj.values] == sv.qi.values], minlength=len(k))
    cols2 = dec["cols2"]
    log(f"B survivors {len(sv):,}; building stage-2 features")
    d = R.stage2_frame(sv.copy(), pool, npl.nname.values, npl.naddr.values, emb, dec.get("use_s2x", False))
    X, y, q = d[cols2].values.astype(np.float32), d.label.values, d.qi.values
    p2 = np.zeros(len(d), np.float32)
    for f in range(5):
        te = fold[q] == f
        m = fit_lgb(X[~te], y[~te], q[~te], C.LGB_STAGE2, a.n_jobs)
        p2[te] = predict(m, X[te])
        if f == 0:
            m0 = m
    rep = {}

    # 4. chunked stage 2 == one pass (predictions of the same model)
    ch = add_pj_features(sv.copy()).sort_values("qi", kind="stable").reset_index(drop=True)
    cuts = np.searchsorted(ch.qi.values, np.quantile(ch.qi.values, [0, 0.33, 0.66]).astype(np.int64)).tolist()
    cuts = sorted(set(cuts)) + [len(ch)]
    outs = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        part = R.stage2_frame(ch.iloc[lo:hi].copy(), pool, npl.nname.values, npl.naddr.values, emb,
                              dec.get("use_s2x", False))
        outs.append(part[["qi", "pj"]].assign(pc_=m0.predict(part[cols2].values.astype(np.float32))))
    chk = d[["qi", "pj"]].assign(p1_=m0.predict(X)).merge(pd.concat(outs), on=["qi", "pj"])
    rep["chunk_check"] = {"pairs": len(chk), "of": len(d), "max_abs_diff": float(np.abs(chk.p1_ - chk.pc_).max())}
    log(f"chunked vs one-pass stage 2: {rep['chunk_check']}")

    # 1. loss decomposition for the current decision rule
    d2 = d[["qi", "pj"]].assign(p=p2)
    s = prep_rules(d2)
    rule0 = "top1_plus" if dec["rule"] == "gated" else dec["rule"]
    params, _ = tune_rule(d2, own, k, tune, rule0, prepared=s)
    kept = apply_rule(d2, rule0, params, prepared=s)
    f, npred, tp = per_entity_f05(kept.qi.values, kept.pj.values, own, k)
    n = hold.sum()
    L = 1 - f
    sing, non = hold & (k == 0), hold & (k > 0)
    parts = {
        "singleton_false_merge": sing & (npred > 0),
        "nonsingleton_predicted_empty": non & (npred == 0),
        "nonsingleton_missed_some_only": non & (npred > 0) & (tp < k) & (npred == tp),
        "nonsingleton_extra_only": non & (npred > 0) & (tp == k) & (npred > tp),
        "nonsingleton_missed_and_extra": non & (npred > 0) & (tp < k) & (npred > tp),
    }
    dec_rep = {"rule": rule0, "params": list(params), "f05_holdout": float(f[hold].mean()),
               "singleton_rate": float((k[hold] == 0).mean())}
    for name, msk in parts.items():
        dec_rep[name] = {"share_of_S1": float(msk.sum() / n), "f05_loss": float(L[msk].sum() / n)}
    miss_pairs = (k - tp)[non].sum()
    dec_rep["missed_pairs"] = {
        "total_share_of_true_pairs": float(miss_pairs / k[non].sum()),
        "not_in_blocking": float((k - cand_tp)[non].sum() / max(miss_pairs, 1)),
        "removed_by_cheap_filter": float((cand_tp - surv_tp)[non].sum() / max(miss_pairs, 1)),
        "in_candidates_but_rejected": float((surv_tp - tp)[non].sum() / max(miss_pairs, 1))}
    empty_non = non & (npred == 0)
    dec_rep["predicted_empty_nonsingletons_with_a_true_candidate"] = float((surv_tp[empty_non] > 0).mean())
    rep["loss_decomposition"] = dec_rep
    log(f"loss decomposition: {dec_rep}")

    # 1b. what the rejected true matches and the false merges look like
    kp = set(zip(kept.qi.values, kept.pj.values))
    dd = d.assign(p2=p2, kept=[(q_, j_) in kp for q_, j_ in zip(d.qi.values, d.pj.values)])
    dd = dd[hold[dd.qi.values]]
    miss = dd[(dd.label == 1) & ~dd.kept]
    fp = dd[(dd.label == 0) & dd.kept]
    miss_empty_q = np.isin(miss.qi.values, np.where(npred == 0)[0])
    cat = {
        "n_missed_in_candidates": int(len(miss)), "n_false_merges": int(len(fp)),
        "missed_pool_name_indic": float(miss.c_indic.mean()),
        "all_true_pool_name_indic": float(dd[dd.label == 1].c_indic.mean()),
        "missed_pool_addr_empty": float(miss.addr_empty_c.mean()),
        "all_true_pool_addr_empty": float(dd[dd.label == 1].addr_empty_c.mean()),
        "missed_whose_S1_got_nothing": float(miss_empty_q.mean()),
        "missed_p2_quantiles": [float(x) for x in np.quantile(miss.p2, [0.1, 0.25, 0.5, 0.75, 0.9])],
        "missed_fwd_rank_nan(rev_only)": float(miss.fwd_rank.isna().mean()),
        "missed_num_conflict": float((miss.num_conflict == 1).mean()),
        "missed_name_jw_quantiles": [float(x) for x in np.quantile(miss.name_jw.fillna(0), [0.1, 0.25, 0.5, 0.75, 0.9])],
        "fp_p2_quantiles": [float(x) for x in np.quantile(fp.p2, [0.1, 0.5, 0.9])] if len(fp) else [],
        "fp_owner_is_other_S1": float((own[fp.pj.values] >= 0).mean()) if len(fp) else None,
        "missed_by_country": pd.Series(cty[miss.qi.values]).value_counts(normalize=True).round(3).to_dict()}
    rep["error_profile"] = cat
    log(f"error profile: {cat}")
    rng = np.random.default_rng(0)
    for title, df_ in (("MISSED TRUE MATCHES", miss), ("FALSE MERGES", fp)):
        log(f"--- sample of {title} ---")
        for i in rng.choice(len(df_), size=min(25, len(df_)), replace=False) if len(df_) else []:
            r = df_.iloc[i]
            q_, j_ = int(r.qi), int(r.pj)
            print(f"  p2={r.p2:.3f} p1={r.p:.3f} cos={r.cos:.3f} fwd={r.fwd_rank} rev={r.rev_rank} "
                  f"name_jw={r.name_jw:.2f} num_share={r.num_share} | S1 [{cty[q_]}] {s1.name.values[q_][:45]} | "
                  f"{s1.addr.values[q_][:70]}")
            print(f"      -> {pool.src.values[j_]} {pool.name.values[j_][:45]} | {pool.addr.values[j_][:70]}")

    # 2. ceiling of perfect singleton detection
    d_or = d2[k[d2.qi.values] > 0]
    s_or = prep_rules(d_or)
    ceil = {}
    for r in C.RULES_TRY:
        pr, _ = tune_rule(d_or, own, k, tune, r, prepared=s_or)
        kk = apply_rule(d_or, r, pr, prepared=s_or)
        ceil[r] = {"params": list(pr), "f05_holdout": macro_f05(kk.qi.values, kk.pj.values, own, k, hold)["f05"]}
    rep["oracle_singletons"] = ceil
    log(f"perfect singleton detection (ceiling): {ceil}")

    # 3. S1-level 'has at least one match' model (5-fold OOF) + decision rule using it
    d["p2"] = p2
    g = d.sort_values(["qi", "p2"], ascending=[True, False])
    top = g.groupby("qi").head(1).set_index("qi")
    g = g.assign(_hi=(g.p2 > 0.5).astype(np.float32), _mid=(g.p2 > 0.2).astype(np.float32))
    gb = g.groupby("qi")
    ent = pd.DataFrame({
        "p2_max": gb.p2.max(), "p2_sum": gb.p2.sum(), "n_cand": gb.size().astype(np.float32),
        "n_hi": gb._hi.sum(), "n_mid": gb._mid.sum(),
        "p2_2nd": g[gb.cumcount().values == 1].set_index("qi").p2,
        "cos_max": gb.cos.max()})
    for c in ["p", "cos", "gap_q", "rev_rank", "rev_margin", "name_jw", "name_tri_idfcos", "addr_tri_idfcos",
              "num_share", "num_conflict", "core_idf_q", "addr_tsr", "s2_nclaim_pj", "s2_pmax_other_pj",
              "unshared_max_idf_q", "addr_unshared_max_idf_q"]:
        if c in top:
            ent[f"top_{c}"] = top[c]
    ent = ent.reindex(np.where(inB)[0])                     # every B S1, including those without candidates
    E = ent.values.astype(np.float32)
    ye = (k[ent.index.values] > 0).astype(np.int8)
    qe = ent.index.values
    e = np.zeros(len(ent), np.float32)
    for f_ in range(5):
        te = fold[qe] == f_
        m = fit_lgb(E[~te], ye[~te], qe[~te], dict(C.LGB_STAGE2, min_data_in_leaf=50), a.n_jobs)
        e[te] = predict(m, E[te])
    has = ent.n_cand.notna().values
    rep["entity_model"] = {"auc_all_B": float(roc_auc_score(ye, e)),
                           "auc_with_candidates": float(roc_auc_score(ye[has], e[has]))}
    log(f"has-a-match model: {rep['entity_model']}")
    emap = pd.Series(e, index=qe)
    s2 = s.assign(e=emap.reindex(s.qi.values).values)
    best, best_f = None, -1
    for tE in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for t1 in (0.2, 0.3, 0.4, 0.5, 0.6):
            for t2 in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
                sel = s2[(s2.e.values >= tE) & (((s2._r.values == 0) & (s2.p.values >= t1)) |
                                                ((s2._r.values > 0) & (s2.p.values >= t2)))]
                fv = macro_f05(sel.qi.values, sel.pj.values, own, k, tune)["f05"]
                if fv > best_f:
                    best, best_f = (tE, t1, t2), fv
    tE, t1, t2 = best
    sel = s2[(s2.e.values >= tE) & (((s2._r.values == 0) & (s2.p.values >= t1)) |
                                    ((s2._r.values > 0) & (s2.p.values >= t2)))]
    rep["entity_rule"] = {"params": list(best), "f05_tune": best_f,
                          **macro_f05(sel.qi.values, sel.pj.values, own, k, hold, cty)}
    log(f"rule with has-a-match gate: {rep['entity_rule']}")
    save_json(rep, os.path.join(W, "tune", "analyze.json"))
    log("ANALYSIS DONE")


if __name__ == "__main__":
    main()
