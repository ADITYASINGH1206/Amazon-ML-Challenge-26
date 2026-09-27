#!/usr/bin/env python3
"""Label-free comparison of the test predictions by country (France has no training labels).
Uses work/feats/test_scores.parquet (qi, pj, p of every scored candidate, written by predict) and output/.
  * uncertain mass: share of candidates with 0.1 <= p < 0.9 and mean per-S1 count of them
  * self-estimated F0.5: each S1's expected F0.5 of the submitted set if the probabilities were calibrated
  * accepted matches with a dissimilar name (token-set ratio < 50): different-name copies are the riskiest
Optionally compares with an older matching_results.tsv (--old) to see which countries a change touched."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from ber.util import Tee, log

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def read_matches(path, id2q, id2p):
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    m = m[m.matched_entity_ids != ""]
    m = m.assign(pid=m.matched_entity_ids.str.split(",")).explode("pid")
    return pd.DataFrame({"qi": id2q.reindex(m.source1_entity_id.values).values,
                         "pj": id2p.reindex(m.pid.values).values}).astype(np.int64)


def main():
    from rapidfuzz import fuzz
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    ap.add_argument("--old", default=None, help="an older matching_results.tsv to compare with")
    a = ap.parse_args()
    sys.stdout = Tee(os.path.join(a.work, "tune", "diag_country.log"))
    s1 = pd.read_parquet(os.path.join(a.work, "cache", "test_s1.parquet"), columns=["entity_id", "country"])
    pool = pd.read_parquet(os.path.join(a.work, "cache", "test_pool.parquet"), columns=["entity_id"])
    ns = pd.read_parquet(os.path.join(a.work, "cache", "test_s1_norm.parquet"), columns=["nname"]).nname.values
    npn = pd.read_parquet(os.path.join(a.work, "cache", "test_pool_norm.parquet"), columns=["nname"]).nname.values
    cty = s1.country.values
    id2q = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    id2p = pd.Series(np.arange(len(pool)), index=pool.entity_id.values)
    del pool

    sc = pd.read_parquet(os.path.join(a.work, "feats", "test_scores.parquet"))
    sc["cty"] = cty[sc.qi.values]
    unc = (sc.p >= 0.1) & (sc.p < 0.9)
    rep = pd.DataFrame({"scored_per_S1": sc.groupby("cty").size() / pd.Series(cty).value_counts(),
                        "uncertain_share": unc.groupby(sc.cty).mean(),
                        "uncertain_per_S1": unc.groupby(sc.cty).sum() / pd.Series(cty).value_counts()})

    kept = read_matches(os.path.join(a.out, "matching_results.tsv"), id2q, id2p)
    kept = kept.merge(sc[["qi", "pj", "p"]], on=["qi", "pj"], how="left")
    # expected F0.5 per S1 under calibration: E[TP] = sum p over kept, E[#true] = sum p over all scored
    tp = kept.groupby("qi").p.sum()
    nk = kept.groupby("qi").size()
    tot = sc.groupby("qi").p.sum()
    q = pd.DataFrame({"tp": tp, "nk": nk}).reindex(np.arange(len(s1))).fillna(0)
    q["tot"] = tot.reindex(q.index).fillna(0).values
    b2 = 0.25
    prec = np.where(q.nk > 0, q.tp / q.nk.clip(lower=1), 1.0)
    rec = np.where(q.tot > 1e-9, q.tp / q.tot.clip(lower=1e-9), 1.0)
    f = np.where((q.nk == 0) & (q.tot < 0.5), 1.0, (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-9))
    rep["selfest_F05"] = pd.Series(f).groupby(cty).mean()
    rep["kept_mean_p"] = kept.p.groupby(cty[kept.qi.values]).mean()
    rep["kept_p_below_0.7"] = (kept.p < 0.7).groupby(cty[kept.qi.values]).mean()
    tsr = np.array([fuzz.token_set_ratio(x, y) for x, y in zip(ns[kept.qi.values], npn[kept.pj.values])])
    rep["kept_name_dissimilar"] = pd.Series(tsr < 50).groupby(cty[kept.qi.values]).mean()
    rep["kept_per_S1"] = kept.groupby(cty[kept.qi.values]).size() / pd.Series(cty).value_counts()
    log("by country:\n" + rep.round(4).to_string())

    if a.old:
        old = read_matches(a.old, id2q, id2p)
        k = set(zip(kept.qi.values.tolist(), kept.pj.values.tolist()))
        o = set(zip(old.qi.values.tolist(), old.pj.values.tolist()))
        add = pd.Series([cty[x[0]] for x in k - o]).value_counts()
        rem = pd.Series([cty[x[0]] for x in o - k]).value_counts()
        n = pd.Series(cty).value_counts()
        log("vs old (per 1,000 S1): added " + (add / n * 1000).round(2).to_dict().__repr__() +
            " removed " + (rem / n * 1000).round(2).to_dict().__repr__())
    log("DIAG DONE")


if __name__ == "__main__":
    main()
