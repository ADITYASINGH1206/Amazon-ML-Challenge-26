#!/usr/bin/env python3
"""03 -- Pairwise features for every candidate pair of a universe.

Groups (see lab/feats.py):
  CTX  blocking context: TF-IDF cosine, forward rank, reverse rank, gap to the S1's best score, reverse margin
       (how much better this pool record matches ITS best S1), mutual-best flag, #candidates, source S3 flag
  NAME char-trigram / token / core-name (legal words removed) similarities, consonant-skeleton overlap,
       Jaro-Winkler, token-set/sort, partial ratio on space-less names (catches 'vavacenter'), IDF-weighted overlap
  ADDR trigram / token / containment / IDF overlap / token-set on normalised addresses, empty flags
  COMP YOUR comma-component idea: each component of the shorter address fuzzy-matched to its best counterpart
  NUM  house-number & postcode agreement, conflicts, containment
Also prints single-feature AUCs so we see which signals carry the most information.
Output: universes/<U>/feats_L<level>.parquet
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lab.common import base_parser, log_result, setup, step, udir
from lab.data import load_universe
from lab.feats import GROUPS, add_context_features, build_idf, compute_pair_features, prep_records
from lab.text import load_dict


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--level", type=int, default=4)
    args = ap.parse_args()
    setup(args, "03_features")
    s1, pool = load_universe(args)
    d = udir(args)
    c = pd.read_parquet(os.path.join(d, "cands.parquet"))
    q_top1 = np.load(os.path.join(d, "q_top1.npy"))
    p_top1 = np.load(os.path.join(d, "p_top1.npy"))
    dct = load_dict(os.path.join(args.work, "models", "translit_dict.json"))
    step(f"{args.universe}: {len(c):,} candidate pairs, level L{args.level}")

    t0 = time.time()
    R_s = prep_records(s1, args.level, dct)
    R_p = prep_records(pool, args.level, dct)
    idf = build_idf([R_s, R_p])
    t_prep = time.time() - t0
    step(f"records prepared in {t_prep:.1f}s")

    t0 = time.time()
    F = compute_pair_features(R_s, R_p, c.qi.values, c.pj.values, idf, args.n_jobs)
    t_feat = time.time() - t0
    rate = len(c) / max(t_feat, 1e-6)
    step(f"pair features: {t_feat:.1f}s ({rate:,.0f} pairs/s with {args.n_jobs} workers)")
    c = add_context_features(c, q_top1, p_top1, pool.src.values)
    for f, v in F.items():
        c[f] = v
    c["fold"] = s1.fold.values[c.qi.values]
    c["country"] = s1.country.values[c.qi.values]
    out = os.path.join(d, f"feats_L{args.level}.parquet")
    c.to_parquet(out, index=False)
    log_result(args, "03_features", f"L{args.level}", pairs=len(c), prep_seconds=t_prep, feat_seconds=t_feat,
               pairs_per_s=rate, pos_rate=float(c.label.mean()))

    step("single-feature AUC (NaN -> median), overall and per country")
    y = c.label.values
    for g, cols in GROUPS.items():
        for f in cols:
            if f not in c:
                continue
            x = c[f].values.astype(float)
            nan = np.isnan(x)
            if nan.all() or y.min() == y.max():
                continue
            x = np.where(nan, np.nanmedian(x), x)
            auc = roc_auc_score(y, x)
            per = {}
            for cty in c.country.unique():
                m = (c.country == cty).values
                if y[m].min() != y[m].max():
                    a = roc_auc_score(y[m], x[m])
                    per[f"auc_{cty}"] = max(a, 1 - a)
            log_result(args, "03_feature_auc", f"{g}:{f}", auc=max(auc, 1 - auc), direction="+" if auc >= 0.5 else "-",
                       nan_rate=float(nan.mean()), **per)


if __name__ == "__main__":
    main()
