#!/usr/bin/env python3
"""08 -- France probe (label-free): does our pipeline behave on the unseen country like on US / India?

There are no labels for the test set, so we use label-free signals, computed per test country on a sample of
S1 records searched against the FULL test pool of that country (HashingVectorizer TF-IDF: no vocabulary needed,
works at any scale -- the answer to "how do we do TF-IDF without knowing the vocabulary"):
  * top-1 cosine and top1-top2 gap distributions
  * house-number agreement between the S1 and its top-1 / top-5 neighbours  (pseudo-precision)
  * how many neighbours look like matches (cos >= 0.5)  vs the ~3.5 matches per S1 seen in train
Compared across normalisation levels (L1 vs L4 -- L4 French canonical forms are untested by labels).
Also prints a few France S1 records with their nearest neighbours so we can eyeball the noise.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from lab.common import base_parser, log_result, setup, step
from lab.data import load_cached
from lab.knn import topk_sparse
from lab.text import load_dict, normalize_series, numbers

# The pool (up to ~4.7M records per country) is streamed through worker processes twice: pass 1 returns only
# document frequencies, pass 2 hashes + scores each chunk against the queries inside the worker and returns only
# its top-10. No pool matrix is ever held in the main process (holding it exhausted 32 GB RAM on India).
HV = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 21, alternate_sign=False,
                       norm=None, dtype=np.float32)
_W = {}


def _init(dct, lvl, idf=None, Q=None):
    _W.update(dct=dct, lvl=lvl, idf=idf, Q=Q)


def _texts(names, addrs, countries, lvl, dct):
    return [a + " " + b for a, b in zip(normalize_series(names, countries, lvl, "name", dct),
                                        normalize_series(addrs, countries, lvl, "addr", dct))]


def _hash_chunk(names, addrs, countries):
    return HV.transform(_texts(names, addrs, countries, _W["lvl"], _W["dct"]))


def _df_chunk(off, names, addrs, countries):
    return np.bincount(_hash_chunk(names, addrs, countries).indices, minlength=HV.n_features).astype(np.int32)


def _score_chunk(off, names, addrs, countries):
    ci, cs = topk_sparse(_W["Q"], _tfidf(_hash_chunk(names, addrs, countries), _W["idf"]), 10, 1)
    return np.where(ci >= 0, ci + off, -1), cs


def _star_df(t):
    return _df_chunk(*t)


def _star_score(t):
    return _score_chunk(*t)


def _tfidf(H, idf):
    H = H.copy()
    np.log(H.data, out=H.data)
    H.data += 1
    H.data *= idf[H.indices]
    return l2_normalize(H, copy=False)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=3000, help="S1 records sampled per country")
    ap.add_argument("--levels", default="1,4")
    ap.add_argument("--countries", default="", help="default: all test countries")
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=50_000, help="pool rows per parallel hashing task")
    ap.add_argument("--workers", type=int, default=8, help="hashing processes (each ~0.5 GB RAM at chunk=50k)")
    args = ap.parse_args()
    setup(args, "08_france_probe")
    # 08 only reads the test data, so it can be run early, in parallel with 06/07; a later call then skips.
    marker = os.path.join(args.work, "results", "08_done.txt")
    key = f"n={args.n} levels={args.levels} countries={args.countries} seed={args.seed}"
    if os.path.exists(marker) and open(marker, encoding="utf-8").read().strip() == key:
        step(f"already done ({key}) -- skipping; delete {marker} to re-run")
        return
    s1 = load_cached(args, "test_s1")
    pool = pd.concat([load_cached(args, "test_s2"), load_cached(args, "test_s3")], ignore_index=True)
    dct = load_dict(os.path.join(args.work, "models", "translit_dict.json"))
    countries = args.countries.split(",") if args.countries else sorted(s1.country.unique())
    workers = max(1, min(args.n_jobs, args.workers))
    for cty in countries:
        q = s1[s1.country == cty].sample(min(args.n, int((s1.country == cty).sum())), random_state=args.seed)
        p = pool[pool.country == cty].reset_index(drop=True)
        qn_nums = [numbers(a) for a in q.addr.values]
        pn_nums = {}  # filled lazily for neighbours only
        n = len(p)
        for lvl in (int(x) for x in args.levels.split(",")):
            t0 = time.time()
            step(f"{cty} L{lvl}: normalise + hash {n:,} pool records in {workers} processes")
            tasks = [(i, p.name.values[i:i + args.chunk], p.addr.values[i:i + args.chunk],
                      p.country.values[i:i + args.chunk]) for i in range(0, n, args.chunk)]
            df = np.zeros(HV.n_features, np.float64)
            with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(dct, lvl)) as mpp:
                for d in mpp.imap_unordered(_star_df, tasks):
                    df += d
            idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)  # smooth idf over the FULL pool
            Q = _tfidf(HV.transform(_texts(q.name.values, q.addr.values, q.country.values, lvl, dct)), idf)
            idx = np.full((len(q), 10), -1, np.int64)
            sc = np.zeros((len(q), 10), np.float32)
            with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(dct, lvl, idf, Q)) as mpp:
                for ci, cs in mpp.imap_unordered(_star_score, tasks):
                    ai, as_ = np.hstack([idx, ci]), np.hstack([sc, cs])
                    o = np.argsort(-as_, axis=1, kind="stable")[:, :10]
                    idx, sc = np.take_along_axis(ai, o, 1), np.take_along_axis(as_, o, 1)
            for j in np.unique(idx[:, :5][idx[:, :5] >= 0]):
                if j not in pn_nums:
                    pn_nums[j] = numbers(p.addr.values[j])
            secs = time.time() - t0
            step(f"{cty} L{lvl}: done in {secs:.0f}s")
            agree1, agree5, has_num = [], [], []
            for i in range(len(q)):
                a = qn_nums[i]
                if not a:
                    continue
                has_num.append(1)
                j = idx[i, 0]
                agree1.append(float(j >= 0 and bool(a & pn_nums[j])))
                top = [jj for jj in idx[i, :5] if jj >= 0]
                agree5.append(np.mean([bool(a & pn_nums[jj]) for jj in top]) if top else 0.0)
            gap = sc[:, 0] - sc[:, 1]
            qs = lambda x: [round(float(v), 3) for v in np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])]
            log_result(args, "08_country_probe", f"{cty}|L{lvl}", pool=len(p), seconds=secs,
                       top1_cos_q=qs(sc[:, 0]), gap_q=qs(gap),
                       num_agree_top1=float(np.mean(agree1)) if agree1 else None,
                       num_agree_top5=float(np.mean(agree5)) if agree5 else None,
                       mean_neighbours_cos_ge_05=float((sc >= 0.5).sum(1).mean()),
                       mean_neighbours_cos_ge_07=float((sc >= 0.7).sum(1).mean()),
                       s1_with_numbers=float(len(has_num) / len(q)))
            if lvl == max(int(x) for x in args.levels.split(",")) and args.show:
                print(f"\n--- {cty}: sample S1 records and their top-4 neighbours (L{lvl}) ---")
                for i in range(min(args.show, len(q))):
                    r = q.iloc[i]
                    print(f"  S1 {r.entity_id} {r['name'][:55]} | {r.addr[:90]}")
                    for jj, s in zip(idx[i, :4], sc[i, :4]):
                        if jj >= 0:
                            pr = p.iloc[jj]
                            print(f"      {s:.3f} {pr.entity_id} {pr['name'][:55]} | {pr.addr[:90]}")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(key)


if __name__ == "__main__":
    main()
