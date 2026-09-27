#!/usr/bin/env python3
"""01 -- Which text normalisation / transliteration / TF-IDF representation retrieves true matches best?

For every variant: char n-gram TF-IDF on "name + address", exact per-country top-50 search (S1 -> pool),
report recall@{1,5,10,20,50}, per country, and on the hard subset of matches written in Indic scripts.

Variants
  norm ladder   : L0 raw, L1 cleanup, L2 translit (anyascii), L2 translit (sanscript), L3 +learned dict,
                  L4 +canonical forms, L4 + consonant skeleton
  representation: char_wb 2-4 (default), char_wb 3-3, char_wb 3-5, word unigrams
  field view    : name+addr (default), name only, addr only
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from lab.common import base_parser, log_result, setup, step
from lab.data import load_universe
from lab.knn import knn_grouped, make_vectorizer
from lab.metrics import owner_index, recall_from_idx
from lab.text import has_indic, load_dict, normalize_series, skel

KS = (1, 5, 10, 20, 50)


def run(args, name, s1_text, pool_text, s1, pool, own, native, vec_kind="char_wb24", extra=None):
    t0 = time.time()
    vec = make_vectorizer(vec_kind)
    vec.fit(list(pool_text) + list(s1_text))
    Q = vec.transform(s1_text)
    P = vec.transform(pool_text)
    idx, _ = knn_grouped(Q, P, s1.country.values, pool.country.values, max(KS), args.n_jobs)
    res = recall_from_idx(idx, own, s1.k.values, KS, s1.country.values, native)
    log_result(args, "01_normalization", name, vocab=len(vec.vocabulary_), seconds=time.time() - t0,
               **res, **(extra or {}))


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    setup(args, "01_normalization")
    s1, pool = load_universe(args)
    own = owner_index(s1, pool)
    native = pool.name.map(has_indic).values
    dct = load_dict(os.path.join(args.work, "models", "translit_dict.json"))
    sc, pc = s1.country.values, pool.country.values
    step(f"U1: S1={len(s1):,} pool={len(pool):,} native-script pool names={native.mean():.1%}")

    ladder = [("L0_raw", 0, None, "anyascii"), ("L1_clean", 1, None, "anyascii"),
              ("L2_translit_anyascii", 2, None, "anyascii"), ("L2_translit_sanscript", 2, None, "sanscript"),
              ("L3_learned_dict", 3, dct, "anyascii"), ("L4_canonical", 4, dct, "anyascii")]
    texts = {}
    for name, lvl, d, mode in ladder:
        if lvl >= 3 and d is None:
            step(f"skip {name}: no dictionary (run 00_prepare.py)")
            continue
        t0 = time.time()
        sn = normalize_series(s1.name.values, sc, lvl, "name", d, mode)
        sa = normalize_series(s1.addr.values, sc, lvl, "addr", d, mode)
        pn = normalize_series(pool.name.values, pc, lvl, "name", d, mode)
        pa = normalize_series(pool.addr.values, pc, lvl, "addr", d, mode)
        norm_rate = (len(s1) + len(pool)) / max(time.time() - t0, 1e-6)
        texts[name] = (sn, sa, pn, pa)
        step(f"{name}: normalised at {norm_rate:,.0f} records/s  e.g. '{pn[0]}' | '{pa[0]}'")
        run(args, name, [a + " " + b for a, b in zip(sn, sa)], [a + " " + b for a, b in zip(pn, pa)],
            s1, pool, own, native, extra={"norm_records_per_s": norm_rate})

    best = "L4_canonical" if "L4_canonical" in texts else list(texts)[-1]
    sn, sa, pn, pa = texts[best]
    step(f"representation / field variants on {best}")
    sk = lambda xs: [" ".join(skel(t) for t in x.split()) for x in xs]
    run(args, f"{best}+skeleton", [a + " " + b for a, b in zip(sk(sn), sa)],
        [a + " " + b for a, b in zip(sk(pn), pa)], s1, pool, own, native)
    for kind in ("char_wb33", "char_wb35", "word"):
        run(args, f"{best}|{kind}", [a + " " + b for a, b in zip(sn, sa)], [a + " " + b for a, b in zip(pn, pa)],
            s1, pool, own, native, vec_kind=kind)
    run(args, f"{best}|name_only", sn, pn, s1, pool, own, native)
    run(args, f"{best}|addr_only", sa, pa, s1, pool, own, native)


if __name__ == "__main__":
    main()
