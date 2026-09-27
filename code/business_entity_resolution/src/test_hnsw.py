#!/usr/bin/env python3
"""HNSW graph index (FAISS, CPU) vs exact search on the held-out slice B: forward top-K_FWD recall of true pairs,
build and search time. Complements test_ivf.py."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from ber import config as C
from ber.util import Tee, log

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def main():
    import faiss
    from ber.data import load_split, owner_index
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--countries", default="India")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=64)
    ap.add_argument("--ef-search", default="32,64,128,256")
    ap.add_argument("--fetch", default="10", help="approximate neighbours fetched, then re-scored exactly -> top K_FWD")
    a = ap.parse_args()
    W = a.work
    sys.stdout = Tee(os.path.join(W, "tune", "test_hnsw.log"))
    s1, pool = load_split(W, "train", columns=None)
    own = owner_index(s1, pool)
    prio, cty, pcty = s1.prio.values, s1.country.values, pool.country.values
    inB = (prio >= C.TRAIN_B[0]) & (prio < C.TRAIN_B[1])
    E1 = np.load(os.path.join(W, "emb", "train_s1.npy"), mmap_mode="r")
    Ep = np.load(os.path.join(W, "emb", "train_pool.npy"), mmap_mode="r")
    faiss.omp_set_num_threads(32)
    for c in a.countries.split(","):
        pi = np.where(pcty == c)[0]
        qn = np.where((cty == c) & inB)[0]
        Pf = np.asarray(Ep[pi], dtype=np.float32)
        Qf = np.asarray(E1[qn], dtype=np.float32)
        n_true = np.isin(own[pi], qn).sum()

        def recall(I):
            rows = pi[np.where(I >= 0, I, 0)]
            return ((own[rows] == qn[:, None]) & (I >= 0)).sum() / max(n_true, 1)

        t = time.time()
        ix = faiss.IndexHNSWFlat(Pf.shape[1], a.M, faiss.METRIC_INNER_PRODUCT)
        ix.hnsw.efConstruction = a.ef_construction
        ix.add(Pf)
        log(f"[{c}] HNSW M={a.M} efC={a.ef_construction}: built on {len(pi):,} pool records in {time.time() - t:.0f}s")
        for fetch in [int(x) for x in a.fetch.split(",")]:
            for ef in [int(x) for x in a.ef_search.split(",")]:
                ix.hnsw.efSearch = max(ef, fetch)
                t = time.time()
                _, I = ix.search(Qf, fetch)
                if fetch > C.K_FWD:                     # exact re-scoring of the fetched set, keep the top K_FWD
                    s_ = np.einsum("nd,nkd->nk", Qf, Pf[np.where(I >= 0, I, 0)])
                    s_[I < 0] = -9
                    I = np.take_along_axis(I, np.argsort(-s_, 1)[:, :C.K_FWD], 1)
                dt = time.time() - t
                log(f"[{c}]   fetch {fetch} efSearch {ef}: fwd recall {recall(I):.5f}, {dt:.1f}s for {len(qn):,} "
                    f"queries ({dt / len(qn) * 1e6:.0f} us/query)")
        del ix, Pf
    log("HNSW TEST DONE")


if __name__ == "__main__":
    main()
