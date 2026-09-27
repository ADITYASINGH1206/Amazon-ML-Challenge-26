#!/usr/bin/env python3
"""IVF index vs exact search on the held-out slice B (training split, test-like density): forward top-K_FWD for the
B S1 and reverse top-M_REV for a sample of pool records. Reports true-pair recall, neighbour overlap and time."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from ber import config as C
from ber.knn import topk_gpu, topk_ivf
from ber.util import Tee, log

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def main():
    from ber.data import load_split, owner_index
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--nprobe", default="16,32,64")
    ap.add_argument("--list-size", type=int, default=C.IVF_LIST_SIZE)
    ap.add_argument("--rev-sample", type=int, default=300_000)
    a = ap.parse_args()
    W = a.work
    sys.stdout = Tee(os.path.join(W, "tune", "test_ivf.log"))
    s1, pool = load_split(W, "train", columns=None)
    own = owner_index(s1, pool)
    prio, cty, pcty = s1.prio.values, s1.country.values, pool.country.values
    inB = (prio >= C.TRAIN_B[0]) & (prio < C.TRAIN_B[1])
    lo, hi = C.TRAIN_HIDDEN_S1
    present = ~((prio >= lo) & (prio < hi))
    E1 = np.load(os.path.join(W, "emb", "train_s1.npy"), mmap_mode="r")
    Ep = np.load(os.path.join(W, "emb", "train_pool.npy"), mmap_mode="r")
    rng = np.random.default_rng(0)
    for c in pd.unique(cty):
        pi = np.where(pcty == c)[0]
        qn = np.where((cty == c) & inB)[0]
        qa = np.where((cty == c) & present)[0]
        Qf, Pf = np.asarray(E1[qn]), np.asarray(Ep[pi])
        ps = np.sort(rng.choice(len(pi), min(a.rev_sample, len(pi)), replace=False))
        Qr, Pr = Pf[ps], np.asarray(E1[qa])
        tf = own[pi]                                   # owner of each pool row (S1 index or -1)
        pos_f = {int(q): r for r, q in enumerate(qn)}

        def fwd_recall(fi):
            rows = pi[np.where(fi >= 0, fi, 0)]
            hit = (own[rows] == qn[:, None]) & (fi >= 0)
            n_true = np.bincount(own[pi][np.isin(own[pi], qn)], minlength=len(own))[qn].sum()
            return hit.sum() / max(n_true, 1)

        def rev_recall(ri):
            o = tf[ps]
            has = o >= 0
            hit = (qa[np.where(ri >= 0, ri, 0)] == o[:, None]) & (ri >= 0)
            return hit.any(1)[has].mean()

        t = time.time()
        ef, _ = topk_gpu(Qf, Pf, C.K_FWD)
        er, _ = topk_gpu(Qr, Pr, C.M_REV)
        te = time.time() - t
        log(f"[{c}] pool {len(pi):,}, S1 {len(qa):,}; exact: fwd recall {fwd_recall(ef):.5f}, "
            f"rev recall {rev_recall(er):.5f}, {te:.0f}s")
        for npb in [int(x) for x in a.nprobe.split(",")]:
            t = time.time()
            vf, _ = topk_ivf(Qf, Pf, C.K_FWD, list_size=a.list_size, nprobe=npb)
            vr, _ = topk_ivf(Qr, Pr, C.M_REV, list_size=a.list_size, nprobe=npb)
            tv = time.time() - t
            ov_f = np.mean([len(set(x) & set(y)) / C.K_FWD for x, y in zip(ef[:20000], vf[:20000])])
            scanned = min(1.0, npb * a.list_size / len(pi))
            log(f"[{c}]   ivf nprobe {npb}: fwd recall {fwd_recall(vf):.5f}, rev recall {rev_recall(vr):.5f}, "
                f"fwd top-{C.K_FWD} overlap {ov_f:.4f}, scans ~{scanned:.2%} of the pool, {tv:.0f}s")
    log("IVF TEST DONE")


if __name__ == "__main__":
    main()
