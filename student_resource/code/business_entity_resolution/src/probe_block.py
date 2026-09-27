#!/usr/bin/env python3
"""Read-only probe: which true pairs does the current blocking miss on slice B (test-like density), and which
wider-search option would recover them at what cost in extra candidates?
Options measured: forward top-K (K up to 50), reverse top-M (M up to 10), exact core-name key inside the country."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from ber import config as C
from ber.knn import topk_gpu
from ber.util import Tee, log

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
W = os.path.join(ROOT, "work")


def main():
    from ber.data import load_split, owner_index
    from ber.prep import load_norm
    from ber.v3 import core_names
    sys.stdout = Tee(os.path.join(W, "tune", "probe_block.log"))
    s1, pool = load_split(W, "train")
    own = owner_index(s1, pool)
    prio, cty, pcty = s1.prio.values, s1.country.values, pool.country.values
    inB = (prio >= C.TRAIN_B[0]) & (prio < C.TRAIN_B[1])
    lo, hi = C.TRAIN_HIDDEN_S1
    present = ~((prio >= lo) & (prio < hi))
    cands = pd.read_parquet(os.path.join(W, "block", "train_cands.parquet"), columns=["qi", "pj"])
    cands = cands[inB[cands.qi.values]]
    tp_pj = np.where((own >= 0))[0]
    tp_pj = tp_pj[inB[own[tp_pj]]]
    tp_q = own[tp_pj]
    found = pd.Series(True, index=pd.MultiIndex.from_arrays([cands.qi.values, cands.pj.values]))
    hit = found.reindex(pd.MultiIndex.from_arrays([tp_q, tp_pj])).fillna(False).values.astype(bool)
    log(f"B: {inB.sum():,} S1, {len(tp_pj):,} true pairs, {len(cands):,} candidates "
        f"({len(cands) / inB.sum():.1f}/S1); blocking pair recall {hit.mean():.5f}; missed {(~hit).sum():,}")
    mq, mp = tp_q[~hit], tp_pj[~hit]

    E1 = np.load(os.path.join(W, "emb", "train_s1.npy"), mmap_mode="r")
    Ep = np.load(os.path.join(W, "emb", "train_pool.npy"), mmap_mode="r")
    KMAX, MMAX = 50, 10
    fr = np.full(len(mq), 999, np.int32)
    rr = np.full(len(mq), 999, np.int32)
    # forward top-50 for B S1 that own a missed pair; reverse top-10 for the missed pool records
    for c in pd.unique(cty):
        pi = np.where(pcty == c)[0]
        sel = np.where(cty[mq] == c)[0]
        if len(sel) == 0:
            continue
        uq = np.unique(mq[sel])
        fi, _ = topk_gpu(E1[uq], Ep[pi], KMAX)
        pos = {int(q): r for r, q in enumerate(uq)}
        for t in sel:
            row = pi[fi[pos[int(mq[t])]]]
            w = np.where(row == mp[t])[0]
            if len(w):
                fr[t] = w[0]
        qa = np.where((cty == c) & present)[0]
        up = np.unique(mp[sel])
        ri, _ = topk_gpu(Ep[up], E1[qa], MMAX)
        posp = {int(p): r for r, p in enumerate(up)}
        for t in sel:
            row = qa[ri[posp[int(mp[t])]]]
            w = np.where(row == mq[t])[0]
            if len(w):
                rr[t] = w[0]
        log(f"  [{c}] probed {len(sel):,} missed pairs")

    ns, npl, _ = load_norm(W, "train")
    cq = core_names(ns.nname.values[mq])
    cp = core_names(npl.nname.values[mp])
    core_eq = cq == cp
    name_eq = ns.nname.values[mq] == npl.nname.values[mp]
    addr_empty = np.array([len(x.strip()) == 0 for x in npl.naddr.values[mp]])

    log("--- recovery of missed true pairs (share of the missed ones) ---")
    for K in (15, 20, 30, 50):
        log(f"forward top-{K}: {(fr < K).mean():.3f}")
    for M in (5, 10):
        log(f"reverse top-{M}: {(rr < M).mean():.3f}")
    log(f"exact core-name: {core_eq.mean():.3f}   exact normalized name: {name_eq.mean():.3f}")
    for K, M in ((20, 5), (30, 5), (20, 10)):
        log(f"fwd{K} | rev{M} | core-name: {((fr < K) | (rr < M) | core_eq).mean():.3f}")
    log(f"missed pairs with empty pool address: {addr_empty.mean():.3f}; never found by any option: "
        f"{((fr >= 50) & (rr >= 10) & ~core_eq).mean():.3f}")

    # cost: extra candidates per B S1 from an exact core-name join inside the country (visible S1 only)
    s1core = core_names(ns.nname.values)
    pcore = core_names(npl.nname.values)
    kq = pd.Series(cty[inB], dtype=object).str.cat(pd.Series(s1core[inB], dtype=object), sep="\t")
    kp = pd.Series(pcty, dtype=object).str.cat(pd.Series(pcore, dtype=object), sep="\t")
    pc = kp.value_counts()
    per = kq.map(pc).fillna(0).values
    log(f"core-name join: {per.sum():,.0f} pairs for B ({per.mean():.2f}/S1, p99 {np.quantile(per, 0.99):.0f}, "
        f"max {per.max():.0f}); capped at 20 per key: {np.minimum(per, 20).sum():,.0f}")
    log("PROBE DONE")


if __name__ == "__main__":
    main()
