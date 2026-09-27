#!/usr/bin/env python3
"""02 -- Blocking / candidate generation: recall vs candidate-set size, scaling, and the candidate file.

B1 forward exact TF-IDF top-K (S1 -> pool, per country)                K = 1..100
B2 separate name-index + address-index top-K, unioned
B3 REVERSE top-m (each pool record -> its m nearest S1).  Since every pool record belongs to at most one S1,
   this gives ~4.7*m candidates per S1 on average.
B4 union of forward top-K and reverse top-m                              grid
B5 key blocking (hash keys, O(n), scales to billions): house-number x name-skeleton, rarest name token,
   house-number x rarest address token -- alone and as an add-on to B4
B6 approximate dense search: TruncatedSVD(128) of TF-IDF + FAISS HNSW vs exact -- what we lose at scale
--scaling : B1/B3/B4 on the nested universes F0.01 .. F0.1 -> extrapolate recall and runtime to full size
Writes universes/<U>/cands.parquet (+ q_top1.npy, p_top1.npy) for the configured K/m (used by 03-07).
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from lab.common import base_parser, log_result, setup, step, udir
from lab.data import load_universe
from lab.feats import prep_records
from lab.knn import build_candidates, knn_grouped, make_vectorizer, svd_dense, topk_dense
from lab.metrics import blocking_stats, owner_index
from lab.text import load_dict, normalize_series, skel


def texts(df, level, dct):
    c = df.country.values
    n = normalize_series(df.name.values, c, level, "name", dct)
    a = normalize_series(df.addr.values, c, level, "addr", dct)
    return n, a


def vectorize(s_txt, p_txt, kind):
    vec = make_vectorizer(kind)
    vec.fit(list(p_txt) + list(s_txt))
    return vec.transform(s_txt), vec.transform(p_txt)


def fwd_pairs(fi, K):
    K = min(K, fi.shape[1])
    qi = np.repeat(np.arange(fi.shape[0]), K)
    pj = fi[:, :K].ravel()
    ok = pj >= 0
    return qi[ok], pj[ok]


def rev_pairs(ri, m):
    m = min(m, ri.shape[1])
    pj = np.repeat(np.arange(ri.shape[0]), m)
    qi = ri[:, :m].ravel()
    ok = qi >= 0
    return qi[ok], pj[ok]


def union(*pairs):
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def key_blocks(R_s, R_p, cap):
    """Returns {key_name: (qi, pj)} for three hash-key schemes."""
    from collections import Counter
    dfc = Counter()
    for R in (R_s, R_p):
        for c, core in zip(R["country"], R["core"]):
            dfc.update((c, skel(t)) for t in core)
    dfa = Counter()
    for R in (R_s, R_p):
        for c, at in zip(R["country"], R["addr_tok"]):
            dfa.update((c, t) for t in at if not t.isdigit())

    def keys(R, scheme):
        out = []
        for c, core, nums, at in zip(R["country"], R["core"], R["nums"], R["addr_tok"]):
            ns = sorted(nums, key=len, reverse=True)[:3]
            if scheme == "num_x_name4":
                ks = [f"{c}|{n}|{skel(t)[:4]}" for n in ns for t in core if len(t) >= 2]
            elif scheme == "rarest_name_tok":
                toks = sorted({skel(t) for t in core if len(t) >= 3}, key=lambda t: dfc[(c, t)])[:2]
                ks = [f"{c}|{t}" for t in toks]
            else:  # num_x_rarest_addr_tok
                cand = [t for t in at if not t.isdigit() and len(t) >= 3]
                r = min(cand, key=lambda t: dfa[(c, t)]) if cand else None
                ks = [f"{c}|{n}|{r}" for n in ns] if r else []
            out.append(ks)
        return out

    res = {}
    for scheme in ("num_x_name4", "rarest_name_tok", "num_x_rarest_addr_tok"):
        ks, kp = keys(R_s, scheme), keys(R_p, scheme)
        a = pd.DataFrame({"key": [k for L in ks for k in L], "qi": np.repeat(np.arange(len(ks)), [len(L) for L in ks])})
        b = pd.DataFrame({"key": [k for L in kp for k in L], "pj": np.repeat(np.arange(len(kp)), [len(L) for L in kp])})
        sz = b.groupby("key").size()
        b = b[b.key.isin(sz[sz <= cap].index)]
        m = a.merge(b, on="key")[["qi", "pj"]].drop_duplicates()
        res[scheme] = (m.qi.values, m.pj.values)
    return res


def run_main(args):
    s1, pool = load_universe(args)
    own = owner_index(s1, pool)
    k = s1.k.values
    sc, pc = s1.country.values, pool.country.values
    dct = load_dict(os.path.join(args.work, "models", "translit_dict.json"))
    step(f"{args.universe}: S1={len(s1):,} pool={len(pool):,}")
    sn, sa = texts(s1, args.level, dct)
    pn, pa = texts(pool, args.level, dct)
    Q, P = vectorize([a + " " + b for a, b in zip(sn, sa)], [a + " " + b for a, b in zip(pn, pa)], args.vec)

    def log(variant, qi, pj, seconds=None, **extra):
        st = blocking_stats(qi, pj, own, k, sc)
        log_result(args, "02_blocking", variant, **st, **({"seconds": seconds} if seconds else {}), **extra)
        return st

    t0 = time.time()
    fi, fs = knn_grouped(Q, P, sc, pc, 100, args.n_jobs)
    t_fwd = time.time() - t0
    step(f"forward top-100 search: {t_fwd:.1f}s ({len(s1) / t_fwd:,.0f} queries/s)")
    for K in (1, 3, 5, 10, 20, 30, 50, 100):
        log(f"B1_fwd_K{K}", *fwd_pairs(fi, K), seconds=t_fwd if K == 100 else None)

    t0 = time.time()
    ri, rs = knn_grouped(P, Q, pc, sc, 5, args.n_jobs)
    t_rev = time.time() - t0
    step(f"reverse top-5 search: {t_rev:.1f}s")
    for m in (1, 2, 3, 5):
        log(f"B3_rev_m{m}", *rev_pairs(ri, m), seconds=t_rev if m == 5 else None)

    for K in (1, 3, 5, 10, 20, 50):
        for m in (1, 2, 3):
            log(f"B4_fwd{K}+rev{m}", *union(fwd_pairs(fi, K), rev_pairs(ri, m)))

    step("B2: separate name and address indexes")
    Qn, Pn = vectorize(sn, pn, args.vec)
    Qa, Pa = vectorize(sa, pa, args.vec)
    ni, _ = knn_grouped(Qn, Pn, sc, pc, 50, args.n_jobs)
    ai, _ = knn_grouped(Qa, Pa, sc, pc, 50, args.n_jobs)
    for K in (5, 10, 20, 50):
        log(f"B2_name{K}+addr{K}", *union(fwd_pairs(ni, K), fwd_pairs(ai, K)))
    log("B2_name10+addr10+fwd10+rev3", *union(fwd_pairs(ni, 10), fwd_pairs(ai, 10), fwd_pairs(fi, 10),
                                                rev_pairs(ri, 3)))

    step("B5: key blocking")
    t0 = time.time()
    R_s = prep_records(s1, args.level, dct, with_comps=False)
    R_p = prep_records(pool, args.level, dct, with_comps=False)
    kb = key_blocks(R_s, R_p, args.key_cap)
    t_key = time.time() - t0
    for name, pr in kb.items():
        log(f"B5_key_{name}", *pr)
    allk = union(*kb.values())
    log("B5_key_union", *allk, seconds=t_key)
    base = union(fwd_pairs(fi, 10), rev_pairs(ri, 3))
    log("B5_fwd10+rev3+key_union", *union(base, allk))

    if not args.skip_dense:
        step("B6: approximate dense search (SVD-128 + HNSW) vs exact")
        t0 = time.time()
        q, p = svd_dense(Q, P, 128)
        t_svd = time.time() - t0
        for hnsw in (False, True):
            t0 = time.time()
            di, _ = knn_grouped(q, p, sc, pc, 50, args.n_jobs,
                                fn=lambda a, b, kk, nj, h=hnsw: topk_dense(a, b, kk, nj, hnsw=h))
            t = time.time() - t0
            dri, _ = knn_grouped(p, q, pc, sc, 3, args.n_jobs,
                                 fn=lambda a, b, kk, nj, h=hnsw: topk_dense(a, b, kk, nj, hnsw=h))
            tag = "hnsw" if hnsw else "exact"
            log(f"B6_svd128_{tag}_fwd50", *fwd_pairs(di, 50), seconds=t + t_svd)
            log(f"B6_svd128_{tag}_fwd10+rev3", *union(fwd_pairs(di, 10), rev_pairs(dri, 3)))

    step(f"writing candidate file: fwd top-{args.k_fwd} + rev top-{args.m_rev}")
    write_cands(args, args.universe, s1, pool, Q, P, own)


def write_cands(args, tag, s1, pool, Q, P, own):
    c, q_top1, p_top1 = build_candidates(Q, P, s1.country.values, pool.country.values, args.k_fwd, args.m_rev,
                                         args.n_jobs)
    c["label"] = (own[c.pj.values] == c.qi.values).astype(np.int8)
    d = udir(args, tag)
    c.to_parquet(os.path.join(d, "cands.parquet"), index=False)
    np.save(os.path.join(d, "q_top1.npy"), q_top1)
    np.save(os.path.join(d, "p_top1.npy"), p_top1)
    with open(os.path.join(d, "cands_meta.json"), "w") as f:
        json.dump({"level": args.level, "vec": args.vec, "k_fwd": args.k_fwd, "m_rev": args.m_rev}, f)
    st = blocking_stats(c.qi.values, c.pj.values, own, s1.k.values, s1.country.values)
    log_result(args, "02_cands_written", f"{tag}|fwd{args.k_fwd}+rev{args.m_rev}", n_pairs=len(c), **st)


def run_u2(args):
    tag = "U2"
    s1, pool = load_universe(args, tag)
    own = owner_index(s1, pool)
    dct = load_dict(os.path.join(args.work, "models", "translit_dict.json"))
    sn, sa = texts(s1, args.level, dct)
    pn, pa = texts(pool, args.level, dct)
    Q, P = vectorize([a + " " + b for a, b in zip(sn, sa)], [a + " " + b for a, b in zip(pn, pa)], args.vec)
    write_cands(args, tag, s1, pool, Q, P, own)


def run_scaling(args):
    tags = sorted((os.path.basename(p) for p in glob.glob(os.path.join(args.work, "universes", "F*"))),
                  key=lambda t: float(t[1:]))
    dct = load_dict(os.path.join(args.work, "models", "translit_dict.json"))
    rows = []
    for tag in tags:
        f = float(tag[1:])
        s1, pool = load_universe(args, tag)
        own = owner_index(s1, pool)
        k, sc, pc = s1.k.values, s1.country.values, pool.country.values
        sn, sa = texts(s1, args.level, dct)
        pn, pa = texts(pool, args.level, dct)
        Q, P = vectorize([a + " " + b for a, b in zip(sn, sa)], [a + " " + b for a, b in zip(pn, pa)], args.vec)
        t0 = time.time()
        fi, _ = knn_grouped(Q, P, sc, pc, 50, args.n_jobs)
        tf = time.time() - t0
        t0 = time.time()
        ri, _ = knn_grouped(P, Q, pc, sc, 3, args.n_jobs)
        tr = time.time() - t0
        for name, pr in (("fwd10", fwd_pairs(fi, 10)), ("fwd50", fwd_pairs(fi, 50)), ("rev3", rev_pairs(ri, 3)),
                         ("fwd5+rev3", union(fwd_pairs(fi, 5), rev_pairs(ri, 3))),
                         ("fwd10+rev3", union(fwd_pairs(fi, 10), rev_pairs(ri, 3)))):
            st = blocking_stats(*pr, own, k)
            rows.append({"cfg": name, "f": f, "pool": len(pool), "recall": st["pair_recall"],
                         "ceiling": st["ceiling_f05"], "cand_mean": st["cand_mean"]})
            log_result(args, "02_scaling", f"{name}|{tag}", frac=f, pool=len(pool), s1=len(s1),
                       fwd_seconds=tf, rev_seconds=tr, **st)
    df = pd.DataFrame(rows)
    for cfg, g in df.groupby("cfg"):
        if len(g) < 2:
            continue
        x = np.log10(g.pool.values.astype(float))  # pool size of each universe
        full = np.log10((g.pool.values / g.f.values).mean())  # pool size at 100% of train
        pr = np.polyfit(x, g.recall.values, 1)
        pc_ = np.polyfit(x, g.ceiling.values, 1)
        log_result(args, "02_scaling_extrapolation", cfg, slope_recall_per_decade=float(pr[0]),
                   recall_at_full=float(np.polyval(pr, full)), ceiling_at_full=float(np.polyval(pc_, full)),
                   full_pool=float(10 ** full))


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--vec", default="char_wb24")
    ap.add_argument("--k-fwd", type=int, default=50)
    ap.add_argument("--m-rev", type=int, default=3)
    ap.add_argument("--key-cap", type=int, default=200)
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument("--scaling", action="store_true", help="also run the scaling curve on F* universes")
    ap.add_argument("--also-u2", action="store_true", help="also write U2 candidates (needed by 06/07)")
    ap.add_argument("--only", default="", help="comma list of: main,u2,scaling (default: main + flags)")
    args = ap.parse_args()
    setup(args, "02_blocking")
    parts = args.only.split(",") if args.only else ["main"] + (["u2"] if args.also_u2 else []) + \
        (["scaling"] if args.scaling else [])
    if "main" in parts:
        run_main(args)
    if "u2" in parts:
        step("U2 candidates")
        run_u2(args)
    if "scaling" in parts:
        step("scaling curve")
        run_scaling(args)


if __name__ == "__main__":
    main()
