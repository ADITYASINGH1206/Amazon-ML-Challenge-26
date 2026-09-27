"""Record preparation and pairwise feature computation (parallel over chunks with fork on Linux/macOS)."""
import math
import multiprocessing as mp
import sys
from collections import Counter, defaultdict

import numpy as np

from .text import (GENERIC_NAME, components, first_number, has_indic, normalize_series, numbers, skel, zips)

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    HAVE_RF = True
except ImportError:
    HAVE_RF = False

GROUPS = {
    "CTX": ["cos", "fwd_rank", "rev_rank", "gap_q", "rev_margin", "mutual", "n_cand_q", "is_s3"],
    "NAME": ["name_tri_j", "name_tok_j", "core_tok_j", "core_contain", "core_exact", "nospace_eq",
             "nospace_contain", "skel_j", "name_jw", "name_tsr", "name_tsort", "name_partial", "core_idf_q",
             "core_idf_c", "name_ntok_q", "name_ntok_c", "c_indic"],
    "ADDR": ["addr_tri_j", "addr_tok_j", "addr_contain", "addr_idf_q", "addr_idf_c", "addr_tsr", "addr_empty_c",
             "addr_ntok_q", "addr_ntok_c"],
    "COMP": ["comp_n_q", "comp_n_c", "comp_mean_best", "comp_frac85", "comp_min_best"],
    "NUM": ["num_n_q", "num_n_c", "num_j", "num_share", "num_contain_c", "num_conflict", "first_eq", "zip_eq"],
}
PAIR_FEATURES = GROUPS["NAME"] + GROUPS["ADDR"] + GROUPS["COMP"] + GROUPS["NUM"]


def prep_records(df, level, dct=None, with_comps=True):
    countries = df.country.values
    name = normalize_series(df.name.values, countries, level, "name", dct)
    addr = normalize_series(df.addr.values, countries, level, "addr", dct)
    core, core_str, skels = [], [], []
    for n in name:
        t = n.split()
        c = [x for x in t if x not in GENERIC_NAME] or t
        core.append(frozenset(c))
        core_str.append(" ".join(c))
        skels.append(frozenset(skel(x) for x in c))
    R = {
        "name": name, "addr": addr, "core": core, "core_str": core_str, "skel": skels,
        "name_tok": [frozenset(n.split()) for n in name], "addr_tok": [frozenset(a.split()) for a in addr],
        "nums": [numbers(a) for a in df.addr.values], "first": [first_number(a) for a in df.addr.values],
        "zips": [zips(a) for a in df.addr.values], "indic": [has_indic(n) for n in df.name.values],
        "country": list(countries),
    }
    if with_comps:
        cache = {}
        comps = []
        for a, c in zip(df.addr.values, countries):
            key = (a, c)
            r = cache.get(key)
            if r is None:
                r = components(a, level, c, dct)
                cache[key] = r
            comps.append(r)
        R["comps"] = comps
    return R


def build_idf(recs_list):
    """Per-country document frequencies for core-name tokens and address tokens."""
    dfn, dfa, N = defaultdict(Counter), defaultdict(Counter), Counter()
    for R in recs_list:
        for c, core, at in zip(R["country"], R["core"], R["addr_tok"]):
            N[c] += 1
            dfn[c].update(core)
            dfa[c].update(at)
    idf = {}
    for c in N:
        n = N[c]
        idf[c] = ({t: math.log((n + 1) / (v + 1)) + 1 for t, v in dfn[c].items()},
                  {t: math.log((n + 1) / (v + 1)) + 1 for t, v in dfa[c].items()},
                  math.log(n + 1) + 1)
    return idf


def _tri(s):
    s = f" {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) > 2 else set()


def _jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _cont(a, b):
    m = min(len(a), len(b))
    return len(a & b) / m if m else 0.0


def _idf_overlap(a, b, idf, dflt):
    tot = sum(idf.get(t, dflt) for t in a)
    return sum(idf.get(t, dflt) for t in (a & b)) / tot if tot else 0.0


_G = {}


def _feats_range(bounds):
    lo, hi = bounds
    Q, P, qi, pj, idf = _G["Q"], _G["P"], _G["qi"], _G["pj"], _G["idf"]
    n = hi - lo
    out = {f: np.full(n, np.nan, np.float32) for f in PAIR_FEATURES}
    for r in range(n):
        a, b = qi[lo + r], pj[lo + r]
        cty = Q["country"][a]
        idfn, idfa, dflt = idf.get(cty, ({}, {}, 10.0))
        nq, nc = Q["name"][a], P["name"][b]
        cq, cc = Q["core"][a], P["core"][b]
        csq, csc = Q["core_str"][a], P["core_str"][b]
        out["name_tri_j"][r] = _jac(_tri(nq), _tri(nc))
        out["name_tok_j"][r] = _jac(Q["name_tok"][a], P["name_tok"][b])
        out["core_tok_j"][r] = _jac(cq, cc)
        out["core_contain"][r] = _cont(cq, cc)
        out["core_exact"][r] = float(csq == csc)
        nsq, nsc = csq.replace(" ", ""), csc.replace(" ", "")
        out["nospace_eq"][r] = float(nsq == nsc)
        out["nospace_contain"][r] = float(len(nsq) >= 4 and len(nsc) >= 4 and (nsq in nsc or nsc in nsq))
        out["skel_j"][r] = _jac(Q["skel"][a], P["skel"][b])
        if HAVE_RF:
            out["name_jw"][r] = JaroWinkler.normalized_similarity(csq, csc)
            out["name_tsr"][r] = fuzz.token_set_ratio(nq, nc) / 100
            out["name_tsort"][r] = fuzz.token_sort_ratio(csq, csc) / 100
            out["name_partial"][r] = fuzz.partial_ratio(nsq, nsc) / 100
        out["core_idf_q"][r] = _idf_overlap(cq, cc, idfn, dflt)
        out["core_idf_c"][r] = _idf_overlap(cc, cq, idfn, dflt)
        out["name_ntok_q"][r] = len(Q["name_tok"][a])
        out["name_ntok_c"][r] = len(P["name_tok"][b])
        out["c_indic"][r] = float(P["indic"][b])

        aq, ac = Q["addr"][a], P["addr"][b]
        tq, tc = Q["addr_tok"][a], P["addr_tok"][b]
        out["addr_empty_c"][r] = float(not ac)
        out["addr_ntok_q"][r] = len(tq)
        out["addr_ntok_c"][r] = len(tc)
        if ac and aq:
            out["addr_tri_j"][r] = _jac(_tri(aq), _tri(ac))
            out["addr_tok_j"][r] = _jac(tq, tc)
            out["addr_contain"][r] = _cont(tq, tc)
            out["addr_idf_q"][r] = _idf_overlap(tq, tc, idfa, dflt)
            out["addr_idf_c"][r] = _idf_overlap(tc, tq, idfa, dflt)
            if HAVE_RF:
                out["addr_tsr"][r] = fuzz.token_set_ratio(aq, ac) / 100

        if "comps" in Q:
            kq, kc = Q["comps"][a], P["comps"][b]
            out["comp_n_q"][r] = len(kq)
            out["comp_n_c"][r] = len(kc)
            if kq and kc and HAVE_RF:
                short, long_ = (kq, kc) if len(kq) <= len(kc) else (kc, kq)
                best = [max(fuzz.ratio(x, y) for y in long_) / 100 for x in short]
                out["comp_mean_best"][r] = sum(best) / len(best)
                out["comp_frac85"][r] = sum(v >= 0.85 for v in best) / len(best)
                out["comp_min_best"][r] = min(best)

        uq, uc = Q["nums"][a], P["nums"][b]
        out["num_n_q"][r] = len(uq)
        out["num_n_c"][r] = len(uc)
        if uq and uc:
            inter = len(uq & uc)
            out["num_j"][r] = inter / len(uq | uc)
            out["num_share"][r] = float(inter > 0)
            out["num_contain_c"][r] = inter / len(uc)
            out["num_conflict"][r] = float(inter == 0)
        fq, fc = Q["first"][a], P["first"][b]
        if fq and fc:
            out["first_eq"][r] = float(fq == fc)
        zq, zc = Q["zips"][a], P["zips"][b]
        if zq and zc:
            out["zip_eq"][r] = float(bool(zq & zc))
    return out


def _init_spawn(idf):
    _G["idf"] = idf


def _feats_chunk_spawn(payload):
    Qs, Ps, qi_l, pj_l = payload
    _G.update(Q=Qs, P=Ps, qi=qi_l, pj=pj_l)
    return _feats_range((0, len(qi_l)))


def _subset(R, idx):
    return {k: [v[i] for i in idx] for k, v in R.items()}


def _spawn_payloads(Q, P, qi, pj, bounds):
    for lo, hi in bounds:
        uq, iq = np.unique(qi[lo:hi], return_inverse=True)
        up, ip = np.unique(pj[lo:hi], return_inverse=True)
        yield _subset(Q, uq), _subset(P, up), iq, ip


def compute_pair_features(Q, P, qi, pj, idf, n_jobs=1, chunk=50_000):
    qi, pj = np.asarray(qi), np.asarray(pj)
    bounds = [(s, min(s + chunk, len(qi))) for s in range(0, len(qi), chunk)]
    methods = mp.get_all_start_methods()
    if n_jobs > 1 and len(bounds) > 1 and "fork" in methods:  # Linux / macOS: workers share memory
        _G.update(Q=Q, P=P, qi=qi, pj=pj, idf=idf)
        with mp.get_context("fork").Pool(n_jobs) as pool:
            parts = pool.map(_feats_range, bounds)
    elif n_jobs > 1 and len(bounds) > 1:  # Windows: ship only the records each chunk needs
        with mp.get_context("spawn").Pool(n_jobs, initializer=_init_spawn, initargs=(idf,)) as pool:
            parts = list(pool.imap(_feats_chunk_spawn, _spawn_payloads(Q, P, qi, pj, bounds), chunksize=1))
    else:
        _G.update(Q=Q, P=P, qi=qi, pj=pj, idf=idf)
        parts = [_feats_range(b) for b in bounds]
    _G.clear()
    if not parts:
        return {f: np.zeros(0, np.float32) for f in PAIR_FEATURES}
    return {f: np.concatenate([p[f] for p in parts]) for f in PAIR_FEATURES}


def add_context_features(c, q_top1, p_top1, pool_src):
    """c: candidates df with qi, pj, cos, fwd_rank, rev_rank."""
    c["gap_q"] = (q_top1[c.qi.values] - c.cos.values).astype(np.float32)
    c["rev_margin"] = (p_top1[c.pj.values] - c.cos.values).astype(np.float32)
    c["mutual"] = ((c.fwd_rank == 0) & (c.rev_rank == 0)).astype(np.float32)
    c["n_cand_q"] = c.groupby("qi").pj.transform("size").astype(np.float32)
    c["is_s3"] = (pool_src[c.pj.values] == "S3").astype(np.float32)
    return c
