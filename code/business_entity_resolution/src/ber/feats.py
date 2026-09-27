"""Pairwise features (name / address / number agreement) computed in parallel worker processes.

Each task carries only the records its pairs need (works with Windows 'spawn'); names/addresses arrive already
L4-normalised, so the workers only build token sets and compute similarities."""
import multiprocessing as mp

import numpy as np

from .config import PAIR_FEATURES
from .text import GENERIC_NAME, first_number, has_indic, numbers, skel, zips
from .util import log

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
except ImportError as e:  # the matcher was trained with these features
    raise SystemExit("rapidfuzz is required: pip install rapidfuzz") from e


def records(nname, naddr, raw_name, raw_addr, country):
    core, core_str, skels = [], [], []
    for n in nname:
        t = n.split()
        c = [x for x in t if x not in GENERIC_NAME] or t
        core.append(frozenset(c))
        core_str.append(" ".join(c))
        skels.append(frozenset(skel(x) for x in c))
    return {"name": list(nname), "addr": list(naddr), "core": core, "core_str": core_str, "skel": skels,
            "name_tok": [frozenset(n.split()) for n in nname], "addr_tok": [frozenset(a.split()) for a in naddr],
            "nums": [numbers(a) for a in raw_addr], "first": [first_number(a) for a in raw_addr],
            "zips": [zips(a) for a in raw_addr], "indic": [has_indic(n) for n in raw_name], "country": list(country)}


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


def pair_features(Q, P, qi, pj, idf):
    n = len(qi)
    out = {f: np.full(n, np.nan, np.float32) for f in PAIR_FEATURES}
    for r in range(n):
        a, b = qi[r], pj[r]
        idfn, idfa, dflt = idf.get(Q["country"][a], ({}, {}, 10.0))
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
            out["addr_tsr"][r] = fuzz.token_set_ratio(aq, ac) / 100

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


def build_tri_idf(nname, naddr, country, max_per_country=1_000_000, seed=0):
    """Per-country IDF of character trigrams (binary tf) for names and addresses, from up to 1M records."""
    import math
    from collections import Counter
    import pandas as pd
    rng = np.random.default_rng(seed)
    out = {}
    for c in pd.unique(country):
        idx = np.where(country == c)[0]
        if len(idx) > max_per_country:
            idx = rng.choice(idx, max_per_country, replace=False)
        dn, da = Counter(), Counter()
        for i in idx:
            dn.update(_tri(nname[i]))
            if naddr[i]:
                da.update(_tri(naddr[i]))
        n = len(idx)
        out[c] = ({t: math.log((n + 1) / (v + 1)) + 1 for t, v in dn.items()},
                  {t: math.log((n + 1) / (v + 1)) + 1 for t, v in da.items()}, math.log(n + 1) + 1)
    return out


def _wcos(A, B, w, dflt):
    """IDF-weighted (binary tf) cosine and weighted containment of two trigram sets."""
    if not A or not B:
        return np.nan, np.nan
    sa = sum(w.get(t, dflt) ** 2 for t in A)
    sb = sum(w.get(t, dflt) ** 2 for t in B)
    si = sum(w.get(t, dflt) ** 2 for t in (A & B))
    return si / (sa * sb) ** 0.5, si / min(sa, sb)


def _maxw(tokens, w, dflt):
    return max((w.get(t, dflt) for t in tokens), default=0.0)


def pair_features_v2(Q, P, qi, pj, idf, tri):
    """TF-IDF-style trigram similarity, name specificity and rare-token disagreement."""
    from .config import PAIR_V2
    n = len(qi)
    out = {f: np.full(n, np.nan, np.float32) for f in PAIR_V2}
    qcache = {}
    for r in range(n):
        a, b = qi[r], pj[r]
        cty = Q["country"][a]
        tn, ta, tdflt = tri.get(cty, ({}, {}, 10.0))
        idfn, idfa, dflt = idf.get(cty, ({}, {}, 10.0))
        if a not in qcache:
            qcache[a] = (_tri(Q["name"][a]), _tri(Q["addr"][a]) if Q["addr"][a] else set())
        qn3, qa3 = qcache[a]
        pn3 = _tri(P["name"][b])
        out["name_tri_idfcos"][r], out["name_tri_idfcont"][r] = _wcos(qn3, pn3, tn, tdflt)
        if P["addr"][b] and qa3:
            out["addr_tri_idfcos"][r], out["addr_tri_idfcont"][r] = _wcos(qa3, _tri(P["addr"][b]), ta, tdflt)
        cq, cc = Q["core"][a], P["core"][b]
        out["core_info_q"][r] = sum(idfn.get(t, dflt) for t in cq)
        out["core_info_c"][r] = sum(idfn.get(t, dflt) for t in cc)
        out["shared_max_idf"][r] = _maxw(cq & cc, idfn, dflt)
        out["unshared_max_idf_q"][r] = _maxw(cq - cc, idfn, dflt)
        out["unshared_max_idf_c"][r] = _maxw(cc - cq, idfn, dflt)
        tq = {t for t in Q["addr_tok"][a] if not t.isdigit()}
        tc = {t for t in P["addr_tok"][b] if not t.isdigit()}
        if tq and tc:
            out["addr_unshared_max_idf_q"][r] = _maxw(tq - tc, idfa, dflt)
            out["addr_unshared_max_idf_c"][r] = _maxw(tc - tq, idfa, dflt)
        out["name_ratio"][r] = fuzz.ratio(Q["name"][a], P["name"][b]) / 100
        out["name_jw_full"][r] = JaroWinkler.normalized_similarity(Q["name"][a], P["name"][b])
    return out


_W = {}


def bounded_imap(pool, fn, tasks, window):
    """Ordered results with at most `window` tasks in flight (keeps queued payloads out of RAM)."""
    from collections import deque
    pending = deque()
    for t in tasks:
        pending.append(pool.apply_async(fn, (t,)))
        if len(pending) >= window:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


def _init(idf, tri=None):
    _W["idf"] = idf
    _W["tri"] = tri


def _task(t):
    qcols, pcols, iq, ip = t
    return pair_features(records(*qcols), records(*pcols), iq, ip, _W["idf"])


def _task_v2(t):
    qcols, pcols, iq, ip = t
    return pair_features_v2(records(*qcols), records(*pcols), iq, ip, _W["idf"], _W["tri"])


def compute(S, Pl, qi, pj, idf, workers, chunk=40_000, tri=None):
    """S / Pl: dicts of arrays nname, naddr, name, addr, country for S1 / pool rows. Returns dict of feature arrays
    aligned with (qi, pj): the base pair features, or the V2 features when `tri` (trigram IDF tables) is given."""
    from .config import PAIR_V2
    PAIR = PAIR_V2 if tri is not None else PAIR_FEATURES
    keys = ("nname", "naddr", "name", "addr", "country")

    def tasks():
        for s in range(0, len(qi), chunk):
            uq, iq = np.unique(qi[s:s + chunk], return_inverse=True)
            up, ip = np.unique(pj[s:s + chunk], return_inverse=True)
            yield [S[k][uq] for k in keys], [Pl[k][up] for k in keys], iq, ip

    parts = []
    n_tasks = (len(qi) + chunk - 1) // chunk
    fn = _task_v2 if tri is not None else _task
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(idf, tri)) as pool:
        for i, part in enumerate(bounded_imap(pool, fn, tasks(), 2 * workers)):
            parts.append(part)
            if (i + 1) % 50 == 0 or i + 1 == n_tasks:
                log(f"  {'V2' if tri is not None else 'pair'} features {min((i + 1) * chunk, len(qi)):,}/{len(qi):,}")
    if not parts:
        return {f: np.zeros(0, np.float32) for f in PAIR}
    return {f: np.concatenate([p[f] for p in parts]) for f in PAIR}
