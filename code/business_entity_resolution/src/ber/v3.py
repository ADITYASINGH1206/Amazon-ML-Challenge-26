"""V3 features aimed at the two largest error causes found on the held-out slice:
  * pool copies with an EMPTY address (66% of rejected true matches): how unique is the name? A copy named
    exactly like one visible S1 of the country, and shared by few pool records, is almost surely that S1's.
  * house-number typos ('6420' vs '642', '104' vs '04', '24' vs '017'): fuzzy number agreement.
Counts use only the S1 records visible to the search (training hides a slice, like the test set)."""
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from .text import GENERIC_NAME, numbers

PAIR_V3 = ["n_s1_core_q", "n_s1_core_c", "n_pool_core_c", "n_pool_core_q", "core_c_eq_unique_s1",
           "num_min_edit", "num_contain", "num_min_reldiff", "first_edit",
           "namesake_agree", "namesake_conflict"]


def core_names(nname):
    out = []
    for n in nname:
        t = n.split()
        out.append(" ".join([x for x in t if x not in GENERIC_NAME] or t))
    return np.array(out, dtype=object)


def _key(country, core):
    return pd.Series(country, dtype=object).str.cat(pd.Series(core, dtype=object), sep="\t").values


def v3_features(qi, pj, s1_nname, s1_country, s1_addr, s1_present, pool_nname, pool_country, pool_addr):
    s1_core, pool_core = core_names(s1_nname), core_names(pool_nname)
    ks, kp = _key(s1_country, s1_core), _key(pool_country, pool_core)
    
    # Pre-group pool indices by core name for OPT-4 (namesake features)
    ukc_set = set(kp[pj])
    from collections import defaultdict
    kc_to_pnums = defaultdict(list)
    for j in range(len(kp)):
        kc_val = kp[j]
        if kc_val in ukc_set:
            nums = numbers(pool_addr[j])
            if nums:
                kc_to_pnums[kc_val].append(nums)

    s1_cnt = pd.Series(ks[s1_present]).value_counts()
    pool_cnt = pd.Series(kp).value_counts()
    kq, kc = ks[qi], kp[pj]
    out = {}
    out["n_s1_core_q"] = pd.Series(kq).map(s1_cnt).fillna(0).values.astype(np.float32)
    out["n_s1_core_c"] = pd.Series(kc).map(s1_cnt).fillna(0).values.astype(np.float32)
    out["n_pool_core_c"] = pd.Series(kc).map(pool_cnt).fillna(0).values.astype(np.float32)
    out["n_pool_core_q"] = pd.Series(kq).map(pool_cnt).fillna(0).values.astype(np.float32)
    out["core_c_eq_unique_s1"] = ((kq == kc) & (out["n_s1_core_c"] == 1)).astype(np.float32)
    uq, up = np.unique(qi), np.unique(pj)
    qn = {int(i): numbers(s1_addr[i]) for i in uq}
    pn = {int(j): numbers(pool_addr[j]) for j in up}
    n = len(qi)
    ed = np.full(n, np.nan, np.float32)
    ct = np.full(n, np.nan, np.float32)
    rd = np.full(n, np.nan, np.float32)
    fe = np.full(n, np.nan, np.float32)
    ns_agree = np.zeros(n, np.float32)
    ns_conflict = np.zeros(n, np.float32)
    first_q = {i: _first(s1_addr[i]) for i in qn}
    first_p = {j: _first(pool_addr[j]) for j in pn}
    for r in range(n):
        a, b = qn[int(qi[r])], pn[int(pj[r])]
        if a and b:
            best_e, best_d, cont = 99, 9.0, 0.0
            for x in a:
                for y in b:
                    e = Levenshtein.distance(x, y)
                    if e < best_e:
                        best_e = e
                    if x != y and len(x) >= 2 and len(y) >= 2 and (x in y or y in x):
                        cont = 1.0
                    if len(x) <= 9 and len(y) <= 9:
                        xi, yi = int(x), int(y)
                        dd = abs(xi - yi) / max(xi, yi, 1)
                        if dd < best_d:
                            best_d = dd
            ed[r], ct[r], rd[r] = best_e, cont, best_d
        fq, fp = first_q[int(qi[r])], first_p[int(pj[r])]
        if fq and fp:
            fe[r] = Levenshtein.distance(fq, fp)
            
        # OPT-4: Namesake agreement/conflict
        if q_nums := qn[int(qi[r])]:
            ag, cf = 0, 0
            for p_nums in kc_to_pnums[kc[r]]:
                if q_nums & p_nums:
                    ag += 1
                else:
                    cf += 1
            ns_agree[r] = ag
            ns_conflict[r] = cf
            
    out.update(num_min_edit=ed, num_contain=ct, num_min_reldiff=rd, first_edit=fe,
               namesake_agree=ns_agree, namesake_conflict=ns_conflict)
    return out


def _first(addr):
    import re
    m = re.search(r"\d+", addr)
    return (m.group(0).lstrip("0") or "0") if m else ""
