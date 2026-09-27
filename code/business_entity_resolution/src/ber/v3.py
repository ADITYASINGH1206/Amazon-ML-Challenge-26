"""V3 features aimed at the two largest error causes found on the held-out slice:
  * pool copies with an EMPTY address (66% of rejected true matches): how unique is the name? A copy named
    exactly like one visible S1 of the country, and shared by few pool records, is almost surely that S1's.
  * house-number typos ('6420' vs '642', '104' vs '04', '24' vs '017'): fuzzy number agreement.
Counts use only the S1 records visible to the search (training hides a slice, like the test set)."""
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from .text import GENERIC_NAME, numbers
from .util import log

PAIR_V3 = ["n_s1_core_q", "n_s1_core_c", "n_pool_core_c", "n_pool_core_q", "core_c_eq_unique_s1",
           "num_min_edit", "num_contain", "num_min_reldiff", "first_edit",
           "namesake_agree", "namesake_conflict"]


def core_names(nname):
    out = []
    for n in nname:
        if not n or not isinstance(n, str):
            out.append("")
            continue
        t = n.split()
        out.append(" ".join([x for x in t if x not in GENERIC_NAME] or t))
    return np.array(out, dtype=object)


def _key(country, core):
    return pd.Series(country, dtype=object).str.cat(pd.Series(core, dtype=object), sep="\t").values


def _first(addr):
    if not addr or not isinstance(addr, str):
        return ""
    m = re.search(r"\d+", addr)
    return (m.group(0).lstrip("0") or "0") if m else ""


def v3_features(qi, pj, s1_nname, s1_country, s1_addr, s1_present, pool_nname, pool_country, pool_addr):
    s1_core, pool_core = core_names(s1_nname), core_names(pool_nname)
    ks, kp = _key(s1_country, s1_core), _key(pool_country, pool_core)

    unique_pj = np.unique(pj)
    ukc_set = set(kp[unique_pj])

    # Pre-group pool numbers by core name for OPT-4 (capped at 50 per name to prevent memory explosion)
    kc_to_pnums = defaultdict(list)
    for j in range(len(kp)):
        kc_val = kp[j]
        if kc_val in ukc_set:
            if len(kc_to_pnums[kc_val]) < 50:
                nums = numbers(pool_addr[j])
                if nums and nums not in kc_to_pnums[kc_val]:
                    kc_to_pnums[kc_val].append(nums)

    # Vectorized count lookups: O(1) memory, avoids 14.7M string map tables
    s1_c = Counter(ks[s1_present])
    s1_cnt_s1 = np.fromiter((s1_c.get(k, 0) for k in ks), dtype=np.float32, count=len(ks))
    s1_cnt_pool = np.fromiter((s1_c.get(k, 0) for k in kp), dtype=np.float32, count=len(kp))

    pool_c = Counter(kp)
    pool_cnt_pool = np.fromiter((pool_c.get(k, 0) for k in kp), dtype=np.float32, count=len(kp))
    pool_cnt_s1 = np.fromiter((pool_c.get(k, 0) for k in ks), dtype=np.float32, count=len(ks))

    key_map = {k: i for i, k in enumerate(set(ks).union(kp))}
    ks_ids = np.fromiter((key_map[k] for k in ks), dtype=np.int32, count=len(ks))
    kp_ids = np.fromiter((key_map[k] for k in kp), dtype=np.int32, count=len(kp))

    out = {}
    out["n_s1_core_q"] = s1_cnt_s1[qi]
    out["n_s1_core_c"] = s1_cnt_pool[pj]
    out["n_pool_core_c"] = pool_cnt_pool[pj]
    out["n_pool_core_q"] = pool_cnt_s1[qi]
    out["core_c_eq_unique_s1"] = ((ks_ids[qi] == kp_ids[pj]) & (s1_cnt_pool[pj] == 1)).astype(np.float32)

    uq, up = np.unique(qi), unique_pj
    qn = {int(i): numbers(s1_addr[i]) for i in uq}
    pn = {int(j): numbers(pool_addr[j]) for j in up}
    first_q = {int(i): _first(s1_addr[i]) for i in uq}
    first_p = {int(j): _first(pool_addr[j]) for j in up}

    n = len(qi)
    ed = np.full(n, np.nan, np.float32)
    ct = np.full(n, np.nan, np.float32)
    rd = np.full(n, np.nan, np.float32)
    fe = np.full(n, np.nan, np.float32)
    ns_agree = np.zeros(n, np.float32)
    ns_conflict = np.zeros(n, np.float32)

    log_step = max(2_000_000, n // 10)
    for r in range(n):
        q_idx = int(qi[r])
        p_idx = int(pj[r])
        a = qn.get(q_idx)
        b = pn.get(p_idx)
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
        fq = first_q.get(q_idx)
        fp = first_p.get(p_idx)
        if fq and fp:
            fe[r] = Levenshtein.distance(fq, fp)

        # OPT-4: Namesake agreement/conflict
        if a:
            p_list = kc_to_pnums.get(kp[p_idx])
            if p_list:
                ag, cf = 0, 0
                for p_nums in p_list:
                    if a & p_nums:
                        ag += 1
                    else:
                        cf += 1
                ns_agree[r] = ag
                ns_conflict[r] = cf

        if (r + 1) % log_step == 0 or r + 1 == n:
            log(f"  V3 pairs {r + 1:,}/{n:,}")

    out.update(num_min_edit=ed, num_contain=ct, num_min_reldiff=rd, first_edit=fe,
               namesake_agree=ns_agree, namesake_conflict=ns_conflict)
    return out
