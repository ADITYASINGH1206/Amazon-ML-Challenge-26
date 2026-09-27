"""V4 features targeting the France gap and descriptor-word swaps:
  * DESCRIPTOR VOCABULARY: tokens that are "swappable" between S1 and pool names (e.g. Club, Sportif, Amicale in
    France; Inc, Service, Group in the US). Learned from training pairs for US/India, and from confident same-address
    pairs for France (no labels). These supplement GENERIC_NAME without changing the existing core name logic.
  * FEATURES:
    - core_nodesc_j: Jaccard of core names after removing descriptors -- captures equivalence when only a descriptor
      token differs ('Club Sportif X' vs 'X Jeunes')
    - desc_swap_count: number of unshared tokens between the two core names that are descriptor-class
    - initials_match: 1 if one side's initials spell out the other side (e.g. 'LC' vs 'Latelier Centre')
    - name_len_ratio: ratio of shorter to longer name length (catches truncation)
    - core_alpha_j: Jaccard on alphabetic-only core tokens (no numbers)
"""
import numpy as np
import pandas as pd
from collections import Counter

from .text import GENERIC_NAME

PAIR_V4 = ["core_nodesc_j", "desc_swap_count", "initials_match",
           "name_len_ratio", "core_alpha_j"]

# ---- descriptor vocabulary learning --------------------------------------------------------

def _learn_descriptors_labelled(s1_nname, pool_nname, qi, pj, label, country,
                                 min_rate=0.05, min_count=50):
    """Learn per-country descriptor words from labelled true pairs.
    A token is a 'descriptor' if it appears in one name but not the other in >min_rate of true pairs.
    Only considers non-GENERIC tokens that appear in >=min_count true pairs."""
    desc = {}
    for c in pd.unique(country[qi]):
        mask = (label == 1) & (country[qi] == c)
        swap_cnt = Counter()   # token -> number of true pairs where it appears on one side but not the other
        total_cnt = Counter()  # token -> number of true pairs where it appears at all
        for q, p in zip(qi[mask], pj[mask]):
            qn = set(s1_nname[q].split()) - GENERIC_NAME
            pn = set(pool_nname[p].split()) - GENERIC_NAME
            sym_diff = (qn - pn) | (pn - qn)
            union = qn | pn
            for t in union:
                total_cnt[t] += 1
            for t in sym_diff:
                swap_cnt[t] += 1
        country_desc = set()
        for t, cnt in swap_cnt.items():
            if total_cnt[t] >= min_count and cnt / total_cnt[t] >= min_rate:
                country_desc.add(t)
        desc[c] = country_desc
    return desc


def _learn_descriptors_unlabelled(s1_nname, pool_nname, qi, pj, prob, s1_addr, pool_addr,
                                    country, min_rate=0.05, min_count=30, p_thresh=0.8):
    """Learn descriptor words for unlabelled countries (France) from confident same-address predictions."""
    from .text import numbers
    desc = {}
    for c in pd.unique(country[qi]):
        mask_c = country[qi] == c
        # confident matches: high probability AND sharing a house number
        confident = []
        for idx in np.where(mask_c)[0]:
            if prob[idx] < p_thresh:
                continue
            q, p = qi[idx], pj[idx]
            qn = numbers(s1_addr[q])
            pn = numbers(pool_addr[p])
            if qn and pn and qn & pn:
                confident.append(idx)
        if len(confident) < 100:
            # too few confident pairs, use probability alone
            confident = np.where(mask_c & (prob >= p_thresh))[0]

        swap_cnt = Counter()
        total_cnt = Counter()
        for idx in confident:
            q, p = qi[idx], pj[idx]
            qn = set(s1_nname[q].split()) - GENERIC_NAME
            pn = set(pool_nname[p].split()) - GENERIC_NAME
            sym_diff = (qn - pn) | (pn - qn)
            union = qn | pn
            for t in union:
                total_cnt[t] += 1
            for t in sym_diff:
                swap_cnt[t] += 1
        country_desc = set()
        for t, cnt in swap_cnt.items():
            if total_cnt[t] >= min_count and cnt / total_cnt[t] >= min_rate:
                country_desc.add(t)
        desc[c] = country_desc
    return desc


def learn_descriptors(s1_nname, pool_nname, qi, pj, label, prob,
                      s1_country, s1_addr, pool_country, pool_addr,
                      labelled_countries=("US", "India")):
    """Learn descriptor vocabulary per country from training data.
    For labelled countries (US, India): uses true-pair label.
    For unlabelled countries (France): uses confident same-address predictions."""
    desc = {}
    cq = s1_country[qi]

    # Labelled countries
    lab_mask = np.isin(cq, list(labelled_countries))
    if lab_mask.any():
        d = _learn_descriptors_labelled(
            s1_nname, pool_nname, qi[lab_mask], pj[lab_mask], label[lab_mask], cq[lab_mask])
        desc.update(d)

    # Unlabelled countries
    unlab_countries = set(pd.unique(cq)) - set(labelled_countries)
    if unlab_countries:
        unlab_mask = np.isin(cq, list(unlab_countries))
        if unlab_mask.any():
            d = _learn_descriptors_unlabelled(
                s1_nname, pool_nname, qi[unlab_mask], pj[unlab_mask], prob[unlab_mask],
                s1_addr, pool_addr, cq[unlab_mask])
            desc.update(d)
    return desc


# ---- feature computation ------------------------------------------------------------------

def _core(nname):
    t = nname.split()
    return [x for x in t if x not in GENERIC_NAME] or t


def _core_no_desc(nname, descs):
    t = nname.split()
    c = [x for x in t if x not in GENERIC_NAME and x not in descs]
    return c or [x for x in t if x not in GENERIC_NAME] or t


def _initials(tokens):
    """First letter of each alphabetic token."""
    return "".join(t[0] for t in tokens if t.isalpha())


def _check_initials(core_q, core_c):
    """Check if one side's tokens' initials spell the other side, or vice versa."""
    if not core_q or not core_c:
        return 0.0
    # If one side is very short (1-3 chars, all alpha), check if it matches the other's initials
    sq = "".join(core_q)
    sc = "".join(core_c)
    iq = _initials(core_q)
    ic = _initials(core_c)
    # Case 1: pool name is initials of S1 name (or vice versa)
    if len(sc) <= 4 and sc.isalpha() and iq and (sc == iq or sc.upper() == iq.upper()):
        return 1.0
    if len(sq) <= 4 and sq.isalpha() and ic and (sq == ic or sq.upper() == ic.upper()):
        return 1.0
    # Case 2: S1 initials match pool initials (weaker signal)
    if len(iq) >= 2 and len(ic) >= 2 and iq == ic:
        return 0.5
    return 0.0


def _jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def v4_features(qi, pj, s1_nname, pool_nname, s1_country, descriptors):
    """Compute V4 features for candidate pairs.
    descriptors: dict country -> set of descriptor tokens."""
    n = len(qi)
    out = {f: np.full(n, np.nan, np.float32) for f in PAIR_V4}

    for r in range(n):
        q, p = qi[r], pj[r]
        cty = s1_country[q]
        desc = descriptors.get(cty, set())

        # Core names (standard)
        cq = _core(s1_nname[q])
        cc = _core(pool_nname[p])

        # Core names after removing descriptors
        cqd = _core_no_desc(s1_nname[q], desc)
        ccd = _core_no_desc(pool_nname[p], desc)

        # 1. Jaccard after removing descriptors
        out["core_nodesc_j"][r] = _jac(frozenset(cqd), frozenset(ccd))

        # 2. Descriptor swap count: unshared tokens that are descriptor-class
        sq, sp = frozenset(cq), frozenset(cc)
        unshared = (sq - sp) | (sp - sq)
        out["desc_swap_count"][r] = sum(1 for t in unshared if t in desc)

        # 3. Initials match
        out["initials_match"][r] = _check_initials(cq, cc)

        # 4. Name length ratio (shorter / longer, catches truncation)
        lq = len(" ".join(cq))
        lc = len(" ".join(cc))
        out["name_len_ratio"][r] = min(lq, lc) / max(lq, lc, 1)

        # 5. Jaccard on alphabetic-only core tokens
        alpha_q = frozenset(t for t in cq if t.isalpha())
        alpha_c = frozenset(t for t in cc if t.isalpha())
        out["core_alpha_j"][r] = _jac(alpha_q, alpha_c)

    return out
