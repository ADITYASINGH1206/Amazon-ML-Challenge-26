"""Parallel L4 normalisation of every record + per-country IDF tables, and the learned transliteration dictionary."""
import multiprocessing as mp
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .config import DICT_MIN_PRIO, DICT_PAIRS, NORM_LEVEL, SEED
from .text import GENERIC_NAME, learn_dict, load_dict, normalize_series, save_dict
from .util import log

_W = {}


def _init(dct):
    _W["dct"] = dct


def _norm_chunk(t):
    names, addrs, countries = t
    dct = _W["dct"]
    nn = normalize_series(names, countries, NORM_LEVEL, "name", dct)
    na = normalize_series(addrs, countries, NORM_LEVEL, "addr", dct)
    dfn, dfa, N = defaultdict(Counter), defaultdict(Counter), Counter()
    for n, a, c in zip(nn, na, countries):
        t = n.split()
        dfn[c].update(set([x for x in t if x not in GENERIC_NAME] or t))
        dfa[c].update(set(a.split()))
        N[c] += 1
    return nn, na, dict(dfn), dict(dfa), N


def learn_translit_dict(work):
    path = os.path.join(work, "models", "translit_dict.json")
    if os.path.exists(path):
        return load_dict(path)
    from .data import load_split
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s1, pool = load_split(work, "train")
    keep = s1[s1.prio >= DICT_MIN_PRIO][["entity_id", "name", "addr"]]
    dp = pool[pool.owner.isin(set(keep.entity_id))][["owner", "name", "addr"]]
    dp = dp.merge(keep.rename(columns={"entity_id": "owner", "name": "q_name", "addr": "q_addr"}), on="owner")
    nonascii = (dp.name + dp.addr).map(lambda x: not x.isascii())
    a, b = dp[nonascii], dp[~nonascii]
    na = min(len(a), int(DICT_PAIRS * 0.8))
    nb = min(len(b), DICT_PAIRS - na)
    dp = pd.concat([a.sample(na, random_state=SEED), b.sample(nb, random_state=SEED)], ignore_index=True)
    log(f"learning native-script -> Latin dictionary from {len(dp):,} training pairs (S1 priority >= {DICT_MIN_PRIO})")
    dct = learn_dict(dp.q_name.values, dp.name.values, dp.q_addr.values, dp.addr.values, log=log)
    save_dict(dct, path)
    return dct


def normalize_split(work, split, dct, workers, chunk=100_000):
    """Adds nname/naddr parquet columns for S1 and pool; writes per-country IDF tables for the split."""
    from .data import load_split
    out1 = os.path.join(work, "cache", f"{split}_s1_norm.parquet")
    outp = os.path.join(work, "cache", f"{split}_pool_norm.parquet")
    out_idf = os.path.join(work, "cache", f"{split}_idf.pkl")
    if os.path.exists(out1) and os.path.exists(outp) and os.path.exists(out_idf):
        return
    s1, pool = load_split(work, split, columns=["name", "addr", "country"])
    dfn, dfa, N = defaultdict(Counter), defaultdict(Counter), Counter()
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(dct,)) as pp:
        for df, out in ((s1, out1), (pool, outp)):
            tasks = [(df.name.values[i:i + chunk], df.addr.values[i:i + chunk], df.country.values[i:i + chunk])
                     for i in range(0, len(df), chunk)]
            nn, na = [], []
            from .feats import bounded_imap
            for i, (a, b, cn, ca, cN) in enumerate(bounded_imap(pp, _norm_chunk, tasks, 2 * workers)):
                nn += a
                na += b
                for c, v in cn.items():
                    dfn[c].update(v)
                for c, v in ca.items():
                    dfa[c].update(v)
                N.update(cN)
                if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                    log(f"  {split}: normalised {min((i + 1) * chunk, len(df)):,}/{len(df):,}")
            pd.DataFrame({"nname": nn, "naddr": na}).to_parquet(out, index=False)
    # tokens seen once (mostly typos) are left out and get the default = rarest possible IDF; this shrinks the
    # tables ~4x, and every feature worker holds a copy
    idf = {}
    for c, n in N.items():
        idf[c] = ({t: math.log((n + 1) / (v + 1)) + 1 for t, v in dfn[c].items() if v >= 2},
                  {t: math.log((n + 1) / (v + 1)) + 1 for t, v in dfa[c].items() if v >= 2},
                  math.log(n + 1) + 1)
    with open(out_idf, "wb") as f:
        pickle.dump(idf, f)
    log(f"{split}: normalisation done; IDF tables for {sorted(idf)}")


def load_norm(work, split):
    d = os.path.join(work, "cache")
    s = pd.read_parquet(os.path.join(d, f"{split}_s1_norm.parquet"))
    p = pd.read_parquet(os.path.join(d, f"{split}_pool_norm.parquet"))
    with open(os.path.join(d, f"{split}_idf.pkl"), "rb") as f:
        idf = pickle.load(f)
    return s, p, idf
