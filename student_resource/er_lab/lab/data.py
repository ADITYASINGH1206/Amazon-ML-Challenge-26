"""Loading, parquet caching and 'universe' sampling.

A universe is a realistic sub-sample of the training data:
  * a random subset of S1 entities (by a fixed random priority),
  * ALL of their true S2/S3 matches,
  * the same fraction of unmatched S2/S3 'distractor' records.
Density (records per country) therefore scales with the fraction, and universes with a smaller
fraction are nested inside larger ones -> used for the blocking scaling experiment.
U1 = main evaluation universe, U2 = disjoint auxiliary universe used to TRAIN learned components
(translit dictionary, fine-tuned encoders, cross-encoder) so nothing leaks into U1 evaluation.
"""
import os

import numpy as np
import pandas as pd

from .common import step, udir

RENAME = {"business_name": "name", "business_address": "addr"}


def read_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def cache_path(args, name):
    return os.path.join(args.work, "cache", f"{name}.parquet")


def cache_sources(args):
    for split in ("train", "test"):
        for s in (1, 2, 3):
            out = cache_path(args, f"{split}_s{s}")
            src = os.path.join(args.data_dir, split, f"{split}_source{s}.tsv")
            if os.path.exists(out) or not os.path.exists(src):
                continue
            step(f"caching {src}")
            read_tsv(src).rename(columns=RENAME).to_parquet(out, index=False)
    out = cache_path(args, "train_pairs")
    if not os.path.exists(out):
        step("caching ground truth pairs")
        gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
        m = gt.matched_entity_ids.str.split(",")
        pairs = pd.DataFrame({"s1": gt.source1_entity_id.repeat(m.str.len()).values,
                              "m": [x.strip() for L in m for x in L]})
        pairs = pairs[pairs.m != ""].reset_index(drop=True)
        pairs.to_parquet(out, index=False)


def load_cached(args, name):
    return pd.read_parquet(cache_path(args, name))


def fold_of(ids, n=5):
    return (pd.util.hash_pandas_object(ids, index=False).values % n).astype(np.int8)


def build_universes(args, fracs, main_frac, aux_frac, dict_pairs_n=600_000):
    step("loading cached train sources")
    s1 = load_cached(args, "train_s1")
    pairs = load_cached(args, "train_pairs")
    pool = pd.concat([load_cached(args, "train_s2").assign(src="S2"),
                      load_cached(args, "train_s3").assign(src="S3")], ignore_index=True)
    pool = pool.merge(pairs.rename(columns={"m": "entity_id", "s1": "owner"}), on="entity_id", how="left")
    pool["owner"] = pool.owner.fillna("")
    rng = np.random.default_rng(args.seed)
    s1["prio"] = rng.random(len(s1))
    s1["fold"] = fold_of(s1.entity_id)
    s1 = s1.merge(pairs.groupby("s1").size().rename("k"), left_on="entity_id", right_index=True, how="left")
    s1["k"] = s1.k.fillna(0).astype(np.int16)
    prio = pd.Series(s1.prio.values, index=s1.entity_id.values)
    fold = pd.Series(s1.fold.values, index=s1.entity_id.values)
    pool["prio"] = pool.owner.map(prio)
    pool["fold"] = pool.owner.map(fold)
    d = pool.owner == ""
    pool.loc[d, "prio"] = rng.random(int(d.sum()))
    pool.loc[d, "fold"] = -1
    pool["fold"] = pool.fold.astype(np.int8)

    def save(tag, lo, hi):
        su = s1[(s1.prio >= lo) & (s1.prio < hi)].drop(columns="prio").reset_index(drop=True)
        pu = pool[(pool.prio >= lo) & (pool.prio < hi)].drop(columns="prio").reset_index(drop=True)
        out = udir(args, tag)
        os.makedirs(out, exist_ok=True)
        su.to_parquet(os.path.join(out, "s1.parquet"), index=False)
        pu.to_parquet(os.path.join(out, "pool.parquet"), index=False)
        step(f"universe {tag}: S1={len(su):,} pool={len(pu):,} (distractors {(pu.owner == '').mean():.1%}) "
             f"countries={su.country.value_counts().to_dict()}")

    for f in fracs:
        save(f"F{f:g}", 0.0, f)
    save("U1", 0.0, main_frac)
    save("U2", 0.5, 0.5 + aux_frac)

    step("building dictionary-learning pairs (S1 priority >= 0.6, disjoint from U1/U2)")
    keep_s1 = s1[s1.prio >= 0.6][["entity_id", "name", "addr", "country"]]
    dp = pairs[pairs.s1.isin(keep_s1.entity_id)]
    dp = dp.merge(keep_s1.rename(columns={"entity_id": "s1", "name": "q_name", "addr": "q_addr"}), on="s1")
    dp = dp.merge(pool[["entity_id", "name", "addr"]].rename(columns={"entity_id": "m", "name": "c_name",
                                                                     "addr": "c_addr"}), on="m")
    nonascii = (dp.c_name + dp.c_addr).map(lambda x: any(ord(ch) > 127 for ch in x))
    a = dp[nonascii]
    b = dp[~nonascii]
    na = min(len(a), int(dict_pairs_n * 0.8))
    nb = min(len(b), dict_pairs_n - na)
    dp = pd.concat([a.sample(na, random_state=args.seed), b.sample(nb, random_state=args.seed)], ignore_index=True)
    dp.to_parquet(os.path.join(args.work, "universes", "dict_pairs.parquet"), index=False)
    step(f"dictionary pairs: {len(dp):,} ({na:,} with non-ASCII match text)")


def load_universe(args, tag=None):
    d = udir(args, tag)
    s1 = pd.read_parquet(os.path.join(d, "s1.parquet"))
    pool = pd.read_parquet(os.path.join(d, "pool.parquet"))
    return s1, pool
