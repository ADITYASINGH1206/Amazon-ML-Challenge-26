"""Read the TSVs once into parquet caches, attach ground truth (train) and a fixed random priority / fold per S1."""
import os

import numpy as np
import pandas as pd

from .config import SEED
from .util import log

RENAME = {"business_name": "name", "business_address": "addr"}


def read_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def fold_of(ids, n=5):
    return (pd.util.hash_pandas_object(pd.Series(ids), index=False).values % n).astype(np.int8)


def cache_paths(work, split):
    d = os.path.join(work, "cache")
    return os.path.join(d, f"{split}_s1.parquet"), os.path.join(d, f"{split}_pool.parquet")


def build_split(data_dir, work, split):
    p1, pp = cache_paths(work, split)
    if os.path.exists(p1) and os.path.exists(pp):
        return
    os.makedirs(os.path.dirname(p1), exist_ok=True)
    src = os.path.join(data_dir, split)
    log(f"reading {split} TSVs from {src}")
    s1 = read_tsv(os.path.join(src, f"{split}_source1.tsv")).rename(columns=RENAME)
    pool = pd.concat([read_tsv(os.path.join(src, f"{split}_source2.tsv")).rename(columns=RENAME).assign(src="S2"),
                      read_tsv(os.path.join(src, f"{split}_source3.tsv")).rename(columns=RENAME).assign(src="S3")],
                     ignore_index=True)
    if split == "train":
        gt = read_tsv(os.path.join(src, "train_ground_truth.tsv"))
        m = gt.matched_entity_ids.str.split(",")
        pairs = pd.DataFrame({"s1": gt.source1_entity_id.repeat(m.str.len()).values,
                              "m": [x.strip() for L in m for x in L]})
        pairs = pairs[pairs.m != ""]
        owner = pd.Series(pairs.s1.values, index=pairs.m.values)
        pool["owner"] = pool.entity_id.map(owner).fillna("")
        # same random draw as the experiment lab -> identical S1 partition (no leakage from its trained parts)
        s1["prio"] = np.random.default_rng(SEED).random(len(s1))
        s1["fold"] = fold_of(s1.entity_id)
        s1["k"] = s1.entity_id.map(pairs.groupby("s1").size()).fillna(0).astype(np.int16)
        log(f"train: S1={len(s1):,} pool={len(pool):,} true pairs={len(pairs):,}")
    else:
        log(f"test: S1={len(s1):,} pool={len(pool):,} countries={s1.country.value_counts().to_dict()}")
    s1.to_parquet(p1, index=False)
    pool.to_parquet(pp, index=False)


def load_split(work, split, columns=None):
    p1, pp = cache_paths(work, split)
    return pd.read_parquet(p1, columns=columns), pd.read_parquet(pp, columns=columns)


def owner_index(s1, pool):
    """Pool row -> S1 row index of its true owner (-1 = distractor)."""
    m = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    return pool.owner.map(m).fillna(-1).astype(np.int64).values
