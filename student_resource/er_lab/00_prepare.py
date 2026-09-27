#!/usr/bin/env python3
"""00 -- cache the TSVs as parquet, build nested evaluation universes, learn the native->Latin dictionary.

Outputs (under --work):
  cache/*.parquet                 fast re-loading of all sources + exploded ground-truth pairs
  universes/F{f}/                 nested universes for the blocking scaling curve
  universes/U1/                   main evaluation universe (same as F{main-frac})
  universes/U2/                   disjoint auxiliary universe (training data for learned models)
  universes/dict_pairs.parquet    disjoint true pairs used to learn the translit dictionary
  models/translit_dict.json       learned dictionary
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from lab.common import base_parser, log_result, setup, step, udir
from lab.data import build_universes, cache_sources, load_universe
from lab.text import has_indic, indic_runs, learn_dict, save_dict


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--fracs", default="0.01,0.02,0.05,0.1", help="nested universe fractions (scaling curve)")
    ap.add_argument("--main-frac", type=float, default=0.05, help="fraction of S1 in U1 (<= 0.5)")
    ap.add_argument("--aux-frac", type=float, default=0.03, help="fraction of S1 in U2 (training aux)")
    ap.add_argument("--dict-pairs", type=int, default=600_000)
    args = ap.parse_args()
    if args.quick:
        args.fracs, args.main_frac, args.aux_frac, args.dict_pairs = "0.01,0.02", 0.02, 0.01, 200_000
    setup(args, "00_prepare")
    assert args.main_frac <= 0.5 and 0.5 + args.aux_frac <= 0.6

    cache_sources(args)
    fracs = [float(x) for x in args.fracs.split(",") if x]
    build_universes(args, fracs, args.main_frac, args.aux_frac, args.dict_pairs)

    step("learning native-script -> Latin dictionary from disjoint training pairs")
    dp = pd.read_parquet(os.path.join(args.work, "universes", "dict_pairs.parquet"))
    dct = learn_dict(dp.q_name.values, dp.c_name.values, dp.q_addr.values, dp.c_addr.values, log=print)
    save_dict(dct, os.path.join(args.work, "models", "translit_dict.json"))
    for field in ("name", "addr"):
        items = list(dct[field].items())[:40]
        print(f"  sample {field} mappings: " + ", ".join(f"{a}->{b}" for a, b in items))

    step("dictionary coverage on U1 native-script tokens")
    s1, pool = load_universe(args, "U1")
    cov = {}
    for field in ("name", "addr"):
        toks = [t for v in pool[field].values if has_indic(v) for t in indic_runs(v)]
        hit = sum(t in dct[field] or t in dct["addr" if field == "name" else "name"] for t in toks)
        cov[f"{field}_native_token_coverage"] = hit / max(len(toks), 1)
        cov[f"{field}_native_tokens"] = len(toks)
    log_result(args, "00_dictionary", "learned", name_entries=len(dct["name"]), addr_entries=len(dct["addr"]), **cov)

    u1 = {"s1": len(s1), "pool": len(pool), "singleton_rate": float((s1.k == 0).mean()),
          "distractor_rate": float((pool.owner == "").mean()), "pool_per_s1": len(pool) / max(len(s1), 1),
          "native_name_rate_pool": float(pool.name.map(has_indic).mean())}
    log_result(args, "00_universe", "U1", **u1)


if __name__ == "__main__":
    main()
