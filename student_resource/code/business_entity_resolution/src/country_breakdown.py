"""
country_breakdown.py — Deep-dive per-country metrics on test predictions.
Reveals the exact France vs India vs US metrics and tunes per-country thresholds.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ber.data import load_split
from ber.model import one_to_one
from ber.util import log
from run_pipeline import find_data_dir, write_lists, ROOT


def main():
    work = os.path.join(ROOT, "work")
    if not os.path.exists(os.path.join(work, "feats", "test_scores.parquet")):
        work = os.path.abspath(os.path.join(ROOT, "..", "work"))
    scores_path = os.path.join(work, "feats", "test_scores.parquet")

    print("=" * 75)
    print("PER-COUNTRY BREAKDOWN & INDEPENDENT THRESHOLD TUNING")
    print("=" * 75)

    s1, pool = load_split(work, "test", columns=["entity_id", "country"])
    d = pd.read_parquet(scores_path)
    d["country"] = s1.country.values[d.qi.values]

    o = one_to_one(d).sort_values(["qi", "p"], ascending=[True, False]).reset_index(drop=True)
    g = o.groupby("qi")
    o["rank"] = g.cumcount().values

    for cty in ["France", "India", "US"]:
        sub_s1 = s1[s1.country == cty]
        sub_qi_set = set(sub_s1.index.values)
        sub_o = o[o.country == cty]
        n_s1 = len(sub_s1)
        print(f"\n--- {cty.upper()} ({n_s1:,} S1 entities, {len(sub_o):,} candidate pairs) ---")
        print(f"{'Rule / Threshold':>25} | {'Matches':>10} | {'Matches/S1':>10} | {'Empty S1':>8} | {'% Empty':>7}")
        print("-" * 75)
        for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            m_mask = sub_o.p.values >= t
            m_cnt = int(m_mask.sum())
            sub_qi = sub_o.qi.values[m_mask]
            counts = np.bincount(sub_qi, minlength=len(s1))[list(sub_qi_set)]
            empty_cnt = int((counts == 0).sum())
            note = ""
            if cty == "France" and 3.20 <= m_cnt / n_s1 <= 3.26:
                note = " <-- v4 BENCHMARK MATCH RATE (3.25/S1)"
            print(f"{f'threshold({t:.2f})':>25} | {m_cnt:>10,} | {m_cnt / n_s1:>10.2f} | {empty_cnt:>8,} | {empty_cnt / n_s1:>6.1%}{note}")

        # top1_plus
        for (t1, t2) in [(0.70, 0.90), (0.70, 0.95), (0.75, 0.95), (0.80, 0.95), (0.85, 0.95)]:
            m_mask = ((sub_o['rank'].values == 0) & (sub_o.p.values >= t1)) | ((sub_o['rank'].values > 0) & (sub_o.p.values >= t2))
            m_cnt = int(m_mask.sum())
            sub_qi = sub_o.qi.values[m_mask]
            counts = np.bincount(sub_qi, minlength=len(s1))[list(sub_qi_set)]
            empty_cnt = int((counts == 0).sum())
            note = ""
            if cty == "France" and 3.20 <= m_cnt / n_s1 <= 3.26:
                note = " <-- v4 BENCHMARK MATCH RATE (3.25/S1)"
            print(f"{f'top1_plus({t1:.2f}, {t2:.2f})':>25} | {m_cnt:>10,} | {m_cnt / n_s1:>10.2f} | {empty_cnt:>8,} | {empty_cnt / n_s1:>6.1%}{note}")


if __name__ == "__main__":
    main()
