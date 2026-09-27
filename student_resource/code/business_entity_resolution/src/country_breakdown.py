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
    parser = argparse.ArgumentParser(description="Per-country breakdown and independent calibration")
    parser.add_argument("--apply", action="store_true", help="write calibrated output/matching_results.tsv")
    parser.add_argument("--france-t1", type=float, default=0.92, help="France top-1 threshold (default 0.92)")
    parser.add_argument("--france-t2", type=float, default=0.97, help="France secondary threshold (default 0.97)")
    parser.add_argument("--india-t1", type=float, default=0.75, help="India top-1 threshold (default 0.75)")
    parser.add_argument("--india-t2", type=float, default=0.95, help="India secondary threshold (default 0.95)")
    parser.add_argument("--us-t1", type=float, default=0.75, help="US top-1 threshold (default 0.75)")
    parser.add_argument("--us-t2", type=float, default=0.95, help="US secondary threshold (default 0.95)")
    args = parser.parse_args()

    work = os.path.join(ROOT, "work")
    if not os.path.exists(os.path.join(work, "feats", "test_scores.parquet")):
        work = os.path.abspath(os.path.join(ROOT, "..", "work"))
    scores_path = os.path.join(work, "feats", "test_scores.parquet")

    print("=" * 78)
    print("PER-COUNTRY BREAKDOWN & INDEPENDENT THRESHOLD TUNING")
    print("=" * 78)

    s1, pool = load_split(work, "test", columns=["entity_id", "country"])
    d = pd.read_parquet(scores_path)
    d["country"] = s1.country.values[d.qi.values]

    o = one_to_one(d).sort_values(["qi", "p"], ascending=[True, False]).reset_index(drop=True)
    g = o.groupby("qi")
    o["rank"] = g.cumcount().values

    # Scan each country
    for cty in ["France", "India", "US"]:
        sub_s1 = s1[s1.country == cty]
        sub_qi_set = set(sub_s1.index.values)
        sub_o = o[o.country == cty]
        n_s1 = len(sub_s1)
        print(f"\n--- {cty.upper()} ({n_s1:,} S1 entities, {len(sub_o):,} candidate pairs) ---")
        print(f"{'Rule / Threshold':>25} | {'Matches':>10} | {'Matches/S1':>10} | {'Empty S1':>8} | {'% Empty':>7}")
        print("-" * 78)
        
        # Grid based on country
        grid_t = [0.70, 0.80, 0.90, 0.95, 0.97, 0.98] if cty == "France" else [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        for t in grid_t:
            m_mask = sub_o.p.values >= t
            m_cnt = int(m_mask.sum())
            sub_qi = sub_o.qi.values[m_mask]
            counts = np.bincount(sub_qi, minlength=len(s1))[list(sub_qi_set)]
            empty_cnt = int((counts == 0).sum())
            note = ""
            if cty == "France" and 3.20 <= m_cnt / n_s1 <= 3.26:
                note = " <-- v4 BENCHMARK (3.25/S1, 5.65% empty)"
            print(f"{f'threshold({t:.2f})':>25} | {m_cnt:>10,} | {m_cnt / n_s1:>10.2f} | {empty_cnt:>8,} | {empty_cnt / n_s1:>6.1%}{note}")

        # top1_plus grid
        top_grid = [(0.80, 0.95), (0.85, 0.95), (0.90, 0.95), (0.90, 0.97), (0.92, 0.97), (0.94, 0.97)] if cty == "France" else [(0.70, 0.90), (0.70, 0.95), (0.75, 0.95), (0.80, 0.95)]
        for (t1, t2) in top_grid:
            m_mask = ((sub_o['rank'].values == 0) & (sub_o.p.values >= t1)) | ((sub_o['rank'].values > 0) & (sub_o.p.values >= t2))
            m_cnt = int(m_mask.sum())
            sub_qi = sub_o.qi.values[m_mask]
            counts = np.bincount(sub_qi, minlength=len(s1))[list(sub_qi_set)]
            empty_cnt = int((counts == 0).sum())
            note = ""
            if cty == "France" and 3.20 <= m_cnt / n_s1 <= 3.27 and empty_cnt / n_s1 >= 0.055:
                note = " <-- TARGET FRANCE RECOVERY (3.25/S1)"
            print(f"{f'top1_plus({t1:.2f}, {t2:.2f})':>25} | {m_cnt:>10,} | {m_cnt / n_s1:>10.2f} | {empty_cnt:>8,} | {empty_cnt / n_s1:>6.1%}{note}")

    if args.apply:
        print("\n" + "=" * 78)
        print("APPLYING PER-COUNTRY CALIBRATION & GENERATING SUBMISSION")
        print("=" * 78)
        cfg = {
            "France": (args.france_t1, args.france_t2),
            "India": (args.india_t1, args.india_t2),
            "US": (args.us_t1, args.us_t2),
        }
        for cty, (t1, t2) in cfg.items():
            print(f"  {cty}: top1_plus(t1={t1:.2f}, t2={t2:.2f})")

        kept_indices = []
        for cty, (t1, t2) in cfg.items():
            sub_o = o[o.country == cty]
            mask = ((sub_o['rank'].values == 0) & (sub_o.p.values >= t1)) | ((sub_o['rank'].values > 0) & (sub_o.p.values >= t2))
            kept_indices.extend(sub_o.index[mask].tolist())

        kept = o.loc[kept_indices].sort_values(["qi", "p"], ascending=[True, False])
        out_dir = os.path.abspath(os.path.join(work, "..", "output"))
        os.makedirs(out_dir, exist_ok=True)
        out_tsv = os.path.join(out_dir, "matching_results.tsv")
        ids1, idsp = s1.entity_id.values, pool.entity_id.values
        write_lists(out_tsv, ids1, kept.qi.values, idsp[kept.pj.values], "matched_entity_ids")

        n_m = np.bincount(kept.qi.values, minlength=len(s1))
        n_empty = int((n_m == 0).sum())
        by_cty = pd.Series(n_m > 0).groupby(s1.country.values).mean().round(3).to_dict()
        print(f"\nWrote {out_tsv}:")
        print(f"  Total matches: {len(kept):,} ({len(kept) / len(s1):.3f} per S1)")
        print(f"  Total empty S1: {n_empty:,} ({n_empty / len(s1):.2%})")
        print(f"  Match rate by country: {by_cty}")

        data_dir = find_data_dir(ROOT)
        test_dir = os.path.join(data_dir, "test") if data_dir else None
        val_candidates = [
            os.path.join(data_dir, "..", "utils", "validate_submission.py") if data_dir else "",
            os.path.join(ROOT, "utils", "validate_submission.py"),
            os.path.join(ROOT, "..", "utils", "validate_submission.py"),
            os.path.join(work, "..", "6ab10eb3b23ba_student_resource", "student_resource", "utils", "validate_submission.py"),
        ]
        val = next((cand for cand in val_candidates if cand and os.path.exists(cand)), None)
        if val and test_dir and os.path.exists(test_dir):
            import subprocess
            r = subprocess.run([sys.executable, val, "--matching", out_tsv,
                                "--candidate", os.path.join(out_dir, "candidate_pairs.tsv"),
                                "--test-dir", test_dir])
            print(f"Validator exit code: {r.returncode}")


if __name__ == "__main__":
    main()
