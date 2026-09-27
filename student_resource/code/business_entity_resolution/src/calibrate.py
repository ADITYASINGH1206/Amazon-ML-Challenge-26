"""
calibrate.py — Instantly calibrate decision thresholds on precomputed stage-2 test scores.

Execution time: ~10-15 seconds.
Directly generates output/matching_results.tsv and verifies with validate_submission.py.
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
    parser = argparse.ArgumentParser(description="Instantly tune decision threshold and generate submission")
    parser.add_argument("--work", default=os.path.join(ROOT, "work"), help="path to work directory")
    parser.add_argument("--out", default=os.path.join(ROOT, "output"), help="path to output directory")
    parser.add_argument("--data-dir", default=find_data_dir(ROOT), help="path to raw dataset directory")
    parser.add_argument("--threshold", type=float, default=None,
                        help="flat threshold on probability p (e.g. 0.85)")
    parser.add_argument("--rule", default=None, choices=["threshold", "top1_plus"],
                        help="rule to apply: 'threshold' or 'top1_plus'")
    parser.add_argument("--t1", type=float, default=0.70, help="top-1 candidate threshold for top1_plus (default 0.70)")
    parser.add_argument("--t2", type=float, default=0.90, help="secondary candidate threshold for top1_plus (default 0.90)")
    parser.add_argument("--max-matches", type=int, default=4, help="maximum matches allowed per S1 entity (default 4)")
    args = parser.parse_args()

    scores_path = os.path.join(args.work, "feats", "test_scores.parquet")
    if not os.path.exists(scores_path):
        sys.exit(f"Error: {scores_path} not found. Run pipeline predict stage first.")

    t0 = time.time()
    log(f"loading test metadata and precomputed stage-2 scores from {scores_path}...")
    s1, pool = load_split(args.work, "test", columns=["entity_id", "country"])
    d = pd.read_parquet(scores_path)
    log(f"loaded {len(d):,} candidate pairs for {len(s1):,} S1 entities ({time.time() - t0:.1f}s)")

    log("enforcing 1-to-1 pool record assignment (0 violations in ground truth)...")
    o = one_to_one(d).sort_values(["qi", "p"], ascending=[True, False]).reset_index(drop=True)
    g = o.groupby("qi")
    rank = g.cumcount().values
    p_vals = o.p.values
    qi_vals = o.qi.values
    N_S1 = len(s1)

    print("\n" + "=" * 78)
    print(f"{'Rule / Threshold':>25} | {'Matches':>12} | {'Matches/S1':>12} | {'Empty S1':>10} | {'% Empty':>8}")
    print("-" * 78)

    # 1. Flat thresholds
    for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97]:
        sub_mask = p_vals >= t
        m_count = int(sub_mask.sum())
        counts = np.bincount(qi_vals[sub_mask], minlength=N_S1)
        n_empty = int((counts == 0).sum())
        note = ""
        if 5_720_000 <= m_count <= 5_780_000:
            note = " <-- OPTIMAL PRECISION TARGET (~5.75M)"
        print(f"{f'threshold({t:.2f})':>25} | {m_count:>12,} | {m_count / N_S1:>12.2f} | {n_empty:>10,} | {n_empty / N_S1:>7.1%}{note}")

    print("-" * 78)
    # 2. top1_plus thresholds (t1 for top-1 candidate, t2 for secondary candidates)
    for (t1, t2) in [(0.70, 0.85), (0.70, 0.90), (0.70, 0.92), (0.70, 0.95), (0.75, 0.90), (0.75, 0.95)]:
        sub_mask = ((rank == 0) & (p_vals >= t1)) | ((rank > 0) & (p_vals >= t2))
        m_count = int(sub_mask.sum())
        counts = np.bincount(qi_vals[sub_mask], minlength=N_S1)
        n_empty = int((counts == 0).sum())
        note = ""
        if 5_720_000 <= m_count <= 5_780_000:
            note = " <-- OPTIMAL PRECISION TARGET (~5.75M)"
        print(f"{f'top1_plus({t1:.2f}, {t2:.2f})':>25} | {m_count:>12,} | {m_count / N_S1:>12.2f} | {n_empty:>10,} | {n_empty / N_S1:>7.1%}{note}")
    print("=" * 78 + "\n")

    # Determine rule to apply
    if args.threshold is not None:
        log(f"applying user-selected flat threshold t = {args.threshold:.3f}...")
        kept_mask = p_vals >= args.threshold
    elif args.rule == "top1_plus":
        log(f"applying user-selected top1_plus(t1={args.t1:.2f}, t2={args.t2:.2f})...")
        kept_mask = ((rank == 0) & (p_vals >= args.t1)) | ((rank > 0) & (p_vals >= args.t2))
    else:
        # Default: auto-select the threshold or top1_plus that lands closest to the true 5.75M matches
        # Find best candidate among top1_plus and flat thresholds
        candidates = []
        for t in [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
            m = int((p_vals >= t).sum())
            candidates.append((abs(m - 5_750_000), ("threshold", t, None), p_vals >= t))
        for (t1, t2) in [(0.70, 0.85), (0.70, 0.90), (0.70, 0.92), (0.70, 0.95), (0.75, 0.90), (0.75, 0.95)]:
            mask = ((rank == 0) & (p_vals >= t1)) | ((rank > 0) & (p_vals >= t2))
            m = int(mask.sum())
            candidates.append((abs(m - 5_750_000), ("top1_plus", t1, t2), mask))
        candidates.sort(key=lambda x: x[0])
        best_diff, best_spec, kept_mask = candidates[0]
        if best_spec[0] == "threshold":
            log(f"auto-calibrated optimal rule: threshold({best_spec[1]:.2f}) (closest to ~5.75M matches)")
        else:
            log(f"auto-calibrated optimal rule: top1_plus({best_spec[1]:.2f}, {best_spec[2]:.2f}) (closest to ~5.75M matches)")

    # Optional cap to eliminate runaway clusters (> 4 matches per S1)
    if args.max_matches is not None and args.max_matches > 0:
        kept_mask = kept_mask & (rank < args.max_matches)

    kept = o[kept_mask].sort_values(["qi", "p"], ascending=[True, False])

    os.makedirs(args.out, exist_ok=True)
    out_tsv = os.path.join(args.out, "matching_results.tsv")
    ids1, idsp = s1.entity_id.values, pool.entity_id.values
    write_lists(out_tsv, ids1, kept.qi.values, idsp[kept.pj.values], "matched_entity_ids")

    n_m = np.bincount(kept.qi.values, minlength=N_S1)
    n_empty = int((n_m == 0).sum())
    by_cty = pd.Series(n_m > 0).groupby(s1.country.values).mean().round(3).to_dict()
    log(f"wrote {out_tsv}: {len(kept):,} matches ({len(kept) / N_S1:.2f} per S1); "
        f"empty S1: {n_empty:,} ({n_empty / N_S1:.1%}); "
        f"match rate by country: {by_cty}")

    val_candidates = [
        os.path.join(args.data_dir, "..", "utils", "validate_submission.py"),
        os.path.join(ROOT, "utils", "validate_submission.py"),
        os.path.join(ROOT, "student_resource", "utils", "validate_submission.py"),
        os.path.join(ROOT, "..", "student_resource", "utils", "validate_submission.py"),
        os.path.join(ROOT, "..", "utils", "validate_submission.py"),
    ]
    val = next((cand for cand in val_candidates if os.path.exists(cand)), None)
    if val:
        import subprocess
        log(f"running submission validator {val}...")
        r = subprocess.run([sys.executable, val, "--matching", out_tsv,
                            "--candidate", os.path.join(args.out, "candidate_pairs.tsv"),
                            "--test-dir", os.path.join(args.data_dir, "test")])
        log(f"validator exit code {r.returncode}")

    log(f"=== calibrate complete in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
