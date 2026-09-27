"""
compare_v4.py — Directly inspect work/models_v4 and compare with current output.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    work = os.path.join(ROOT, "work")
    v4_dir = os.path.join(work, "models_v4")
    v4_dec = os.path.join(v4_dir, "decision.json")
    v4_tsv = os.path.join(v4_dir, "matching_results.tsv")
    curr_tsv = os.path.join(ROOT, "output", "matching_results.tsv")

    print("=" * 70)
    print("V4 MODEL & BACKUP AUDIT")
    print("=" * 70)

    if os.path.exists(v4_dec):
        with open(v4_dec) as f:
            d = json.load(f)
        print(f"Found v4 decision.json: {json.dumps(d, indent=2)}")
    else:
        print(f"v4 decision.json not found at {v4_dec}")

    if os.path.exists(v4_tsv):
        print(f"\nAnalyzing v4 matching_results.tsv (the 0.98111 benchmark):")
        df_v4 = pd.read_csv(v4_tsv, sep="\t", dtype=str).fillna("")
        matches_v4 = df_v4.matched_entity_ids.str.split(",").apply(lambda x: len([i for i in x if i]))
        n_m4 = matches_v4.sum()
        empty_v4 = (matches_v4 == 0).sum()
        print(f"  Total matches in v4: {n_m4:,} ({n_m4 / len(df_v4):.3f} per S1)")
        print(f"  Empty S1 in v4: {empty_v4:,} ({empty_v4 / len(df_v4):.2%})")

        if os.path.exists(curr_tsv):
            print(f"\nComparing with current output/matching_results.tsv:")
            df_curr = pd.read_csv(curr_tsv, sep="\t", dtype=str).fillna("")
            matches_curr = df_curr.matched_entity_ids.str.split(",").apply(lambda x: len([i for i in x if i]))
            n_mc = matches_curr.sum()
            empty_curr = (matches_curr == 0).sum()
            print(f"  Total matches in current: {n_mc:,} ({n_mc / len(df_curr):.3f} per S1)")
            print(f"  Empty S1 in current: {empty_curr:,} ({empty_curr / len(df_curr):.2%})")

            # Check overlap
            v4_sets = df_v4.matched_entity_ids.apply(lambda x: set(x.split(",")) if x else set())
            curr_sets = df_curr.matched_entity_ids.apply(lambda x: set(x.split(",")) if x else set())
            
            exact_match_rows = (v4_sets == curr_sets).sum()
            print(f"  Entities with 100% identical predictions: {exact_match_rows:,} / {len(df_v4):,} ({exact_match_rows / len(df_v4):.1%})")

            # Where do they disagree?
            diff_s1 = np.where(v4_sets != curr_sets)[0]
            print(f"  Entities with differing predictions: {len(diff_s1):,}")
            if len(diff_s1) > 0:
                print("\nSample discrepancies (first 5 differing entities):")
                for idx in diff_s1[:5]:
                    sid = df_v4.source1_entity_id.iloc[idx]
                    s_v4 = df_v4.matched_entity_ids.iloc[idx]
                    s_curr = df_curr.matched_entity_ids.iloc[idx]
                    print(f"  S1: {sid}")
                    print(f"    v4:      {s_v4}")
                    print(f"    current: {s_curr}")
    else:
        print(f"v4 matching_results.tsv not found at {v4_tsv}")

    # Check work/models/decision.json
    curr_dec = os.path.join(work, "models", "decision.json")
    if os.path.exists(curr_dec):
        with open(curr_dec) as f:
            print(f"\nCurrent work/models/decision.json: {json.dumps(json.load(f), indent=2)}")

if __name__ == "__main__":
    main()
