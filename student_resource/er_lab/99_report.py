#!/usr/bin/env python3
"""99 -- Collect every experiment result into one readable SUMMARY.md (send this file back)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lab.common import base_parser

SPEC = {
    "00_universe": (None, None),
    "00_dictionary": (None, None),
    "01_normalization": (["R@1", "R@5", "R@10", "R@20", "R@50", "R@10_India", "R@10_US", "native_R@1", "native_R@10",
                          "native_R@50", "vocab", "seconds", "norm_records_per_s"], "R@10"),
    "02_blocking": (["pair_recall", "ceiling_f05", "entity_full_recall", "cand_mean", "cand_p50", "cand_p90",
                     "cand_max", "no_cand_rate", "recall_India", "recall_US", "seconds"], None),
    "02_cands_written": (["n_pairs", "pair_recall", "ceiling_f05", "cand_mean", "cand_max"], None),
    "02_scaling": (["frac", "s1", "pool", "pair_recall", "ceiling_f05", "cand_mean", "fwd_seconds", "rev_seconds"],
                   None),
    "02_scaling_extrapolation": (None, None),
    "03_features": (None, None),
    "03_feature_auc": (["auc", "direction", "nan_rate", "auc_India", "auc_US"], "auc"),
    "04_matcher": (["n_features", "auc", "ap", "logloss", "f05_holdout", "hold_micro_precision", "hold_micro_recall",
                    "hold_f05_singletons", "hold_f05_India", "hold_f05_US", "params", "seconds"], "f05_holdout"),
    "04_candsize": (["pair_recall", "ceiling_f05", "entity_full_recall", "cand_mean", "cand_p90", "cand_max",
                     "no_cand_rate", "recall_India", "recall_US"], None),
    "04_transfer": (None, None),
    "04_importance": (["gain_share"], "gain_share"),
    "05_decision": (["params", "f05_tune", "f05_holdout", "hold_micro_precision", "hold_micro_recall",
                     "hold_avg_pred_per_s1", "hold_f05_singletons", "hold_singleton_false_merge_rate",
                     "hold_f05_India", "hold_f05_US"], "f05_holdout"),
    "05_stage2_model": (None, None),
    "05_best": (None, None),
    "06_embeddings": (["R@1", "R@5", "R@10", "R@50", "R@10_India", "R@10_US", "native_R@10", "native_R@50", "dim",
                       "texts_per_s"], "R@10"),
    "06_finetune_train": (None, None),
    "07_cross_encoder": (None, None),
    "08_country_probe": (None, None),
}
META = {"exp", "variant", "universe", "time"}


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e5 else f"{v:,.0f}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) if not isinstance(x, float) else f"{x:.3g}" for x in v) + "]"
    return str(v).replace("|", "/")


def table(rows, cols):
    out = ["| variant | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for r in rows:
        out.append(f"| {fmt(r['variant'])} | " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = os.path.join(args.work, "results", "results.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    latest = {}
    for r in rows:
        latest[(r["exp"], r["variant"], r.get("universe"))] = r
    by_exp = {}
    for r in latest.values():
        by_exp.setdefault(r["exp"], []).append(r)
    lines = ["# Experiment summary", "", f"results file: `{path}` ({len(rows)} rows, {len(latest)} unique)", ""]
    order = list(SPEC) + sorted(e for e in by_exp if e not in SPEC)
    for exp in order:
        if exp not in by_exp:
            continue
        rs = by_exp[exp]
        cols, primary = SPEC.get(exp, (None, None))
        if cols is None:
            seen = []
            for r in rs:
                for c in r:
                    if c not in META and c not in seen:
                        seen.append(c)
            cols = seen[:18]
        cols = [c for c in cols if any(c in r for r in rs)]
        if primary and all(isinstance(r.get(primary), (int, float)) for r in rs if primary in r):
            rs = sorted(rs, key=lambda r: -(r.get(primary) or -1e9))
        lines += [f"## {exp}", ""]
        if primary and rs and primary in rs[0]:
            lines += [f"**best by {primary}: {rs[0]['variant']} = {fmt(rs[0].get(primary))}**", ""]
        lines += [table(rs, cols), ""]
    text = "\n".join(lines)
    out = args.out or os.path.join(args.work, "SUMMARY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
