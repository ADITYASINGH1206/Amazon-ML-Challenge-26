#!/usr/bin/env python3
"""07 -- Cross-encoder matcher (GPU strongly recommended).

A small multilingual transformer reads BOTH records together ("name | address" [SEP] "name | address") and
outputs P(same business). Trained on candidate pairs of the disjoint U2 universe (positives + hard negatives
from blocking), evaluated on U1 candidates. Its score is saved as an extra feature (group CE in 04), and it is
also scored standalone (one-to-one + threshold) so we see if it can replace or only complement LightGBM.
Throughput is reported to project the cost on the full test set.

Default backbone intfloat/multilingual-e5-small (MIT, 118M). Alternatives: xlm-roberta-base (MIT),
microsoft/mdeberta-v3-base (MIT). All far below the 8B limit -- verify licences before final use.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from lab.common import base_parser, log_result, setup, step, udir
from lab.data import load_universe
from lab.decide import one_to_one, thr, tune_and_eval
from lab.metrics import owner_index


def pair_texts(df):
    return [f"{n} | {a}" if a else n for n, a in zip(df.name.values, df.addr.values)]


def batches(n, bs):
    for s in range(0, n, bs):
        yield slice(s, min(s + bs, n))


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--train-pairs", type=int, default=400_000)
    ap.add_argument("--neg-per-pos", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--infer-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--infer-max-rank", type=int, default=20,
                    help="score U1 candidates with fwd_rank < this or any reverse rank (others get NaN)")
    args = ap.parse_args()
    setup(args, "07_cross_encoder")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    step(f"device={dev}, base={args.base}")

    s2, p2 = load_universe(args, "U2")
    c2 = pd.read_parquet(os.path.join(udir(args, "U2"), "cands.parquet"))
    pos = c2[c2.label == 1]
    neg = c2[(c2.label == 0) & ((c2.fwd_rank < 10) | c2.rev_rank.notna())]
    n_pos = min(len(pos), int(args.train_pairs / (1 + args.neg_per_pos)))
    pos = pos.sample(n_pos, random_state=args.seed)
    neg = neg.sample(min(len(neg), int(n_pos * args.neg_per_pos)), random_state=args.seed)
    tr = pd.concat([pos, neg]).sample(frac=1, random_state=args.seed)
    ta, tb = pair_texts(s2), pair_texts(p2)
    A = [ta[i] for i in tr.qi.values]
    B = [tb[j] for j in tr.pj.values]
    Y = tr.label.values.astype(np.float32)
    step(f"training pairs: {len(tr):,} ({n_pos:,} positive)")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(A) / args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    model.train()
    it = 0
    for ep in range(args.epochs):
        order = np.random.default_rng(args.seed + ep).permutation(len(A))
        for sl in batches(len(A), args.batch):
            ids = order[sl]
            enc = tok([A[i] for i in ids], [B[i] for i in ids], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            y = torch.tensor(Y[ids], device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logit = model(**enc).logits.squeeze(-1)
                loss = lossf(logit.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            if it % 200 == 0:
                step(f"  step {it}/{steps} loss={loss.item():.4f}")
    t_train = time.time() - t0
    out_dir = os.path.join(args.work, "models", "cross_encoder")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    s1, pool = load_universe(args)
    own = owner_index(s1, pool)
    c = pd.read_parquet(os.path.join(udir(args), "cands.parquet"))
    sel = ((c.fwd_rank < args.infer_max_rank) | c.rev_rank.notna()).values
    qa, pb = pair_texts(s1), pair_texts(pool)
    idx = np.where(sel)[0]
    # score in length-sorted batches: far less padding, identical scores
    plen = np.array([len(qa[q]) + len(pb[p]) for q, p in zip(c.qi.values[idx], c.pj.values[idx])])
    idx = idx[np.argsort(plen, kind="stable")]
    scores = np.full(len(c), np.nan, np.float32)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for sl in batches(len(idx), args.infer_batch):
            rows = idx[sl]
            enc = tok([qa[q] for q in c.qi.values[rows]], [pb[p] for p in c.pj.values[rows]], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logit = model(**enc).logits.squeeze(-1).float()
            scores[rows] = torch.sigmoid(logit).cpu().numpy()
    t_inf = time.time() - t0
    rate = len(idx) / max(t_inf, 1e-6)
    pd.DataFrame({"qi": c.qi.values, "pj": c.pj.values, "ce": scores}) \
        .to_parquet(os.path.join(udir(args), "extra_ce.parquet"), index=False)
    y = c.label.values[sel]
    d = pd.DataFrame({"qi": c.qi.values[sel], "pj": c.pj.values[sel], "p": scores[sel]})
    _, rep = tune_and_eval(one_to_one(d), thr, [(t,) for t in np.arange(0.1, 0.96, 0.05)], own, s1.k.values,
                           s1.fold.values, s1.country.values)
    log_result(args, "07_cross_encoder", args.base, train_seconds=t_train, infer_pairs=len(idx),
               infer_pairs_per_s=rate, auc=roc_auc_score(y, scores[sel]) if 0 < y.mean() < 1 else None,
               ap=average_precision_score(y, scores[sel]) if 0 < y.mean() < 1 else None,
               projected_hours_for_20M_test_pairs=20e6 / rate / 3600, **rep)


if __name__ == "__main__":
    main()
