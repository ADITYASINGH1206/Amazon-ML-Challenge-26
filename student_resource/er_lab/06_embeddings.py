#!/usr/bin/env python3
"""06 -- Neural embeddings (GPU strongly recommended).

A  Off-the-shelf multilingual sentence encoders (MIT / Apache-2.0 licensed -- verify before final use):
     intfloat/multilingual-e5-small (MIT), sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0),
     sentence-transformers/LaBSE (Apache-2.0)
   views: joint "name , address" | name only | address only  (your idea: separate name / address embeddings)
   -> retrieval recall@k (exact search per country, compare with TF-IDF in 01/02)
   -> cosine for every U1 candidate pair saved as extra features (04 picks them up as group EMB)
B  --finetune: contrastive fine-tuning (MultipleNegativesRankingLoss, in-batch + mined HARD negatives) on the
   disjoint U2 universe -- so there is no leakage into U1 evaluation. One model per view (joint / name / addr).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from lab.common import base_parser, log_result, setup, step, udir
from lab.data import load_universe
from lab.knn import knn_grouped, topk_dense
from lab.metrics import owner_index, recall_from_idx
from lab.text import has_indic

KS = (1, 5, 10, 20, 50)


def view_texts(df, view):
    if view == "joint":
        return [f"{n} , {a}" if a else n for n, a in zip(df.name.values, df.addr.values)]
    if view == "name":
        return list(df.name.values)
    return [a if a else "-" for a in df.addr.values]


def prefix_for(model_name):
    return "query: " if "e5" in model_name.lower() else ""


def encode(model, texts, prefix, batch):
    import torch
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        return model.encode([prefix + t for t in texts], batch_size=batch, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def evaluate_model(args, model, short, prefix, s1, pool, own, cands, views):
    native = pool.name.map(has_indic).values
    feats = {}
    for view in views:
        t0 = time.time()
        Es = encode(model, view_texts(s1, view), prefix, args.batch)
        Ep = encode(model, view_texts(pool, view), prefix, args.batch)
        t_enc = time.time() - t0
        idx, _ = knn_grouped(Es, Ep, s1.country.values, pool.country.values, max(KS), args.n_jobs,
                             fn=lambda a, b, kk, nj: topk_dense(a, b, kk, nj, hnsw=False))
        res = recall_from_idx(idx, own, s1.k.values, KS, s1.country.values, native)
        log_result(args, "06_embeddings", f"{short}|{view}", encode_seconds=t_enc,
                   texts_per_s=(len(s1) + len(pool)) / t_enc, dim=Es.shape[1], **res)
        feats[f"emb_{short}_{view}"] = (Es[cands.qi.values] * Ep[cands.pj.values]).sum(1).astype(np.float32)
    out = pd.DataFrame({"qi": cands.qi.values, "pj": cands.pj.values, **feats})
    out.to_parquet(os.path.join(udir(args), f"extra_emb_{short}.parquet"), index=False)


def training_triplets(args, view, n_max):
    s1, pool = load_universe(args, "U2")
    c = pd.read_parquet(os.path.join(udir(args, "U2"), "cands.parquet"))
    neg = c[c.label == 0].sort_values(["qi", "cos"], ascending=[True, False]).drop_duplicates("qi")
    hard = dict(zip(neg.qi.values, neg.pj.values))
    pos = c[c.label == 1]
    pos = pos[pos.qi.isin(hard)]
    if len(pos) > n_max:
        pos = pos.sample(n_max, random_state=args.seed)
    st = view_texts(s1, view)
    pt = view_texts(pool, view)
    return ([st[q] for q in pos.qi.values], [pt[p] for p in pos.pj.values], [pt[hard[q]] for q in pos.qi.values])


def finetune(args, base, view, out_dir):
    import torch
    from sentence_transformers import SentenceTransformer
    try:
        from sentence_transformers.sentence_transformer import losses
    except ImportError:
        from sentence_transformers import losses
    a, p, n = training_triplets(args, view, args.ft_pairs)
    pre = prefix_for(base)
    a, p, n = [pre + x for x in a], [pre + x for x in p], [pre + x for x in n]
    step(f"fine-tuning {base} on {len(a):,} (anchor, positive, hard negative) triplets, view={view}")
    model = SentenceTransformer(base, device=args.device)
    model.max_seq_length = args.max_len
    loss = losses.MultipleNegativesRankingLoss(model)
    cuda = torch.cuda.is_available()
    try:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        try:
            from sentence_transformers.training_args import BatchSamplers
        except ImportError:
            from sentence_transformers.sentence_transformer.training_args import BatchSamplers
        ds = Dataset.from_dict({"anchor": a, "positive": p, "negative": n})
        targs = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(out_dir, "_tmp"), num_train_epochs=args.ft_epochs,
            per_device_train_batch_size=args.ft_batch, learning_rate=args.ft_lr, warmup_ratio=0.1,
            fp16=cuda, batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no", logging_steps=200,
            report_to="none", seed=args.seed)
        SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
    except ImportError:
        from sentence_transformers import InputExample
        from sentence_transformers.datasets import NoDuplicatesDataLoader
        dl = NoDuplicatesDataLoader([InputExample(texts=[x, y, z]) for x, y, z in zip(a, p, n)],
                                    batch_size=args.ft_batch)
        model.fit(train_objectives=[(dl, loss)], epochs=args.ft_epochs, warmup_steps=int(0.1 * len(dl)),
                  optimizer_params={"lr": args.ft_lr}, use_amp=cuda)
    model.save(out_dir)
    return model


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--models", default="intfloat/multilingual-e5-small,"
                                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,"
                                        "sentence-transformers/LaBSE")
    ap.add_argument("--views", default="joint,name,addr")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-offshelf", action="store_true")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--ft-base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--ft-views", default="joint,name,addr")
    ap.add_argument("--ft-pairs", type=int, default=300_000)
    ap.add_argument("--ft-epochs", type=int, default=1)
    ap.add_argument("--ft-batch", type=int, default=128)
    ap.add_argument("--ft-lr", type=float, default=2e-5)
    args = ap.parse_args()
    setup(args, "06_embeddings")
    from sentence_transformers import SentenceTransformer
    s1, pool = load_universe(args)
    own = owner_index(s1, pool)
    cands = pd.read_parquet(os.path.join(udir(args), "cands.parquet"), columns=["qi", "pj"])
    views = args.views.split(",")
    if not args.skip_offshelf:
        for name in args.models.split(","):
            short = name.split("/")[-1].replace("-", "_").lower()[:24]
            step(f"off-the-shelf: {name}")
            model = SentenceTransformer(name, device=args.device)
            model.max_seq_length = args.max_len
            evaluate_model(args, model, short, prefix_for(name), s1, pool, own, cands, views)
            del model
    if args.finetune:
        for view in args.ft_views.split(","):
            out_dir = os.path.join(args.work, "models", f"ft_{view}")
            t0 = time.time()
            model = finetune(args, args.ft_base, view, out_dir)
            log_result(args, "06_finetune_train", f"ft_{view}", train_seconds=time.time() - t0)
            evaluate_model(args, model, f"ft_{view}", prefix_for(args.ft_base), s1, pool, own, cands, [view])
            del model


if __name__ == "__main__":
    main()
