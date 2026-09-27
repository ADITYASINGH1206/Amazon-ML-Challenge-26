"""Sentence encoder: contrastive fine-tuning of multilingual-e5-small and resumable fp16 encoding to .npy memmaps."""
import json
import os

import numpy as np
import pandas as pd

from .config import (EMB_BASE, EMB_MAXLEN, EMB_PREFIX, FT_BATCH, FT_LR, FT_PAIRS, FT_RANGE, SEED)
from .util import log


def joint_texts(names, addrs):
    """Raw 'name , address' text, exactly the view the encoder was fine-tuned on."""
    return [f"{EMB_PREFIX}{n} , {a}" if a else f"{EMB_PREFIX}{n}" for n, a in zip(names, addrs)]


def load_model(path):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(path, device="cuda" if torch.cuda.is_available() else "cpu")
    m.max_seq_length = EMB_MAXLEN
    if torch.cuda.is_available():
        m.half()
    return m


def encode_to_file(model, texts, out_path, chunk=200_000, batch=None):
    """Encode in chunks into a float16 .npy memmap; a crash resumes from the last finished chunk."""
    import torch
    if batch is None:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            batch = 2048 if vram_gb >= 24 else (1024 if vram_gb >= 14 else 512)
        else:
            batch = 256
    prog = out_path + ".progress.json"
    if os.path.exists(out_path) and not os.path.exists(prog):
        return
    n = len(texts)
    dim = model.get_sentence_embedding_dimension()
    tmp = out_path + ".partial.npy"
    done = 0
    if os.path.exists(prog) and os.path.exists(tmp):
        done = json.load(open(prog))["done"]
        mm = np.lib.format.open_memmap(tmp, mode="r+")
    else:
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, dim))
    for s in range(done, n, chunk):
        e = model.encode(texts[s:s + chunk], batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)
        mm[s:s + len(e)] = e.astype(np.float16)
        mm.flush()
        json.dump({"done": s + len(e)}, open(prog, "w"))
        log(f"  encoded {s + len(e):,}/{n:,} -> {os.path.basename(out_path)}")
    del mm
    os.replace(tmp, out_path)
    os.remove(prog)


def finetune(work, out_dir):
    """MultipleNegativesRankingLoss on (S1 text, true match, hard negative) triplets from a disjoint slice of
    training S1 entities (priority in FT_RANGE). Hard negative = the most similar non-match under the base encoder,
    searched inside the same country over that slice's records plus a same-sized share of distractors."""
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, \
        SentenceTransformerTrainingArguments
    try:
        from sentence_transformers.sentence_transformer import losses
    except ImportError:
        from sentence_transformers import losses
    try:
        from sentence_transformers.training_args import BatchSamplers
    except ImportError:
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    from .data import load_split
    from .knn import topk_gpu

    s1, pool = load_split(work, "train")
    lo, hi = FT_RANGE
    s1u = s1[(s1.prio >= lo) & (s1.prio < hi)].reset_index(drop=True)
    ids = set(s1u.entity_id)
    rng = np.random.default_rng(SEED + 1)
    dist = pool[(pool.owner == "") & (rng.random(len(pool)) < (hi - lo))]
    pu = pd.concat([pool[pool.owner.isin(ids)], dist], ignore_index=True)
    log(f"fine-tune slice: S1={len(s1u):,} pool={len(pu):,}")
    base = load_model(EMB_BASE)
    Eq = base.encode(joint_texts(s1u.name, s1u.addr), batch_size=512, normalize_embeddings=True,
                     convert_to_numpy=True).astype(np.float16)
    Ep = base.encode(joint_texts(pu.name, pu.addr), batch_size=512, normalize_embeddings=True,
                     convert_to_numpy=True).astype(np.float16)
    del base
    torch.cuda.empty_cache()
    own = pu.owner.values
    qid = s1u.entity_id.values
    hard = {}
    for c in s1u.country.unique():
        qi = np.where(s1u.country.values == c)[0]
        pi = np.where(pu.country.values == c)[0]
        idx, _ = topk_gpu(Eq[qi], Ep[pi], 10)
        for a, row in zip(qi, idx):
            for j in row:
                if j >= 0 and own[pi[j]] != qid[a]:
                    hard[a] = pi[j]
                    break
    pos = pu[pu.owner.isin(ids)][["owner", "name", "addr"]]
    qpos = pd.Series(np.arange(len(s1u)), index=qid)
    pos = pos.assign(qa=pos.owner.map(qpos).values)
    pos = pos[pos.qa.isin(list(hard))]
    if len(pos) > FT_PAIRS:
        pos = pos.sample(FT_PAIRS, random_state=SEED)
    qt = joint_texts(s1u.name, s1u.addr)
    pt = joint_texts(pu.name, pu.addr)
    ptxt = joint_texts(pos.name, pos.addr)
    ds = Dataset.from_dict({"anchor": [qt[a] for a in pos.qa.values], "positive": ptxt,
                            "negative": [pt[hard[a]] for a in pos.qa.values]})
    log(f"fine-tuning {EMB_BASE} on {len(ds):,} (anchor, positive, hard negative) triplets")
    model = SentenceTransformer(EMB_BASE, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = EMB_MAXLEN
    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(out_dir, "_tmp"), num_train_epochs=FT_EPOCHS, per_device_train_batch_size=FT_BATCH,
        learning_rate=FT_LR, warmup_ratio=0.1, fp16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no", logging_steps=200, report_to="none", seed=SEED)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                               loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(out_dir)
    log(f"fine-tuned encoder saved to {out_dir}")


def embed_split(work, split, model_dir):
    from .data import load_split
    d = os.path.join(work, "emb")
    os.makedirs(d, exist_ok=True)
    o1, op = os.path.join(d, f"{split}_s1.npy"), os.path.join(d, f"{split}_pool.npy")
    if os.path.exists(o1) and os.path.exists(op) and not os.path.exists(o1 + ".progress.json") \
            and not os.path.exists(op + ".progress.json"):
        return
    s1, pool = load_split(work, split, columns=["name", "addr"])
    model = load_model(model_dir)
    for df, out in ((s1, o1), (pool, op)):
        log(f"encoding {split} {os.path.basename(out)}: {len(df):,} records")
        encode_to_file(model, joint_texts(df.name.values, df.addr.values), out)
