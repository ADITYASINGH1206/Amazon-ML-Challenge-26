"""Cross-encoder scoring for entity resolution (OPT-1).

Trains a cross-encoder on the training pairs and uses it as a feature in the LightGBM matcher.
Lab results show +0.0019 F0.5 from the CE feature on a 5% sample with a weak model;
a stronger backbone (mdeberta-v3-base) should push this to +0.002-0.004.

Architecture:
  1. TRAIN: fine-tune a multilingual cross-encoder on A-slice pairs
     Input: "name1 [SEP] addr1 [SEP] name2 [SEP] addr2" (normalised text)
  2. SCORE: predict match probability for all surviving candidate pairs (B for validation, test for submission)
  3. The CE score is added as a feature column to the LightGBM matcher (stage 1 or stage 2)

Models considered (MIT/Apache-2.0, <=8B):
  - cross-encoder/ms-marco-multilingual-MiniLM-L6-v2  (MIT, 107M) -- faster, decent
  - microsoft/mdeberta-v3-base  (MIT, 278M) -- stronger multilingual understanding
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .util import log

# ---- configuration -----------------------------------------------------------------------

CE_MODEL = "microsoft/mdeberta-v3-base"   # MIT, 278M params
CE_MAXLEN = 128          # max tokens per pair (name+addr for both records)
CE_BATCH_TRAIN = 64
CE_BATCH_SCORE = 256
CE_EPOCHS = 2
CE_LR = 2e-5
CE_WARMUP_RATIO = 0.1
CE_MAX_PAIRS = 400_000   # max training pairs (balanced)


# ---- dataset -----------------------------------------------------------------------------

class PairDataset(Dataset):
    """Pairs of (text_a, text_b, label) for cross-encoder training/scoring."""
    def __init__(self, texts_a, texts_b, labels=None):
        self.a = texts_a
        self.b = texts_b
        self.labels = labels

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        item = {"text_a": self.a[idx], "text_b": self.b[idx]}
        if self.labels is not None:
            item["label"] = float(self.labels[idx])
        return item


def _pair_text(nname_q, naddr_q, nname_p, naddr_p):
    """Format a pair as 'name1 | addr1 [SEP] name2 | addr2'."""
    a = f"{nname_q} | {naddr_q}" if naddr_q else nname_q
    b = f"{nname_p} | {naddr_p}" if naddr_p else nname_p
    return a, b


# ---- training ----------------------------------------------------------------------------

def train_cross_encoder(nname_s1, naddr_s1, nname_pool, naddr_pool,
                        qi, pj, labels, output_dir, n_pairs=CE_MAX_PAIRS):
    """Fine-tune a cross-encoder on balanced training pairs.

    Args:
        nname_s1, naddr_s1: normalised name/address arrays for S1 (indexed by qi)
        nname_pool, naddr_pool: normalised name/address arrays for pool (indexed by pj)
        qi, pj: candidate pair indices
        labels: binary labels (1=match)
        output_dir: where to save the fine-tuned model
        n_pairs: max training pairs (balanced positive/negative)
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    log(f"CE training: {len(qi):,} pairs, {labels.sum():,} positives; cap at {n_pairs:,} balanced pairs")

    # Balance positives and negatives
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = min(len(pos_idx), n_pairs // 2)
    n_neg = min(len(neg_idx), n_pairs // 2)
    rng = np.random.default_rng(C.SEED)
    sel_pos = rng.choice(pos_idx, n_pos, replace=False)
    sel_neg = rng.choice(neg_idx, n_neg, replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)

    # Build text pairs
    texts_a, texts_b, labs = [], [], []
    for idx in sel:
        q, p = qi[idx], pj[idx]
        a, b = _pair_text(nname_s1[q], naddr_s1[q], nname_pool[p], naddr_pool[p])
        texts_a.append(a)
        texts_b.append(b)
        labs.append(labels[idx])
    labs = np.array(labs, dtype=np.float32)
    log(f"  selected {len(sel):,} pairs: {labs.sum():.0f} pos, {(1-labs).sum():.0f} neg")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL, num_labels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 8
    batch_train = 128 if vram_gb >= 24 else CE_BATCH_TRAIN
    log(f"  using batch_train={batch_train} with AMP mixed precision")

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=CE_LR, eps=1e-6, weight_decay=0.01)
    n_steps = CE_EPOCHS * ((len(sel) + batch_train - 1) // batch_train)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(n_steps * CE_WARMUP_RATIO), n_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    use_amp = torch.cuda.is_available()
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_amp else torch.float32)
    use_scaler = use_amp and not use_bf16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    log(f"  using batch_train={batch_train} with AMP ({'bfloat16' if use_bf16 else 'float16'})")

    model.train()
    t0 = time.time()
    for epoch in range(CE_EPOCHS):
        perm = rng.permutation(len(sel))
        losses = []
        for i in range(0, len(perm), batch_train):
            batch_idx = perm[i:i + batch_train]
            batch_a = [texts_a[j] for j in batch_idx]
            batch_b = [texts_b[j] for j in batch_idx]
            batch_y = torch.tensor(labs[batch_idx], dtype=torch.float32, device=device)

            enc = tokenizer(batch_a, batch_b, padding=True, truncation=True,
                           max_length=CE_MAXLEN, return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(**enc).logits.squeeze(-1)

            # Compute loss in float32 for numerical stability (avoids DeBERTa-v3 NaN in lower precision)
            loss = loss_fn(logits.float(), batch_y)
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            if (i // batch_train) % 200 == 0 and i > 0:
                log(f"  epoch {epoch+1} step {i//batch_train}: loss {np.mean(losses[-200:]):.4f}")

        log(f"  epoch {epoch+1} done: mean loss {np.mean(losses):.4f} ({time.time()-t0:.0f}s)")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log(f"  CE model saved to {output_dir}")

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()


# ---- scoring -----------------------------------------------------------------------------

def score_pairs(nname_q, naddr_q, nname_pool, naddr_pool,
                qi, pj, model_dir, batch_size=None):
    """Score candidate pairs with the trained cross-encoder.

    Returns: np.ndarray of float32 probabilities, shape (len(qi),)
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if batch_size is None:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 8
        batch_size = 512 if vram_gb >= 24 else CE_BATCH_SCORE

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Build text pairs
    texts_a, texts_b = [], []
    for q, p in zip(qi, pj):
        a, b = _pair_text(nname_q[q], naddr_q[q], nname_pool[p], naddr_pool[p])
        texts_a.append(a)
        texts_b.append(b)

    # Sort by total length for efficient batching (less padding)
    lengths = np.array([len(a) + len(b) for a, b in zip(texts_a, texts_b)])
    order = np.argsort(lengths)
    scores = np.zeros(len(qi), dtype=np.float32)

    t0 = time.time()
    use_amp = torch.cuda.is_available()
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_amp else torch.float32)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        for i in range(0, len(order), batch_size):

            batch_idx = order[i:i + batch_size]
            batch_a = [texts_a[j] for j in batch_idx]
            batch_b = [texts_b[j] for j in batch_idx]

            enc = tokenizer(batch_a, batch_b, padding=True, truncation=True,
                           max_length=CE_MAXLEN, return_tensors="pt").to(device)
            logits = model(**enc).logits.squeeze(-1)
            scores[batch_idx] = torch.sigmoid(logits).cpu().numpy()

            if i % (batch_size * 50) == 0 and i > 0:
                rate = i / (time.time() - t0)
                eta = (len(order) - i) / rate
                log(f"  CE scoring {i:,}/{len(order):,} ({rate:.0f} pairs/s, ETA {eta/60:.0f}m)")

    elapsed = time.time() - t0
    log(f"  CE scoring done: {len(order):,} pairs in {elapsed:.0f}s ({len(order)/elapsed:.0f} pairs/s)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return scores
