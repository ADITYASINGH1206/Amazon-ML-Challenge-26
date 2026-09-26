"""
train_cross_encoder.py — Cross-Encoder fine-tuning (Stage 3b).

Fine-tunes microsoft/mdeberta-v3-base as a cross-encoder for entity matching.
Uses hard negatives mined from LightGBM high-score false matches.

Input format: [CLS] name1, addr1 [SEP] name2, addr2 [EOS]
Output: binary classification (match / no-match)

Only runs on the pruned top-K candidates from LightGBM (cascade architecture).
"""

import gc
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Set, List, Tuple
from tqdm import tqdm

from src import config
from src.utils import log, timed, load_ground_truth, log_memory
from src.preprocess import load_preprocessed
from src.features import _safe_str


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class EntityPairDataset(Dataset):
    """Dataset for cross-encoder training: (text_a, text_b, label)."""

    def __init__(self, text_pairs: List[Tuple[str, str]], labels: List[int],
                 tokenizer, max_length: int):
        self.text_pairs = text_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_a, text_b = self.text_pairs[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text_a, text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────
# TRAINING DATA PREPARATION
# ─────────────────────────────────────────────────────────────

@timed
def prepare_cross_encoder_data(
    df_val_features: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    lgbm_model=None,
    max_pairs: int = None,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Prepare training data for the cross-encoder using ultra-fast vectorized filtering.

    Strategy:
    - Positive pairs: true matches from ground truth
    - Hard negatives: high LightGBM score but label=0 (most confusing pairs)
    - Easy negatives: a random sample of low-score negatives

    Ratio: 1:1 positive to negative, with negatives biased toward hard cases.
    """
    if max_pairs is None:
        max_pairs = config.CE_MAX_TRAIN_PAIRS

    # If lgbm_prob is missing, compute it; otherwise use the precomputed probabilities
    if "lgbm_prob" not in df_val_features.columns:
        from src.features import get_feature_columns
        feature_cols = get_feature_columns()
        if all(col in df_val_features.columns for col in feature_cols):
            X = df_val_features[feature_cols].to_numpy(dtype=np.float32, copy=False)
            df_val_features = df_val_features.copy()
            df_val_features["lgbm_prob"] = lgbm_model.predict_proba(X)[:, 1] if lgbm_model else 0.0
            del X
        else:
            df_val_features = df_val_features.copy()
            df_val_features["lgbm_prob"] = 0.0

    if "label" not in df_val_features.columns:
        true_matches_set = {(s1, cand) for s1, cands in ground_truth.items() for cand in cands}
        df_val_features = df_val_features.copy()
        df_val_features["label"] = [
            1 if (s1, s2) in true_matches_set else 0
            for s1, s2 in zip(df_val_features["s1_id"], df_val_features["s2s3_id"])
        ]
        del true_matches_set

    log.info("Mining hard negatives and balancing classes via vectorized filtering...")
    df_pos = df_val_features[df_val_features["label"] == 1]
    df_hard = df_val_features[(df_val_features["label"] == 0) & (df_val_features["lgbm_prob"] > 0.3)]
    df_easy = df_val_features[(df_val_features["label"] == 0) & (df_val_features["lgbm_prob"] <= 0.3)]

    log.info(f"  Available positives: {len(df_pos):,}")
    log.info(f"  Available hard negatives: {len(df_hard):,}")
    log.info(f"  Available easy negatives: {len(df_easy):,}")

    max_per_class = max_pairs // 2
    if len(df_pos) > max_per_class:
        df_pos = df_pos.sample(n=max_per_class, random_state=config.RANDOM_SEED)

    n_pos = len(df_pos)
    n_hard = min(len(df_hard), int(n_pos * 0.7))
    n_easy = min(len(df_easy), n_pos - n_hard)

    if len(df_hard) > n_hard:
        # Sort by highest LightGBM false match confidence to get the hardest negatives
        df_hard = df_hard.nlargest(n_hard, "lgbm_prob")
    if len(df_easy) > n_easy:
        df_easy = df_easy.sample(n=n_easy, random_state=config.RANDOM_SEED)

    df_selected = pd.concat([df_pos, df_hard, df_easy], ignore_index=True)
    df_selected = df_selected.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    del df_pos, df_hard, df_easy
    gc.collect()

    log.info("Building fast text lookup for sampled pairs...")
    s1_text = dict(zip(df_s1["entity_id"], df_s1["name_addr"].fillna("")))
    target_text = dict(zip(df_targets["entity_id"], df_targets["name_addr"].fillna("")))

    text_pairs = []
    labels = []
    for s1, s2, lbl in zip(df_selected["s1_id"], df_selected["s2s3_id"], df_selected["label"]):
        t1 = s1_text.get(s1, "")
        t2 = target_text.get(s2, "")
        if t1 and t2:
            text_pairs.append((t1, t2))
            labels.append(int(lbl))

    del df_selected, s1_text, target_text
    gc.collect()

    log.info(f"  Total training pairs: {len(text_pairs):,} (pos: {sum(labels):,}, neg: {len(labels) - sum(labels):,})")
    return text_pairs, labels


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

@timed
def train_cross_encoder(text_pairs: List[Tuple[str, str]],
                        labels: List[int]) -> str:
    """
    Fine-tune cross-encoder and save checkpoint.
    Uses batch size 32 with gradient accumulation for guaranteed VRAM safety on RTX 5090.
    """
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Training device: {device}")

    # Load tokenizer and model
    log.info(f"  Loading {config.CROSS_ENCODER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.CROSS_ENCODER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.CROSS_ENCODER_MODEL, num_labels=1
    )
    model.to(device)

    # Use micro-batch size 32 with grad accum 2 for rock-solid VRAM stability
    micro_batch_size = min(config.CE_BATCH_SIZE, 32)
    grad_accum_steps = max(1, config.CE_BATCH_SIZE // micro_batch_size)

    # Create dataset
    dataset = EntityPairDataset(
        text_pairs, labels, tokenizer, config.CROSS_ENCODER_MAX_LEN
    )
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.CE_LEARNING_RATE,
        weight_decay=0.01,
    )

    # Scheduler
    total_steps = (len(dataloader) // grad_accum_steps) * config.CE_EPOCHS
    warmup_steps = int(total_steps * config.CE_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # Mixed precision with BF16 (optimal for RTX 5090 / Ampere / Ada / Blackwell)
    use_bf16 = (config.CE_FP16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = (config.CE_FP16 and device.type == "cuda" and not use_bf16)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    # Training loop
    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(config.CE_EPOCHS):
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.CE_EPOCHS}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels_batch = batch["label"].to(device, non_blocking=True)

            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(-1)
                    loss = loss_fn(logits, labels_batch)
                    loss = loss / grad_accum_steps
                loss.backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            elif scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(-1)
                    loss = loss_fn(logits, labels_batch)
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits, labels_batch)
                loss = loss / grad_accum_steps
                loss.backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            total_loss += loss.item() * grad_accum_steps
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss/n_batches:.4f}"})

        avg_loss = total_loss / n_batches
        log.info(f"  Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

    # Save model
    save_dir = config.MODELS_DIR / "cross_encoder"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"  Cross-encoder saved to {save_dir}")

    # Clean up GPU memory
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(save_dir)


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

class CrossEncoderScorer:
    """Wrapper for cross-encoder inference."""

    def __init__(self, model_dir: str = None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if model_dir is None:
            model_dir = str(config.MODELS_DIR / "cross_encoder")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"  CrossEncoder device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, text_pairs: List[Tuple[str, str]],
                    batch_size: int = None) -> np.ndarray:
        """
        Score a list of (text_a, text_b) pairs.

        Returns: array of match probabilities (sigmoid of logits).
        """
        if batch_size is None:
            batch_size = config.INFERENCE_BATCH_SIZE

        all_probs = []

        for start in tqdm(range(0, len(text_pairs), batch_size),
                          desc="Cross-encoder scoring", mininterval=5):
            end = min(start + batch_size, len(text_pairs))
            batch = text_pairs[start:end]

            texts_a = [p[0] for p in batch]
            texts_b = [p[1] for p in batch]

            encodings = self.tokenizer(
                texts_a, texts_b,
                max_length=config.CROSS_ENCODER_MAX_LEN,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            if config.CE_FP16 and self.device.type == "cuda":
                ce_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast("cuda", dtype=ce_dtype):
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)

            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs)


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

@timed
def run_cross_encoder_training():
    """Full cross-encoder training pipeline."""
    # Check if model already exists
    ce_dir = config.MODELS_DIR / "cross_encoder"
    if ce_dir.exists():
        log.info(f"Cross-encoder model already exists at {ce_dir}. Skipping training.")
        return

    # Load labeled training features
    train_path = config.FEATURES_DIR / "train_features_labeled.parquet"
    if not train_path.exists():
        log.error("Run LightGBM training first to generate labeled features")
        return

    df_train = pd.read_parquet(train_path)
    log.info(f"Loaded {len(df_train):,} labeled training pairs")

    # Load preprocessed data for text generation
    df_s1 = load_preprocessed("train", "s1")
    df_s2 = load_preprocessed("train", "s2")
    df_s3 = load_preprocessed("train", "s3")
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    # Load ground truth and LightGBM model
    gt = load_ground_truth()
    from src.train_lgbm import load_lgbm_model
    lgbm_model = load_lgbm_model()

    # Prepare cross-encoder training data (hard negative mining)
    text_pairs, labels = prepare_cross_encoder_data(
        df_train, df_s1, df_targets, gt, lgbm_model
    )

    del df_train, df_s1, df_s2, df_s3, df_targets
    gc.collect()

    # Train
    model_dir = train_cross_encoder(text_pairs, labels)
    log.info(f"Cross-encoder training complete. Model at: {model_dir}")


if __name__ == "__main__":
    run_cross_encoder_training()
