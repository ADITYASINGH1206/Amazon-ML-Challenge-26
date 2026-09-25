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
    lgbm_model,
    max_pairs: int = None,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Prepare training data for the cross-encoder.

    Strategy:
    - Positive pairs: true matches from ground truth
    - Hard negatives: high LightGBM score but label=0 (most confusing pairs)
    - Easy negatives: a random sample of low-score negatives

    Ratio: 1:1 positive to negative, with negatives biased toward hard cases.
    """
    if max_pairs is None:
        max_pairs = config.CE_MAX_TRAIN_PAIRS

    # Build lookup
    s1_lookup = {}
    for _, row in df_s1.iterrows():
        s1_lookup[row["entity_id"]] = row

    target_lookup = {}
    for _, row in df_targets.iterrows():
        target_lookup[row["entity_id"]] = row

    def make_text(row):
        name = _safe_str(row.get("name_clean", ""))
        addr = _safe_str(row.get("addr_clean", ""))
        return f"{name}, {addr}" if addr else name

    # Collect positive and negative pairs
    positives = []
    hard_negatives = []
    easy_negatives = []

    # Use the labeled features DataFrame
    from src.features import get_feature_columns
    feature_cols = get_feature_columns()

    if "label" not in df_val_features.columns:
        # Label if not already done
        labels = []
        for _, row in df_val_features.iterrows():
            s1_id = row["s1_id"]
            s2s3_id = row["s2s3_id"]
            true_matches = ground_truth.get(s1_id, set())
            labels.append(1 if s2s3_id in true_matches else 0)
        df_val_features = df_val_features.copy()
        df_val_features["label"] = labels

    # Score with LightGBM to find hard negatives
    X = df_val_features[feature_cols].values
    probs = lgbm_model.predict_proba(X)[:, 1]
    df_val_features = df_val_features.copy()
    df_val_features["lgbm_prob"] = probs

    for _, row in df_val_features.iterrows():
        s1_id = row["s1_id"]
        s2s3_id = row["s2s3_id"]
        label = int(row["label"])
        prob = row["lgbm_prob"]

        if s1_id not in s1_lookup or s2s3_id not in target_lookup:
            continue

        text_a = make_text(s1_lookup[s1_id])
        text_b = make_text(target_lookup[s2s3_id])

        if label == 1:
            positives.append((text_a, text_b))
        elif prob > 0.3:  # Hard negative: LightGBM thought it was a match
            hard_negatives.append((text_a, text_b))
        else:
            easy_negatives.append((text_a, text_b))

    log.info(f"  Positives: {len(positives):,}")
    log.info(f"  Hard negatives: {len(hard_negatives):,}")
    log.info(f"  Easy negatives: {len(easy_negatives):,}")

    # Balance: target 1:1 ratio
    max_per_class = max_pairs // 2

    # Sample positives
    if len(positives) > max_per_class:
        random.seed(config.RANDOM_SEED)
        positives = random.sample(positives, max_per_class)

    n_pos = len(positives)
    n_hard = min(len(hard_negatives), int(n_pos * 0.7))  # 70% hard
    n_easy = min(len(easy_negatives), n_pos - n_hard)      # 30% easy

    random.seed(config.RANDOM_SEED)
    sampled_hard = random.sample(hard_negatives, n_hard) if n_hard > 0 else []
    sampled_easy = random.sample(easy_negatives, n_easy) if n_easy > 0 else []

    # Combine
    text_pairs = positives + sampled_hard + sampled_easy
    labels = [1] * len(positives) + [0] * (len(sampled_hard) + len(sampled_easy))

    # Shuffle
    combined = list(zip(text_pairs, labels))
    random.shuffle(combined)
    text_pairs, labels = zip(*combined) if combined else ([], [])
    text_pairs = list(text_pairs)
    labels = list(labels)

    log.info(f"  Total training pairs: {len(text_pairs):,} "
             f"(pos: {sum(labels):,}, neg: {len(labels) - sum(labels):,})")

    return text_pairs, labels


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

@timed
def train_cross_encoder(text_pairs: List[Tuple[str, str]],
                        labels: List[int]) -> str:
    """
    Fine-tune cross-encoder and save checkpoint.

    Returns: path to saved model directory
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

    # Create dataset
    dataset = EntityPairDataset(
        text_pairs, labels, tokenizer, config.CROSS_ENCODER_MAX_LEN
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.CE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.CE_LEARNING_RATE,
        weight_decay=0.01,
    )

    # Scheduler
    total_steps = len(dataloader) * config.CE_EPOCHS
    warmup_steps = int(total_steps * config.CE_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if (config.CE_FP16 and device.type == "cuda") else None

    # Training loop
    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(config.CE_EPOCHS):
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.CE_EPOCHS}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(-1)
                    loss = loss_fn(logits, labels_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
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
                with torch.amp.autocast("cuda"):
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
