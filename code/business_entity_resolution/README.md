# Business Entity Resolution — Amazon ML Challenge 2026

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python run_pipeline.py all
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 0: Preprocessing (preprocess.py)                     │
│  - Unicode normalization, legal suffix standardization      │
│  - Address abbreviation expansion, component extraction     │
│  - Country-agnostic: handles US, India, France              │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Multi-Strategy Blocking (blocking.py)             │
│  - Composite-key inverted index (city + name + postal)      │
│  - TF-IDF character n-gram sparse retrieval                 │
│  - FAISS ANN dense retrieval (multilingual embeddings)      │
│  - Union of all strategies → candidate_pairs.tsv            │
│  Target: >95% recall, <50 candidates per entity             │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: Feature Engineering (features.py)                 │
│  - 40 features per pair: string similarity, components,     │
│    IDF-weighted overlap, embedding cosine, meta features    │
├─────────────────────────────────────────────────────────────┤
│  Stage 3a: LightGBM Fast Filter (train_lgbm.py)            │
│  - Cascade: 50 candidates → top 5 per entity               │
│  - Scores millions of pairs in seconds                      │
├─────────────────────────────────────────────────────────────┤
│  Stage 3b: Cross-Encoder Re-ranking (train_cross_encoder.py)│
│  - mdeberta-v3-base fine-tuned on hard negatives            │
│  - Only runs on ~5M pruned pairs (not 85M)                  │
├─────────────────────────────────────────────────────────────┤
│  Stage 4: Ensemble + Thresholding (inference.py)            │
│  - Weighted average: 0.35 × LightGBM + 0.65 × CrossEncoder │
│  - Margin-based precision protection                        │
│  - Global dedup (each S2/S3 → at most one S1)               │
│  - Singleton safeguard (empty if max score < τ)             │
│  → matching_results.tsv                                     │
└─────────────────────────────────────────────────────────────┘
```

## Running Individual Stages

Each stage saves intermediate results, so you can resume from any point:

```bash
python run_pipeline.py preprocess    # Stage 0
python run_pipeline.py block         # Stage 1
python run_pipeline.py features      # Stage 2
python run_pipeline.py train_lgbm    # Stage 3a
python run_pipeline.py train_ce      # Stage 3b
python run_pipeline.py validate      # Threshold optimization
python run_pipeline.py infer         # Final inference
```

## Key Design Decisions

1. **Multilingual embeddings** (`paraphrase-multilingual-MiniLM-L12-v2`):
   France is 15% of test data but 0% of training. English-only embeddings
   would fail on French business names and addresses.

2. **Two-stage cascade** (LightGBM → Cross-Encoder):
   Running mdeberta-v3-base on 85M pairs would take 24-48 hours.
   LightGBM pre-filters to top-5 per entity (~8.5M pairs), reducing
   cross-encoder time to ~30 minutes on a single GPU.

3. **No top-1-per-source constraint**:
   Entities can have multiple matches per source (avg 1.77 from S2,
   1.89 from S3). All candidates above threshold τ are included.

4. **Margin-based thresholding**:
   When a S2/S3 entity is claimed by multiple S1 entities with similar
   scores, the assignment is rejected (ambiguous). Protects precision.

5. **F₀.₅ optimization**:
   Threshold τ is grid-searched on validation data to maximize the
   precision-heavy metric. Typical optimum: τ ∈ [0.72, 0.85].

## Output Files

| File | Purpose |
|------|---------|
| `output/matching_results.tsv` | Final entity matches (scored on leaderboard) |
| `output/candidate_pairs.tsv` | Blocking output (audited for quality) |

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥16GB VRAM (for cross-encoder)
- **RAM**: ≥32GB (for inverted index + TF-IDF on 10M records)
- **Storage**: ≥50GB (embeddings, features, model checkpoints)
- **Estimated runtime**: 4-6 hours end-to-end on RTX 5090

## Configuration

All hyperparameters are in [`src/config.py`](src/config.py):
- Paths, model names, blocking params, LightGBM params
- Cross-encoder training settings
- Threshold search range, margin delta
- Ensemble weights
