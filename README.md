# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20%7C%20CUDA%2012.8-EE4C2C.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Optimized%20for-RTX%205090%20(32GB)-76B900.svg)](https://www.nvidia.com/)
[![Score](https://img.shields.io/badge/Public%20LB-0.9811%20F0.5-gold.svg)](#benchmark-progression)

End-to-end, GPU-accelerated solution for large-scale **Business Entity Resolution (ER)** across heterogeneous, noisy data sources ($S_1$, $S_2$, $S_3$) spanning the **United States, India, and France**. Evaluated under the precision-heavy **macro $F_{0.5}$ metric**.

---

## Benchmark Progression

Validation is measured on held-out Slice B (priority 0.15–0.30, folds 3–4) under full-scale, test-like pool density:

| Iteration | Key Innovations & Methodology | Held-Out Macro $F_{0.5}$ | Public Leaderboard |
|:---|:---|:---:|:---:|
| **v1** | Contrastive `multilingual-e5-small` blocking + LightGBM + stage-2 stacking | 0.9796 | 0.9744 |
| **v2** | Test pool density matching + V2 trigram IDF features + French canonicalization | 0.9826 | 0.9793 |
| **v4 (Current)** | Name uniqueness + fuzzy house-number alignment + sibling core features | **0.9852** | **0.9811** |

---

## Architectural Overview

The system processes **24+ million records** without quadratic comparison bottlenecks using a multi-stage funnel:

```
[Raw Sources: S1, S2, S3]
           │
           ▼
[Level-4 Normalization & Learned Transliteration Dictionary]
           │
           ▼
[Fine-Tuned multilingual-e5-small Embedding (FP16)]
           │
           ▼
[Exact GPU Nearest-Neighbor Blocking (Forward Top-10 + Reverse Top-3)]
   └─> Output: ~20 Candidate Pairs per S1 Entity
           │
           ▼
[Cheap LightGBM Reranker Filter (Zero Recall Loss)]
   └─> Output: ~4.0 to 6.0 Pairs per S1 ──> candidate_pairs.tsv
           │
           ▼
[Full LightGBM Matcher (50+ Trigram IDF, Core-Name & Numeric Signals)]
           │
           ▼
[Stage-2 Group-Aware Stacking (Query Competition & Cluster Context)]
           │
           ▼
[Calibrated Thresholding + Global 1-to-1 Assignment Policy]
   └─> Output: High-Precision Matches ──> matching_results.tsv
```

---

## Core Technical Highlights

### 1. Level-4 Normalization & Learned Transliteration
- **NFKC Unicode Normalization:** Unifies compound characters, removes ligatures, and strips non-printable artifacts.
- **Learned Indic-to-Latin Transliteration Dictionary:** ~45% of Indian records in Sources 2 & 3 appear in native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali). Rather than relying on generic phonetics, token mappings are automatically induced directly from high-confidence training pairs with fallback to `anyascii`.
- **Jurisdiction-Aware Canonicalization:** Standardizes commercial suffixes (`pvt ltd`, `inc`, `llc`, `sarl`) and street types across US, India, and France.

### 2. Contrastive Deep Embedding (`multilingual-e5-small`)
- Fine-tuned with `MultipleNegativesRankingLoss` over $(q_i, p_j^+, p_j^-)$ triplets from a strictly disjoint training slice.
- Mines hard intra-country negative records under the base model to pull true variants together while repelling distinct entities with identical namesakes.
- Encodes all text as `"query: {name} , {address}"` into 384-dimensional $L_2$-normalized vectors in FP16.

### 3. Exact GPU Nearest-Neighbor Blocking
- **Zero-Loss Exact kNN:** Instead of lossy ANN graphs, computes exact matrix inner products directly on GPU tensor cores within each country.
- **Forward Search ($K=10$):** Identifies top-10 candidate pool records for each Source 1 query.
- **Reverse Search ($M=3$):** Each pool item nominates its top-3 Source 1 queries, recovering true matches in crowded commercial hubs.
- **FP32 Exact Rescoring:** FP16 candidates are rescored in exact FP32 precision to break ties accurately.

### 4. Hierarchical Two-Stage Matching
- **Cheap Reranker Filter:** Small LightGBM model evaluating search context and token overlaps. Threshold $t_{\text{cheap}}$ is calibrated on held-out data to retain $>99.9\%$ of recall while pruning $\sim 70\%$ of negative pairs.
- **Full LightGBM Matcher:** 50+ features spanning:
  - *Context:* Cosine score, forward rank, reverse rank, top-1 gap, reverse margin, mutual top-1 flag.
  - *Text Similarity:* Character trigram Jaccard, token Jaccard, core-name overlap, consonant skeleton, Jaro-Winkler.
  - *Information Theory:* Trigram IDF cosine, unshared max IDF penalty, core token IDF coverage.
  - *Numeric & Address:* First house number equality, shared house numbers, PIN/postal code equality, minimum Levenshtein house-number distance.
- **Stage-2 Group-Aware Stacking:** Out-of-fold stacked LightGBM incorporating candidate probability share, runner-up margin, competing candidates count, and pool-side competition.
- **Global 1-to-1 Bipartite Matching:** Enforces that every pool record matches at most one Source 1 query, reflecting real-world business registry constraints.

---

## Hardware Acceleration (NVIDIA RTX 5090 / Blackwell)

The pipeline incorporates native Blackwell architecture optimizations:
- **Compute Capability `sm_120` Support:** Native PyTorch build with CUDA 12.8 (`cu128`).
- **Adaptive Matrix Block Sizing:** For GPUs with $\ge 24\text{GB}$ VRAM, scales `p_block` to $1,000,000$ and `q_chunk` to $4096$, reducing kernel launches by $16\times$ compared to 8GB setups.
- **FP16 Inference Batches:** Dynamically scales embedding batches to $2048$ sequences, fully saturating tensor cores.
- **Subprocess Memory Isolation:** Each stage runs in an isolated Python process to return CUDA caching memory to the OS.
- **Total Pipeline Execution Time:** **~35 to 45 minutes** end-to-end from scratch on an RTX 5090.

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ADITYASINGH1206/Amazon-ML-Challenge-26.git
cd Amazon-ML-Challenge-26

# Install PyTorch with CUDA 12.8 (required for RTX 5090 / Blackwell)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

Verify your GPU is active:
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| Capability:', torch.cuda.get_device_capability(0))"
```

### 2. Running the Pipeline

From the pipeline directory:
```bash
cd code/business_entity_resolution
python run_pipeline.py
```

### 3. Pipeline Stages & CLI Options

The pipeline automatically caches all intermediate artifacts in `work/` and resumes seamlessly if interrupted:

```bash
# Run everything end-to-end
python run_pipeline.py --stages all

# Run specific stages only
python run_pipeline.py --stages load,dict,norm
python run_pipeline.py --stages finetune,embed,block
python run_pipeline.py --stages feats,train
python run_pipeline.py --stages predict     # Predict test set with trained models (~5 min)

# Custom paths (if data is stored elsewhere)
python run_pipeline.py --data-dir /path/to/dataset --work /path/to/work --out /path/to/output
```

---

## Submission Artifacts

When the pipeline completes, it produces the two required submission files in `output/`:
1. **`candidate_pairs.tsv`**: The final candidate set produced after blocking and cheap filtering (~4.0 pairs per $S_1$ entity).
2. **`matching_results.tsv`**: The high-precision matched entities after Stage-2 stacking and 1-to-1 assignment.

The pipeline automatically validates both files against all competition constraints:
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

---

## Repository Structure

```
Amazon-ML-Challenge-26/
├── README.md                                  # Project overview and reproduction manual
├── ARCHITECTURE_AND_METHODOLOGY.md            # In-depth architectural & mathematical reference
├── PROGRESS_CONTEXT.md                        # Detailed experimental notes & benchmarks
├── eda.py                                     # Exploratory data analysis script
├── eda_findings_and_report.md                 # Detailed data findings report
│
├── code/
│   └── business_entity_resolution/            # Main runnable pipeline
│       ├── run_pipeline.py                    # Runner wrapper
│       ├── requirements.txt                   # Pinned dependency environment
│       ├── README.md                          # Pipeline execution guide
│       └── src/
│           ├── run_pipeline.py                # Pipeline orchestrator
│           └── ber/                           # Core library modules
│               ├── config.py                  # Global configurations & hyperparameters
│               ├── data.py                    # Data loading, schema conversion & indexing
│               ├── prep.py                    # L4 normalization & dictionary learning
│               ├── text.py                    # Text processing & tokenizers
│               ├── embed.py                   # Contrastive E5 fine-tuning & FP16 encoding
│               ├── knn.py                     # Exact GPU blocking & context feature generation
│               ├── feats.py                   # Pairwise feature extraction engine
│               ├── v3.py                      # Name uniqueness & fuzzy house number features
│               ├── model.py                   # LightGBM training, stacking & decision rules
│               └── util.py                    # Logging, timing & caching utilities
│
└── student_resource/
    ├── Documentation_template.md              # Completed official methodology template
    ├── LAB_RESULTS.md                         # Empirical findings across ablations
    ├── GPU_RUN_UPTO06.md                      # Lab run logs
    ├── er_lab/                                # Modular experiment lab (00_prepare -> 99_report)
    └── utils/
        └── validate_submission.py             # Official submission compliance validator
```

---

## License & Attribution

All models (`intfloat/multilingual-e5-small`, `LightGBM`) and dependencies utilized comply with MIT and Apache 2.0 open-source licensing guidelines and are within the 8B parameter contest limit.
