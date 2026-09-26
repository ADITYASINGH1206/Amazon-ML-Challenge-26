# Execution & Architecture Guide: Amazon ML Challenge 2026 Pipeline

This guide details the end-to-end execution, architecture, and memory-safety mechanisms of the Business Entity Resolution pipeline on high-performance machines (e.g., 64 GB RAM, NVIDIA RTX 5090 32 GB VRAM).

---

## 1. Prerequisites

Ensure the execution environment has:
- **Python 3.10+**
- **NVIDIA GPU Drivers & CUDA Toolkit** (CUDA 12.x+ recommended for RTX 5090)
- **Git** (to track and update branches)
- **64 GB System RAM** (the pipeline is engineered to stay strictly under ~16 GB at peak)

---

## 2. Setup the Environment

```bash
# Clone and enter directory
git clone <your-github-repo-url>
cd Amazon-ML-Challenge-26/code/business_entity_resolution

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Data Directory Configuration

By default, data is expected relative to the project root at:
`../../6ab10eb3b23ba_student_resource/student_resource/dataset/`

To specify a custom dataset location without modifying code, set `STUDENT_RESOURCE_DIR`:

```bash
# Windows (PowerShell):
$env:STUDENT_RESOURCE_DIR="C:\Path\To\student_resource"

# Linux/macOS (Bash):
export STUDENT_RESOURCE_DIR="/path/to/student_resource"
```
*(The specified directory must contain the `dataset/` folder with `train/` and `test/` TSV files).*

---

## 4. Pipeline Execution

The pipeline is fully orchestrated via `run_pipeline.py`.

### End-to-End Execution
```bash
python run_pipeline.py all
```

### Stage-by-Stage Execution
Because intermediate artifacts are persisted to disk in `workdir/`, any stage can be run or resumed independently:

| Stage Command | Description | Peak RAM | Expected Time |
| :--- | :--- | :--- | :--- |
| `python run_pipeline.py preprocess` | **Stage 0**: Country-agnostic text normalization | ~4 GB | 5-10 min |
| `python run_pipeline.py block` | **Stage 1**: Multi-strategy candidate generation (Inverted index, TF-IDF, PyTorch Dense ANN) | ~14 GB | 30-45 min |
| `python run_pipeline.py features` | **Stage 2**: 40-feature extraction across 70.5M candidate pairs (Streamed IPC) | ~8 GB | 60-90 min |
| `python run_pipeline.py train_lgbm` | **Stage 3a**: Cascade classifier training & hard-negative labeling (Zero-copy) | ~12 GB | 15-25 min |
| `python run_pipeline.py train_ce` | **Stage 3b**: DeBERTa-v3 Cross-Encoder fine-tuning (Mixed precision, grad accum) | ~6 GB (5GB VRAM) | 45-60 min |
| `python run_pipeline.py validate` | **Validation**: Vectorized margin threshold search on validation split | ~4 GB | 1-2 min |
| `python run_pipeline.py infer` | **Stage 4**: Test candidate generation, cascade pruning, cross-encoder re-ranking & submission formatting | ~10 GB | 40-60 min |

---

## 5. Memory-Safety & Anti-Crash Architecture

On datasets with 2.2M queries and 10.3M targets, unoptimized pipelines frequently encounter Out-Of-Memory (OOM) crashes and Segmentation Faults. This codebase incorporates zero-copy and disk-streaming patterns designed specifically to eliminate all RAM bottlenecks:

### A. Stage 2: Disk-Streamed Multiprocessing (`src/features.py`)
- **Direct Candidate Streaming**: Candidate pairs are streamed directly from generator buffers into 100,000-pair Parquet chunks on disk. Massive 70.5M Python string lists and DataFrame objects are never accumulated in RAM.
- **IPC SegFault & Overhead Elimination**: Workers receive lightweight chunk file paths instead of serialized DataFrame objects over Loky IPC channels, eliminating pickling bottlenecks and C-level IPC segfaults.
- **Worker RAM Bounds**: Worker processes are capped to `min(cpu_count, 16)` with 100k chunk sizes, constraining total worker memory to < 2.5 GB. Each worker writes its feature output directly to `_out.parquet` on disk.
- **Streaming ParquetWriter Merge**: Output chunks are merged into `train_features.parquet` via `pyarrow.parquet.ParquetWriter`. The full 70.5M row matrix is never held in memory simultaneously during concatenation.

### B. Stage 3a: Zero-Copy LightGBM Pipeline (`src/train_lgbm.py`)
- **Zero-Copy Float32 Slicing**: Feature columns are extracted as contiguous 2D float32 NumPy arrays before deleting the master DataFrame. Train and validation splits are indexed via boolean masks without duplicating memory.
- **Lightweight Probability Buffering**: LightGBM prediction on 50M+ rows is executed in 5,000,000-row batches. Only lightweight ID-only dataframes (`s1_id`, `s2s3_id`, `label`, `lgbm_prob`) are persisted for downstream re-ranking.
- **Vectorized Validation Search**: The validation threshold search avoids nested DataFrame filtering loops, evaluating all thresholds in seconds using hash-set lookups.

### C. Stage 3b: Vectorized Hard Negative Mining & VRAM Safety (`src/train_cross_encoder.py`)
- **Fast Boolean Negative Mining**: Hard negatives (`label == 0` and high `lgbm_prob`) are extracted using vectorized pandas filtering, bypassing slow iterative row loops.
- **VRAM Stability (RTX 5090)**: Training DeBERTa-v3-base uses micro-batch size 32 with gradient accumulation steps of 2 (effective batch size 64) and PyTorch FP16 AMP. VRAM usage remains stably under 6 GB of the 32 GB available.

### D. Stage 4: Cascaded Filtering & O(N) Conflict Resolution (`src/inference.py`)
- **LightGBM Fast Filter**: Reduces candidate pairs from ~50 down to top-5 per entity using vectorized `sort_values` + `groupby.head()`.
- **Pre-Filtering Cross-Encoder Inputs**: Candidates with `lgbm_prob < 0.05` are automatically pruned prior to cross-encoder evaluation, avoiding redundant GPU passes on hopeless negatives.
- **Linear-Time Global Deduplication**: The conflict resolution engine operates on pre-filtered matches in single-pass hash tables, eliminating nested DataFrame scans.

---

## 6. Final Outputs

Upon completion, evaluation files are written to `student_resource/output/`:
1. **`matching_results.tsv`**: TSV file containing `source1_entity_id` and comma-separated `matched_entity_ids` (empty string for singletons).
2. **`candidate_pairs.tsv`**: TSV file containing `source1_entity_id` and comma-separated `candidate_entity_ids`.
