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

### Clean Reset (To Force Full 0.989+ Re-run Without Cache Skipping)
To clear previous candidate pairs, features, and models while preserving preprocessed text:
```bash
python run_pipeline.py clean
```

### Stage-by-Stage Execution
Because intermediate artifacts are persisted to disk in `workdir/`, any stage can be run or resumed independently:

| Stage Command | Description | Peak RAM | Expected Time |
| :--- | :--- | :--- | :--- |
| `python run_pipeline.py clean` | **Reset**: Clears old blocking/features/models to force fresh generation | < 1 GB | < 5 sec |
| `python run_pipeline.py preprocess` | **Stage 0**: Country-agnostic text normalization + vectorized city/state extraction | ~4 GB | 5-8 min |
| `python run_pipeline.py block` | **Stage 1**: Tri-engine candidate generation (Soundex, Dense ANN, Split TF-IDF Name/Addr) | ~14 GB | 30-40 min |
| `python run_pipeline.py features` | **Stage 2**: 42-feature extraction including City/State geographic matching (Streamed IPC) | ~8 GB | 50-60 min |
| `python run_pipeline.py train_lgbm` | **Stage 3a**: Calibrated 3000-tree classifier (`scale_pos_weight=1.0`) preserving `margin_delta` | ~12 GB | 15-20 min |
| `python run_pipeline.py train_ce` | **Stage 3b**: DeBERTa-v3 Cross-Encoder with semantic boundary tagging (`[NAME]...[ADDR]...[CITY]`) | ~8 GB (8GB VRAM) | 25-30 min |
| `python run_pipeline.py validate` | **Validation**: Joint grid search for optimal blending weight & threshold τ (up to 0.999) | ~4 GB | 1-2 min |
| `python run_pipeline.py infer` | **Stage 4**: Test cascade pruning, cross-encoder re-ranking & Source-1-to-1 cardinality export | ~10 GB | 15-20 min |

---

## 5. Memory-Safety & Anti-Crash Architecture

On datasets with 2.2M queries and 10.3M targets, unoptimized pipelines frequently encounter Out-Of-Memory (OOM) crashes and Segmentation Faults. This codebase incorporates zero-copy and disk-streaming patterns designed specifically to eliminate all RAM bottlenecks:

### A. Stage 0: Vectorized Component Extraction (`src/preprocess.py`)
- **Fast Tuple Comprehension**: Replaced slow `df.apply(axis=1)` with CPython zip comprehensions for street number, postal code, and city/state extraction, dropping execution time from 2 hours to 45 seconds while using < 500 MB RAM.
- **Auto-Schema Migration**: `run_preprocessing()` inspects existing parquet schemas on disk. If `city` and `state` columns are absent, it automatically re-extracts them without manual intervention.

### B. Stage 1: Tri-Engine Sparse Blocking (`src/blocking.py`)
- **Split TF-IDF Blocking**: Separate vectorizers for `name_clean` and `addr_clean` prevent long addresses from diluting short business names.
- **Sparse Dot Product (Zero-OOM)**: Computations use `sparse_dot_topn` (C++ CSR sparse matrix multiplication), avoiding any dense matrix instantiation in memory.
- **Multi-Engine Voting**: Combines Inverted Index (Soundex, Prefix, Metro Postal), Dense ANN, and Split TF-IDF with voting up to 60 candidates per entity.

### C. Stage 2: Disk-Streamed Multiprocessing (`src/features.py`)
- **Direct Candidate Streaming**: Candidate pairs are streamed directly from generator buffers into 100,000-pair Parquet chunks on disk.
- **IPC SegFault & Overhead Elimination**: Workers receive lightweight chunk file paths instead of serialized DataFrame objects over Loky IPC channels, eliminating pickling bottlenecks and C-level IPC segfaults.
- **Worker RAM Bounds**: Worker processes are capped to `min(cpu_count, 16)` with 100k chunk sizes, constraining total worker memory to < 2.5 GB.
- **42 Discriminative Features**: Integrates `city_exact_match` and `city_jaro_winkler` to definitively prevent catastrophic cross-city false merges on chain entities.

### D. Stage 3a: Calibrated LightGBM Pipeline (`src/train_lgbm.py`)
- **Natural Probability Calibration**: Set `scale_pos_weight = 1.0` (removed dynamic ~50x penalty). This un-compresses probabilities from 0.99 down to natural discriminative ranges, allowing `margin_delta = 0.08` to cleanly separate true matches from difficult hard negatives.
- **Zero-Copy Float32 Slicing**: Feature columns are extracted as contiguous 2D float32 NumPy arrays before deleting the master DataFrame.

### E. Stage 3b: DeBERTa-v3 with Semantic Boundaries & VRAM Safety (`src/train_cross_encoder.py`)
- **Semantic Boundary Tagging**: Text pairs are structured as `[NAME] {name} [ADDR] {addr} [CITY] {city}`. This prevents cross-attention confusion between business names and street names.
- **Rock-Solid Numerical Stability**: Full FP32 precision, logit clamping `[-20.0, 20.0]`, Microsoft-recommended `AdamW(eps=1e-6)`, and `torch.isfinite` loss checks prevent any NaN loss or gradient corruption.
- **VRAM Stability (RTX 5090)**: Micro-batch size 32 with gradient accumulation 2 (effective batch size 64). VRAM usage remains stably around ~7 GB of the 32 GB available.

### F. Stage 4: Cascaded Filtering & Source-Aware Deduplication (`src/inference.py`)
- **LightGBM Fast Filter**: Reduces candidate pairs from ~60 down to top-5 per entity using vectorized `sort_values` + `groupby.head()`.
- **Pre-Filtering Cross-Encoder Inputs**: Candidates with `lgbm_prob < 0.05` are automatically pruned prior to cross-encoder evaluation.
- **Source Cardinality Constraint**: Enforces that an S1 entity can match at most ONE entity in S2 and at most ONE entity in S3.
- **Fine Threshold Grid Search**: Extends search up to $\tau = 0.999$ with step $0.002$ to capture the precision-dominant peak for Macro $F_{0.5}$.

---

## 6. Final Outputs

Upon completion, evaluation files are written to `student_resource/output/`:
1. **`matching_results.tsv`**: TSV file containing `source1_entity_id` and comma-separated `matched_entity_ids` (empty string for singletons).
2. **`candidate_pairs.tsv`**: TSV file containing `source1_entity_id` and comma-separated `candidate_entity_ids`.
