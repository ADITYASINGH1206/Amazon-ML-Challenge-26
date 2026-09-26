# Execution Guide — Business Entity Resolution Pipeline (Optimized)

## 1. System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 8 cores | 16+ cores |
| **RAM** | 16 GB | 32 GB |
| **GPU** | — | NVIDIA RTX 5090 (32 GB VRAM) |
| **Disk** | 20 GB free | 50 GB free |
| **Python** | 3.10 | 3.11 |
| **OS** | Windows 10/11, Linux | Windows 11, Ubuntu 22.04+ |

---

## 2. Environment Setup

### 2a. Create Virtual Environment

```bash
cd code/business_entity_resolution
python -m venv venv
```

### 2b. Activate (Windows)

```powershell
.\venv\Scripts\Activate.ps1
```

### 2b. Activate (Linux/Mac)

```bash
source venv/bin/activate
```

### 2c. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For GPU support (recommended):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2d. Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import lightgbm; print(f'LightGBM: {lightgbm.__version__}')"
python -c "from sparse_dot_topn import sp_matmul_topn; print('sparse_dot_topn OK')"
```

---

## 3. Dataset Placement

The pipeline expects datasets at the default path structure:

```
Amazon-ML-Challenge-26/
├── 6ab10eb3b23ba_student_resource/
│   └── student_resource/
│       └── dataset/
│           ├── train/
│           │   ├── train_source1.tsv
│           │   ├── train_source2.tsv
│           │   ├── train_source3.tsv
│           │   └── train_ground_truth.tsv
│           └── test/
│               ├── test_source1.tsv
│               ├── test_source2.tsv
│               └── test_source3.tsv
└── code/
    └── business_entity_resolution/  ← Run from here
```

To override the dataset location:

```bash
set STUDENT_RESOURCE_DIR=D:\path\to\student_resource
```

---

## 4. Model Downloads

Models are downloaded automatically on first run:

| Model | Size | Purpose |
|-------|------|---------|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | ~400 MB | Dense embeddings for blocking + cosine features |
| `microsoft/mdeberta-v3-base` | ~750 MB | Cross-encoder for re-ranking |

Models are cached in `~/.cache/huggingface/` and reused across runs.

---

## 5. Pipeline Commands

### 5a. Run Complete Pipeline (End-to-End)

```bash
python run_pipeline.py all
```

### 5b. Run Individual Stages

```bash
# Stage 0: Preprocess and normalize data
python run_pipeline.py preprocess

# Stage 1: Blocking / candidate generation
python run_pipeline.py block

# Stage 2: Feature extraction
python run_pipeline.py features

# Stage 3a: Train LightGBM classifier
python run_pipeline.py train_lgbm

# Stage 3b: Train cross-encoder (DeBERTa)
python run_pipeline.py train_ce

# Threshold optimization on validation split
python run_pipeline.py validate

# Stage 4: Full inference on test data
python run_pipeline.py infer
```

### 5c. Stage Dependencies

```
preprocess → block → features → train_lgbm → train_ce → validate → infer
```

---

## 6. Resource-Safe Configuration

### Batch Size Tuning (in `src/config.py`)

| Parameter | Default | OOM-Safe | Max Performance |
|-----------|---------|----------|-----------------|
| `EMBEDDING_BATCH_SIZE` | 512 | 256 | 4096 (GPU) |
| `FEATURE_BATCH_SIZE` | 50,000 | 25,000 | 150,000 |
| `INFERENCE_BATCH_SIZE` | 256 | 128 | 512 (GPU) |
| `CE_BATCH_SIZE` | 64 | 32 | 128 (GPU) |
| `FAISS_TOP_K` | 30 | 20 | 50 |

### If You Hit OOM

**CPU RAM OOM:**
- Reduce `MAX_CANDIDATES_PER_ENTITY` to 30
- Reduce `FEATURE_BATCH_SIZE` to 25,000

**GPU VRAM OOM:**
- Reduce `EMBEDDING_BATCH_SIZE` to 128
- Reduce `CE_BATCH_SIZE` to 16
- Set `CE_FP16 = False` to disable mixed precision

---

## 7. Expected Outputs

| File | Location | Purpose |
|------|----------|---------|
| `matching_results.tsv` | `student_resource/output/` | **Final matches (submit this)** |
| `candidate_pairs.tsv` | `student_resource/output/` | Blocking candidate set |

---

## 8. Re-running Individual Stages

```powershell
# Force re-run preprocessing (deletes ALL downstream)
Remove-Item -Recurse -Force workdir\preprocessed\*, workdir\blocking\*, workdir\features\*, workdir\models\*, output\* -ErrorAction SilentlyContinue

# Force re-run blocking only
Remove-Item -Recurse -Force workdir\blocking\*, workdir\features\*, output\*candidates* -ErrorAction SilentlyContinue

# Force re-run LightGBM training only
Remove-Item workdir\models\lgbm_model.pkl, workdir\models\lgbm_threshold.pkl -ErrorAction SilentlyContinue

# Force re-run cross-encoder training only
Remove-Item -Recurse -Force workdir\models\cross_encoder -ErrorAction SilentlyContinue
```

---

## 9. Key Optimizations Applied (SOLUTION.md)

| Priority | Fix | Impact |
|----------|-----|--------|
| **P1** | `scale_pos_weight = 1.0` (hardcoded) | Stops score compression |
| **P2** | City extraction + `city_exact_match` feature | Prevents cross-city false merges |
| **P3** | Threshold search max 0.999, step 0.002 | Finds optimal F0.5 threshold |
| **P4** | Split TF-IDF (name/addr), Soundex to Metaphone | Blocking recall 93% to 98%+ |
| **P5** | `[NAME] [ADDR] [CITY]` semantic tags | DeBERTa precision on edge cases |

---

## 10. Quick Start (Copy-Paste)

```powershell
cd code\business_entity_resolution
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py all
```
