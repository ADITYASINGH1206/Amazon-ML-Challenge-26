# Business Entity Resolution: end-to-end pipeline

This pipeline takes the three test sources and produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`. It trains everything it needs from the training data, and nothing else.

## Setup

- Python 3.12 and an NVIDIA GPU (tested on an RTX 4060 Laptop with 8 GB and 32 GB of system RAM).
- Install the CUDA build of torch first, then the rest:

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The first run downloads `intfloat/multilingual-e5-small` (MIT licence, 118M parameters) from Hugging Face.

## Run

The pipeline expects this layout:

```
<root>/dataset/{train,test}/...     # challenge data
<root>/utils/validate_submission.py # optional; run automatically at the end
<root>/code/business_entity_resolution/   # this folder
```

From this folder, run:

```
python src/run_pipeline.py
```

- **Defaults:** `--data-dir <root>/dataset`, `--work <root>/work` (caches and models) and `--out <root>/output`.
- **Resuming:** every stage saves its result under `--work` and is skipped when that result already exists, so an interrupted run carries on where it stopped.
- **Running some stages:** use `--stages`, for example `--stages predict` to re-predict with the trained models.
- **Adjusting for the machine:** use `--n-jobs` (CPU workers, default 24) and `--norm-workers` (default 16).

## What each stage does

| Stage | What it does |
|---|---|
| `load` | Reads the TSVs into parquet. For training data, attaches the ground truth and a fixed random priority and fold (seed 42) to each Source 1 record. |
| `dict` | Learns a native-script → Latin token dictionary (Devanagari, Tamil, Telugu, …) from training pairs whose Source 1 priority is ≥ 0.6. |
| `norm` | Normalizes every name and address in parallel (level 4, described below), and builds per-country IDF tables for names and addresses. |
| `finetune` | Fine-tunes multilingual-e5-small on (Source 1 text, true match, hard negative) triplets, using Source 1 records with priority 0.50–0.53 and MultipleNegativesRankingLoss for one epoch. |
| `embed` | Encodes every record as `"name , address"` with the fine-tuned encoder, in fp16 on the GPU. Resumable. |
| `block` | Candidate generation, described below. |
| `feats` | Pair features for the training candidates. |
| `train` | Trains the matchers and the second stage, sets the thresholds, and estimates full-scale F0.5 on held-out data. |
| `predict` | Test: pair features → cheap filter → full matcher → second stage → threshold + one-to-one → the two TSVs, then runs the validator. |

**Level-4 normalization** does the following:
- NFKC cleanup;
- URL, phone and `#id` removal;
- digit and letter splitting;
- Indic runs mapped through the learned dictionary, falling back to `anyascii`;
- canonical legal suffixes (private → pvt, limited → ltd, …) and street types (road → rd, …);
- US and Indian state names mapped to codes;
- French régions and départements mapped to one code.

## Candidate generation (blocking)

- **Scales without all-pairs comparison:** every record becomes one 384-dimensional vector, and candidates are exact nearest neighbours **within the same country**, computed on the GPU. Scores are computed in fp16, and the top candidates are then re-scored exactly in fp32.
- **Forward search:** each Source 1 record takes its **top 10** pool records.
- **Reverse search:** each Source 2 / Source 3 record nominates its **top 3** Source 1 records. This recovers matches that fall out of a crowded forward list, and it keeps recall stable as the pool grows.
- **Cheap re-ranker:** a small LightGBM model on the search context plus set-overlap features (address numbers, name tokens, consonant skeletons) scores the ~17 candidates per record. It keeps those with p ≥ `t_cheap`. The threshold is chosen on held-out training data as the largest one that loses at most 0.001 of pair recall.
- **What `candidate_pairs.tsv` contains:** exactly this filtered list, which is the set the final matcher runs on. It averages about 4 records per Source 1 record.

## Matching

1. **Full matcher.** LightGBM on:
   - search context: cosine, forward and reverse rank, gap to the best, reverse margin, mutual nearest neighbour, list size, source;
   - name features: token, trigram, core-name and IDF overlaps, Jaro-Winkler, fuzzy ratios, consonant skeletons;
   - address features: token, trigram and IDF overlaps, containment, fuzzy ratio;
   - number features: shared house numbers, conflicts, first-number and postcode equality.
2. **Second stage.** It adds group features computed from the full matcher's probabilities: rank and share of the Source 1 record's best candidate, how many strong candidates compete, competition for the same pool record, and similarity to the record's strongest other candidate. It is used only if it beats the full matcher on held-out data.
3. **Decision.** A probability threshold, then **one-to-one** assignment: each pool record goes to at most one Source 1 record, which holds for 100% of the training ground truth.

## Honest evaluation

Training Source 1 records are split by the fixed random priority:

| Slice | Priority | Used for |
|---|---|---|
| **A** | 0.00–0.15 | Fits the cheap and full matchers |
| **B** | 0.15–0.30 | Out-of-sample: chooses `t_cheap`, trains the second stage by 5-fold stacking, tunes the threshold on folds 0–2, and **reports F0.5 on folds 3–4** |
| Encoder fine-tuning | 0.50–0.53 | Fine-tunes the encoder |
| Dictionary | ≥ 0.60 | Learns the transliteration dictionary |

- The slices never overlap.
- Candidate search for A and B runs against the **full** training pool and all training Source 1 records, so the scores reflect full-scale density rather than a sample.
- Results are written to `<work>/train_report.json`.
- Country is never a model feature. It's only used to group the search, so France, which appears only in the test set, is handled the same way as the other countries.
