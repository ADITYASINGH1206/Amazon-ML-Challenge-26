# ER experiment lab

A set of experiments that decide, with numbers, how to build each part of the pipeline:

| Part | Question | Script |
|---|---|---|
| Text | Which cleaning / transliteration makes matches easiest to find? | `01_normalization.py` |
| Blocking | How few candidates per S1 can we keep without losing matches? Does it scale? | `02_blocking.py` |
| Features | Which signals separate matches from look-alikes? | `03_features.py` |
| Matcher | Which feature groups matter? How small can `candidate_pairs.tsv` get? Does a US-trained model work on India (France proxy)? | `04_matcher.py` |
| Decision | Threshold vs one-to-one vs expected-F0.5 vs stacking — what maximises F0.5? | `05_decision.py` |
| Neural (GPU) | Off-the-shelf vs fine-tuned embeddings (joint vs separate name/address) | `06_embeddings.py` |
| Neural (GPU) | Cross-encoder: does reading both records together beat LightGBM? | `07_cross_encoder.py` |
| France | Does the pipeline behave on France like on US/India? (no labels, proxy signals) | `08_france_probe.py` |
| Report | Everything in one file | `99_report.py` → `er_work/SUMMARY.md` |

## Setup

Put the `er_lab` folder inside `student_resource/` (next to `dataset/`), then from `student_resource/`:

```
pip install -r er_lab/requirements.txt
```

`rapidfuzz` is required. It wasn't installed on the EDA machine, so the fuzzy-match features need it.
For the GPU steps, keep your existing CUDA build of torch and install only the rest of `requirements-gpu.txt`.

## Run

```
# 0) optional sanity check on a tiny synthetic dataset (~5 min, CPU)
python er_lab/run_all.py --toy

# 1) fast first pass on the real data (2% universe, fewer boosting rounds) -- about 1 hour
python er_lab/run_all.py --quick

# 2) full run incl. GPU experiments (5% universe) -- several hours, good for overnight
python er_lab/run_all.py --gpu
```

- A step that fails is reported, and the run continues.
- Resume with `--from 04`, or run selected steps with `--only 02,03`.

**Send back:** `er_work/SUMMARY.md` and the folder `er_work/logs/`. The `05_decision` log contains an error analysis; paste that too if it's long.

## How the evaluation is kept honest

- **Universe.** All experiments run on a *universe*: a random subset of S1 entities, together with **all** their true matches, plus the same fraction of unmatched distractor records. This keeps the real proportions: about 4.7 pool records per S1 and about 26% distractors.
  - Universes of 1%, 2%, 5% and 10% are nested.
  - `02 --scaling` measures how recall drops as density grows, and extrapolates to 100% size.
- **Learned components never see U1.** The translit dictionary, fine-tuned encoders and cross-encoder are trained on disjoint S1 entities (U2 / priority ≥ 0.6). Nothing leaks into U1 scores.
- **Matchers use 5-fold CV grouped by S1 entity.**
- **Decision thresholds** are tuned on folds 0–2 and reported on folds 3–4 (`f05_holdout`).
- **F0.5 is computed exactly as the challenge defines it**: macro-average per S1, where an empty prediction for a singleton scores 1.

## Main ideas being tested

- **Transliteration.** About 45% of India S2/S3 names are in Indic scripts. The tests use anyascii, plus a *learned* native→Latin word dictionary from training pairs, plus phonetic skeletons.
- **Reverse blocking.** Every pool record belongs to at most one S1 (0 violations in train). So each pool record searches for its nearest S1s, which gives about 4.7×m candidates per S1.
- **Cheap re-ranker → small `candidate_pairs.tsv`.** `04_candsize` shows recall versus average list size.
- **Your comma-component address matching** is the COMP feature group. Your separate name/address embeddings are views in 06.
- **Decision layer.**
  - One-to-one assignment.
  - Expected-F0.5-optimal lists.
  - Stacking with "sibling" agreement: does a candidate look like the S1's strongest candidate? This is the S2↔S3 agreement idea.

## Licences

The default neural models are:
- `intfloat/multilingual-e5-small` (MIT);
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0);
- `sentence-transformers/LaBSE` (Apache-2.0).

All are well under the 8B-parameter limit. Re-check each model card before the final submission.
