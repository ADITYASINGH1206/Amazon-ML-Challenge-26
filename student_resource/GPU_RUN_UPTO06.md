# `--gpu` run: results through step 06 (snapshot at 26 Sep, ~18:00 IST)

This run used the 5% sample (U1): 109,955 Source 1 records and a pool of 514,073.
- 5.8% of Source 1 records have no match, and 26.2% of pool records match nothing.
- The raw files are saved in `er_work_gpu/snapshot_upto06/`: `results.jsonl`, the step logs and `gpu_run.log`.
- Step 06 was stopped partway through, so it could be restarted with the GPU fixes described in the last section.

## 00. Dictionary

- 1,347 name entries.
- They cover 97.7% of native-script name tokens.

## 01. Normalization (candidate recall on 5%; quick-run 2% figures in brackets)

| Normalization | R@10 | R@50 | R@10 India | Native-script R@10 |
|---|---|---|---|---|
| L0 raw | 0.959 (0.968) | 0.976 | 0.914 | 0.707 |
| L1 clean | 0.948 | 0.968 | 0.883 | 0.560 |
| L2 anyascii | 0.968 | 0.984 | 0.932 | 0.823 |
| L3 + learned dictionary | 0.978 | 0.990 | 0.957 | 0.985 |
| L4 canonical | 0.981 | 0.992 | 0.964 | 0.986 |
| **L4 canonical, word tokens** | **0.989** (0.993) | **0.997** | **0.980** | **0.989** |
| L4 + phonetic skeleton | 0.965 | 0.983 | 0.938 | 0.964 |
| L4, name only | 0.813 | 0.898 | 0.805 | 0.912 |
| L4, address only | 0.899 | 0.928 | 0.867 | 0.856 |

Conclusions from the quick run hold at 5%:
- L4 canonical with word tokens is best.
- The learned dictionary is the main gain on Indic-script names.
- Phonetic skeletons make things worse.
- Both the name and address fields are needed.

## 02. Blocking on 5% (quick-run 2% figures in brackets)

| Candidates | Recall | Mean candidates | Recall India |
|---|---|---|---|
| forward top-10 | 0.981 (0.987) | 10 | 0.964 |
| forward top-50 | 0.992 | 50 | 0.984 |
| reverse top-3 | 0.988 | 14.0 | 0.979 |
| **forward 10 + reverse 3** | **0.993** (0.995) | **18.4** | **0.986** |
| forward 50 + reverse 3 (written to disk) | 0.996 | 54.4 | 0.992 |
| name 10 + address 10 + forward 10 + reverse 3 | 0.996 | 28.8 | 0.993 |
| key blocking (union) | 0.902 | 87.6 | — |
| SVD-128 dense, forward 50 | 0.922 | 50 | — |

**Extrapolation to the full 10.3M pool**, now fitted on four sample sizes (1, 2, 5 and 10%):

| Candidates | Recall at full size | F0.5 ceiling | Recall lost per 10× pool growth |
|---|---|---|---|
| forward 10 + reverse 3 | **0.984** | 0.995 | −0.0065 |
| forward 50 | 0.981 | 0.994 | −0.0077 |
| forward 5 + reverse 3 | 0.980 | 0.994 | −0.0080 |
| reverse 3 | 0.974 | 0.991 | −0.0102 |
| forward 10 | 0.962 | 0.988 | −0.0144 |

- The two-point extrapolation from the quick run was slightly optimistic: 0.988 against 0.984 here.
- **Adding the name and address indexes** (name 10 + address 10 + forward 10 + reverse 3) gives the best recall on 5%, 0.996 with 28.8 candidates, and also helps India most. It wasn't in the scaling curve, so it's worth testing at full scale.

## 03. Features

- The candidate files hold 5.99M U1 pairs (6.3% of them true matches) and 3.59M U2 pairs.
- The features computed at about 42k pairs per second.
- AUC by feature:
  - blocking cosine: 0.990;
  - reverse margin: 0.986;
  - address containment: 0.955;
  - name Jaro-Winkler: 0.940;
  - comma-component address match: 0.914;
  - house-number Jaccard: 0.902.

## 06. Embeddings (partial, off-the-shelf models, 5% sample)

| Model / text view | R@1 | R@10 | R@50 | R@10 India |
|---|---|---|---|---|
| multilingual-e5-small, name + address | 0.270 | 0.970 | 0.984 | 0.948 |
| multilingual-e5-small, name only | 0.213 | 0.736 | 0.829 | 0.631 |
| multilingual-e5-small, address only | 0.262 | 0.902 | 0.932 | 0.879 |
| paraphrase-multilingual-MiniLM, name + address | 0.249 | 0.747 | 0.811 | 0.688 |

- Off-the-shelf embeddings trail TF-IDF on L4 word tokens: 0.970 against 0.989 at R@10.
- MiniLM is much weaker (0.747).
- Still to run: LaBSE, and the fine-tuned e5-small on all three views. Fine-tuning is the real test of whether embeddings add anything.

## Speed fixes made before restarting from step 06

- **GPU search** in `lab/knn.py` (`topk_dense`): exact search with `torch.topk` on the GPU, replacing numpy `argpartition` on the CPU.
  - It returns the same neighbours; the 0.013% that differ are exact ties.
  - Scores match to within 3e-7.
  - Before this, each text view spent about 5–10 minutes on this search while the GPU sat idle.
- **Half-precision encoding** in `06_embeddings.py`: encoding runs under fp16 autocast. It is 2.2× faster, and cosine to the full-precision embeddings is 1.0000.
- **Length-sorted scoring** in `07_cross_encoder.py`: pairs are scored in length-sorted batches, so batches carry far less padding. The scores are the same.
- **Log flushing** in `lab/common.py`: result lines are flushed to the logs as soon as they are written.
