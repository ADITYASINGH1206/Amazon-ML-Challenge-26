# ER lab results (26 Sep 2026)

This file covers two runs:
- **Part A, the `--gpu` run** on the 5% sample (`er_work_gpu/`). **Every step finished OK.** These are the headline results.
- **Part B, the earlier `--quick` run** on the 2% sample (`er_work/`), kept below for comparison.

# Part A: `--gpu` run, 5% sample (110k Source 1 records, 514k pool)

The full tables are in `er_work_gpu/SUMMARY.md`, and the per-step logs are in `er_work_gpu/logs/`. Results through step 06 are also written up in `GPU_RUN_UPTO06.md`.

## Final decisions

| Question | Answer | Evidence |
|---|---|---|
| Text cleaning | **L4 canonical forms + learned Indic→Latin dictionary** | R@10 0.989 with TF-IDF word tokens; native-script 0.989 (raw text: 0.707) |
| Candidate search | **Forward 10 + reverse 3**, plus the name and address indexes | 0.993 on 5%, about **0.984 extrapolated** to the full 10.3M pool; with name and address indexes: 0.996 |
| **Fine-tuned embeddings** | **Use them.** This is the biggest new finding. | Fine-tuned e5-small reaches **R@10 0.9965**, against 0.989 for TF-IDF. India 0.995; native-script 0.997. |
| Matcher features | **CTX + NAME + ADDR + NUM + embeddings + cross-encoder.** Drop COMP. | F0.5 **0.9917**; CE and EMB together add +0.0026 over the TF-IDF features alone |
| Cross-encoder | **Use as a feature only**, not on its own | Alone: 0.9864. As a feature: +0.0019 |
| Decision layer | **Second-stage model with group features, then a plain threshold of 0.7** | **F0.5 0.9934** (upper bound on these candidates: 0.9987) |
| Level-2 vs level-4 features | L4 | L2 is 0.0006–0.0020 worse in every configuration |

## 06: Embeddings (candidate search on their own)

| Model / text view | R@10 | R@50 | R@10 India | Native-script R@10 |
|---|---|---|---|---|
| TF-IDF, L4 word tokens (reference) | 0.989 | 0.997 | 0.980 | 0.989 |
| e5-small, off the shelf, name + address | 0.970 | 0.984 | 0.948 | 0.901 |
| LaBSE, off the shelf, name + address | 0.922 | 0.939 | 0.914 | 0.949 |
| MiniLM, off the shelf, name + address | 0.747 | 0.811 | 0.688 | 0.283 |
| **e5-small, fine-tuned, name + address** | **0.9965** | **0.9991** | **0.9953** | **0.9971** |
| e5-small, fine-tuned, name only | 0.888 | 0.962 | 0.880 | 0.718 |
| e5-small, fine-tuned, address only | 0.917 | 0.941 | 0.893 | 0.916 |

- Off-the-shelf models all trail TF-IDF.
- Off the shelf, e5 on names alone finds only 2.9% of Indic-script matches in its top 10.
- One epoch of contrastive fine-tuning (228k triplets, 12 min on the RTX 4060) makes e5-small **better than TF-IDF everywhere**.
- It was trained on U2, which shares no records with U1, so there's no leakage.

## 07: Cross-encoder

- Training took 14 minutes (400k pairs); scoring ran at 2.8k pairs per second. The full test set (about 20M pairs) would take about **2 hours**.
- On its own it reaches F0.5 0.9864 (AUC 0.9998). It's weak on records with no match, wrongly matching 4.6% of them.
- It only helps as a LightGBM feature.

## 04: Matcher (LightGBM, 5-fold, held-out F0.5)

| Feature groups | F0.5 | India | US |
|---|---|---|---|
| CTX | 0.9530 | 0.934 | 0.966 |
| + NAME | 0.9805 | 0.977 | 0.983 |
| + ADDR | 0.9876 | 0.986 | 0.988 |
| + NUM | 0.9890 | 0.988 | 0.990 |
| + COMP (all TF-IDF features) | 0.9891 | 0.989 | 0.990 |
| all + CE | 0.9910 | 0.991 | 0.991 |
| all + EMB | 0.9914 | 0.991 | 0.992 |
| **all + EMB + CE** | **0.9917** | **0.9917** | **0.9916** |

- **Features that matter most when removed one at a time:**
  - NAME: −0.0117
  - CTX: −0.0018
  - NUM: −0.0012
  - ADDR: ~0
  - COMP: ~0
- **Feature importance:** the candidate-search features carry most of the model.
  - `rev_rank`: 50% of the gain
  - `cos`: 28%
  - `rev_margin`: 9%
- **Country transfer (proxy for France):**
  - Training on US and applying to India loses 0.0089 F0.5; India → US loses 0.0057.
  - The source threshold is about as good as the best possible target threshold (0.9793 vs 0.9794).
- **Size of `candidate_pairs.tsv`:** the cheap re-ranker, keeping p ≥ 0.003, gives **4.2 candidates per record** with recall 0.9951. Keeping all 54 candidates gives 0.9956.

## 05: Decision layer

| Setup | Held-out F0.5 | No-match F0.5 | No-match wrongly matched |
|---|---|---|---|
| Perfect matcher on these candidates (upper bound) | 0.9987 | — | — |
| **Second-stage model with group features, threshold 0.7** | **0.9934** | **0.9929** | **0.7%** |
| Same, with one-to-one assignment | 0.9934 | 0.9929 | 0.7% |
| Second-stage model, group features only, one-to-one + relative threshold | 0.9934 | 0.9890 | 1.1% |
| Best first-stage model (from 04) | 0.9917 | 0.9822 | 1.8% |

- The second-stage model adds +0.0017 and more than halves wrong matches for no-match records (1.8% → 0.7%).
- One-to-one assignment adds nothing once the second-stage model is in place.
- By country: India 0.9931, US 0.9936.
- Expect somewhat lower at full scale, because candidate recall drops to about 0.984 (section 02 in `GPU_RUN_UPTO06.md`).

## 08: France check (no labels, full test pool per country, TF-IDF top-10)

| Country / normalization | Pool | Top-1 cosine (median) | Top-1 shares a house number | Top-5 share a house number | Neighbours with cosine ≥ 0.7 |
|---|---|---|---|---|---|
| France L1 | 1.43M | 0.950 | 0.833 | 0.553 | 6.67 |
| **France L4** | 1.43M | **1.000** | **0.877** | 0.566 | 6.67 |
| India L1 | 4.72M | 0.937 | 0.896 | 0.719 | 4.25 |
| India L4 | 4.72M | 0.986 | 0.933 | 0.769 | 5.35 |
| US L1 | 3.82M | 0.934 | 0.833 | 0.568 | 4.38 |
| US L4 | 3.82M | 0.988 | 0.845 | 0.571 | 4.69 |

- **L4 helps in every country.** The top-1 cosine median rises to about 0.99–1.0, and house-number agreement improves most for France (+0.044) and India (+0.037).
- **France with L4 behaves like the US on these signals:**
  - top-1 house-number agreement: 0.877 for France, 0.845 for the US;
  - top-5 agreement: 0.566 for France, 0.571 for the US.
- **France has more near-duplicate distractors** (6.7 neighbours with cosine ≥ 0.7, against about 4.5–5.4 elsewhere). The name and house-number features will do more of the work there.

## Issues found and fixed during the run

1. **Step 08 ran out of memory** on India's 4.7M-record test pool, three times.
   - It now streams the pool through 8 worker processes in 50k-record chunks. Each worker returns only document frequencies, then only its top-10.
   - Results match the original on the toy data.
2. **Step 06 was slow** because the exact nearest-neighbour search ran in numpy on the CPU.
   - It now runs on the GPU with `torch.topk`, giving the same results. Encoding now uses half precision (2.2× faster).
   - Off-the-shelf evaluation dropped from about 2.5 h to about 32 min.
3. **Step 06 fine-tuning slowed down** on its third view, because GPU memory from earlier models wasn't freed and Windows moved the overflow to system RAM. It still finished correctly.
   - In the final pipeline, run each fine-tuning job in its own process.
4. **My own mistake:** I ran step 08 in parallel with step 06, which ran out of RAM and crashed step 06. I restarted step 06 cleanly from scratch, and no results were affected.

## Recommended final pipeline

1. **Text cleaning:** L4 canonical forms plus the learned dictionary.
2. **Candidate search:** TF-IDF forward 10 + reverse 3, **merged with fine-tuned e5 (name + address) top-K**.
   - Both run on the GPU, streamed through the pool as in `eda.py`.
3. **Pair features:** CTX, NAME, ADDR, NUM, plus embedding cosines and the cross-encoder score.
   - The cross-encoder adds about 2 hours of scoring for about +0.0003 F0.5; drop it if time is short.
4. **Matcher:** LightGBM, then the second-stage model with group features, then a threshold of about 0.7.
5. **`candidate_pairs.tsv`:** use the cheap re-ranker, keeping p ≥ 0.003 (about 4 candidates per record).

# Part B: `--quick` run, 2% sample

This part summarises the `run_all.py --quick` run. The full tables are in `er_work/SUMMARY.md`, and the per-step logs are in `er_work/logs/`.

## Status

| Step | Status | Time |
|---|---|---|
| 00 prepare | OK | 2.8 min |
| 01 normalization | OK | 7.9 min |
| 02 blocking | OK | 5.9 min |
| 03 features | OK | 2.2 min |
| 04 matcher | OK | 9.3 min |
| 05 decision | OK | 3.8 min |
| 08 France probe | **Partly done.** France L1, France L4 and India L1 are finished. India L4, US L1 and US L4 are missing. | — |
| 99 report | OK | — |
| `--gpu` run (06 embeddings, 07 cross-encoder, 5% universe) | **Not started** | — |

**Step 08 fix.** As shipped, step 08 built the whole hashed TF-IDF matrix for each country's test pool at once. India's pool is 4.7M records, and the step ran out of memory on the 32 GB machine twice.

`er_lab/08_france_probe.py` now works in two parallel passes:
1. Workers return only document frequencies.
2. Each worker hashes its chunk and scores it against the queries, then returns only its top-10.

On the toy data, the new version gives exactly the same results as the original. France took about 90 seconds per level, down from 250. The re-run on the real data was then blocked by the permission classifier. To finish the step, see "What's left to run" at the end.

## Sample used for evaluation (U1, 2% of training)

- 43,948 Source 1 records and a pool of 205,673 records, about 4.68 pool records per Source 1 record.
- 5.7% of Source 1 records have no match, and 26.1% of pool records match nothing. Both are the same as in the full data.
- The learned native-script → Latin dictionary has 1,343 name entries. It covers **97.8%** of native-script name tokens in U1.

## 1. Normalization (01): canonical forms plus the learned dictionary work best

| Normalization | R@10 | R@10 India | Native-script R@10 |
|---|---|---|---|
| L0 raw | 0.968 | 0.930 | 0.749 |
| L1 clean | 0.959 | 0.905 | 0.619 |
| L2 anyascii transliteration | 0.977 | 0.950 | 0.864 |
| L3 + learned dictionary | 0.984 | 0.968 | 0.991 |
| **L4 canonical, word tokens** | **0.993** | **0.987** | **0.995** |

- The learned dictionary is the biggest single gain on Indic-script names: native R@10 rises from 0.86 to 0.99.
- With L4 text, word tokens beat character 2–4-grams (0.993 vs 0.987) and are about 9× faster.
- Name-only and address-only views are much worse (0.864 and 0.914), so both fields are needed.
- Adding phonetic skeletons makes results worse (0.974).

## 2. Blocking (02): reverse search keeps recall as the pool grows

| Candidates | Recall (2% sample) | Mean candidates per Source 1 record | Recall extrapolated to 10.3M pool |
|---|---|---|---|
| forward top-10 | 0.987 | 10 | 0.968 |
| forward top-50 | 0.995 | 50 | 0.985 |
| reverse top-3 | 0.992 | 14 | 0.981 |
| **forward 10 + reverse 3** | **0.995** | **18** | **0.988** (F0.5 ceiling 0.996) |
| forward 50 + reverse 3 (written to disk) | 0.997 | 54 | — |

- **Recall loss as the pool grows (per 10× pool size):**
  - forward top-10 alone loses 0.011;
  - forward 10 + reverse 3 loses 0.004.
  - This shows the reverse search matters at full scale.
- **Other blocking methods are clearly worse:**
  - dense SVD-128 embeddings (exact or HNSW search): 0.92–0.94;
  - exact-key blocking: 0.93 recall with 82 candidates.
- **Caveat:** in `--quick` mode, the extrapolation rests on only two points (1% and 2%). The `--gpu` run adds 5% and 10%.

## 3. Features (03) and matcher (04)

- **Best single features by AUC:**
  - blocking cosine: 0.993;
  - reverse margin: 0.986;
  - address containment: 0.976.
- **The best matcher uses the CTX, NAME, ADDR and NUM feature groups:** held-out F0.5 **0.9917**. Precision is 0.996 and recall 0.985.
- **Effect of each feature group:**

  | Change | F0.5 | Difference |
  |---|---|---|
  | Add COMP (comma-component address matching) | 0.9916 | no gain; skip it |
  | Drop NUM | 0.9904 | −0.0013 |
  | Drop NAME | 0.9842 | −0.0075 |
  | CTX only | 0.9681 | −0.0236 |

- **Feature importance:** `cos` has 45% of the gain and `rev_rank` 39%, so the blocking context does most of the work.
- **Smaller `candidate_pairs.tsv`:** the cheap re-ranker, keeping candidates with p ≥ 0.003, cuts the list to **3.97 candidates per Source 1 record** with recall 0.9968. Keeping all 54 gives 0.9974.
- **Transfer between countries as a stand-in for France:**
  - training on US and applying to India loses 0.0069 F0.5;
  - India → US loses 0.0033.
  - The threshold learned on the source country is almost the same as the best threshold for the target (0.9845 vs 0.9846). So a single threshold should carry over reasonably to France.

## 4. Decision layer (05)

| Setup | Held-out F0.5 |
|---|---|
| Perfect matcher on these candidates (upper bound) | 0.9992 |
| **Second-stage model with group features, then one-to-one + relative threshold** | **0.9936** |
| Second-stage model with group features, then a plain threshold | 0.9935 |
| Best first-stage model, one-to-one + threshold | 0.9917 |
| Best first-stage model, plain threshold | 0.9911 |

- By country: US 0.9942, India 0.9927.
- Records with no match: F0.5 0.987, with 1.3% of them wrongly given a match.
- The second-stage model adds about +0.002. One-to-one assignment and the expected-F0.5 rule each add less than 0.001 once it is in place.
- These scores are on the 2% sample. At full scale, candidate recall drops to about 0.988 (section 2), so the real F0.5 should be somewhat lower.

## 5. France check (08, no labels, full test pool)

| Country / normalization | Test pool | Top-1 cosine (median) | Top-1 shares a house number | Neighbours with cosine ≥ 0.7 |
|---|---|---|---|---|
| France L1 | 1.43M | 0.950 | 0.833 | 6.67 |
| France L4 | 1.43M | 1.000 | 0.877 | 6.67 |
| India L1 | 4.72M | 0.937 | 0.896 | 4.26 |

- **L4 French canonical forms help.** Compared with L1, the top-1 cosine median rises from 0.95 to 1.00, and top-1 house-number agreement from 0.833 to 0.877.
- **France has more near-duplicate distractors:** 6.7 neighbours with cosine ≥ 0.7, versus 4.3 for India.
  - Examples in the log: "Dunkerque Fetes" at no. 9 next to "Mediation Visages Fetes" at no. 3, on the same street.
  - This means house numbers and name features will matter more for France than blocking cosine does.

## Status of the quick run's step 08

The quick run's step 08 was never re-run. The complete France, India and US check is in the `--gpu` run (Part A, section 08). It uses the same full test pool, so the numbers apply to both runs.
