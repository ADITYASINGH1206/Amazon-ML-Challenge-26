# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Omnivision    
**Team Members:** Aditya Kumar Singh, Aditya Raj Gupta and Sharad Polackal Sunil
**Submission Date:** September 2026

---

## 1. Executive Summary
We propose a high-precision, GPU-accelerated, multi-stage Entity Resolution architecture that resolves noisy business records across three heterogeneous data sources ($S_1$, $S_2$, $S_3$) across the US, India, and France. Our approach couples **Level-4 domain-specific normalization** and **native-script Indic transliteration** with a **contrastively fine-tuned multilingual transformer encoder (`multilingual-e5-small`)**, executing exact nearest-neighbor blocking directly on GPU tensor cores. Candidates are scored using a two-stage gradient boosted decision tree pipeline featuring character-trigram IDF, fuzzy house-number alignment, group-aware candidate competition stacking, and a globally constrained one-to-one assignment policy, achieving a verified held-out macro $F_{0.5}$ of **0.9852** and a Public Leaderboard score of **0.9811**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) over the 24+ million records across training and test splits revealed critical structural noise patterns:
1. **Transliteration & Script Disparities:** ~45% of Indian business names in Sources 2 and 3 appear in native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati) while Source 1 is predominantly Latin script. Off-the-shelf transliterators introduce systematic phonetic drift.
2. **Missing Addresses & Asymmetric Namesakes:** 66% of missed true matches in baseline models were due to empty or truncated pool addresses where identical business names could not be differentiated from geographic namesakes.
3. **Address Typographical Drift:** Minor house-number typos (e.g., `#12-B` vs `#128`) broke strict token matching.
4. **Pool Density Shift:** The test set pool contains 5.75 records per $S_1$ entity (vs 4.68 in train) with ~40% ownerless distractor records.
5. **Cold-Start France Domain:** The evaluation set introduces France, unseen in the training distribution, demanding models agnostic to specific country labels.

### 2.2 Solution Strategy
- **Approach Type:** Contrastive Deep Embedding Blocking + Hierarchical LightGBM Classification + Group Stacking + 1-to-1 Hungarian/Greedy Assignment.
- **Core Innovations:**
  1. *Learned Transliteration Dictionary:* Automatically learned token correspondences between native scripts and Latin alphabet directly from training ground-truth pairs.
  2. *Exact Dual-Direction GPU Blocking:* Forward top-10 ($S_1 \to \text{pool}$) combined with reverse top-3 ($\text{pool} \to S_1$) nearest-neighbor search with exact FP32 rescoring, ensuring high recall ($>99.2\%$) under high distractor density.
  3. *Group-Aware Stacking:* Captures intra-query candidate competition, relative score margins, and embedding cluster density to penalize ambiguous duplicates.

---

## 3. Candidate Generation (Blocking)
To avoid the intractable $O(N \times M)$ pairwise comparison ($1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ pairs), candidate generation proceeds via metric embedding search:
- **Blocking Keys & Vectors:** Each normalized record (`"query: {name} , {address}"`) is encoded into a 384-dimensional unit vector using our fine-tuned `multilingual-e5-small` model.
- **Search Execution:** Search is partitioned strictly by country. Inside each country:
  - **Forward Search:** Top-10 pool records for each $S_1$ query.
  - **Reverse Search:** Top-3 $S_1$ entities for each pool record to recover matches suppressed by dense clusters.
- **Candidate Pool Size:** Generates an average of ~20 candidate pairs per $S_1$ record (total ~35M candidate pairs).
- **Cheap Reranker Pruning:** A lightweight LightGBM model evaluating search context and token overlaps filters low-probability candidates down to an optimal candidate set of **~4.0 to 6.0 pairs per $S_1$ record** (`candidate_pairs.tsv`) with $<0.001$ recall loss.

---

## 4. Matching Model

### 4.1 Feature Engineering (50+ Signals)
1. **Search Context Features:** Cosine similarity, forward rank, reverse rank, top-1 score gap, reverse score margin, mutual nearest neighbor flag, candidate list size.
2. **Name Similarity Features:** Character trigram Jaccard, word token Jaccard, core name containment, exact alphanumeric equality, consonant skeleton similarity, Jaro-Winkler, token sort ratio, partial ratio, query core-name IDF, and candidate Indic script flag.
3. **Address Similarity Features:** Address token Jaccard, address character trigram Jaccard, substring containment, address IDF query/candidate coverage, address token sort ratio, empty address indicator.
4. **Numerical & Postal Features:** Shared house number count, number conflict indicator, first number equality, postal/PIN code equality, minimum Levenshtein distance on house numbers.
5. **Information-Theoretic Features (V2 & V3):** Trigram IDF cosine similarity, unshared maximum IDF penalty, name core uniqueness within country.

### 4.2 Model Type & Architecture
- **Stage 1 (Full Matcher):** LightGBM gradient boosted decision trees trained with binary log-loss, optimized for ranking discrimination.
- **Stage 2 (Group Stacking):** 5-fold out-of-fold stacked LightGBM classifier incorporating query-level aggregation statistics: best candidate probability share, runner-up margin, competing candidates count, and pool-side competition.
- **Decision Rule & One-to-One Optimization:**
  - Probability threshold tuned specifically on held-out slice B to maximize Macro $F_{0.5}$.
  - Globally enforced **one-to-one constraint**: each pool record ($S_2$ or $S_3$) can match at most one $S_1$ entity (consistent with 100% of ground-truth integrity).

---

## 5. Results & Error Analysis

### 5.1 Validation Progression

| Iteration | Description | Validation Macro $F_{0.5}$ | Public Leaderboard |
|---|---|---|---|
| **v1** | Base E5 fine-tuning + LightGBM + stage-2 stacking | 0.9796 | 0.9744 |
| **v2** | Test-density alignment + V2 pair features + French canonicalization | 0.9826 | 0.9793 |
| **v4 (Final)** | Name uniqueness + fuzzy house numbers + core sibling features | **0.9852** | **0.9811** |

*Held-out validation is evaluated on Slice B (disjoint priority 0.15–0.30, folds 3–4) with full-scale pool density.*

### 5.2 Error Analysis
- **False Positives (Wrong Merges):** ~12% of error budget. Primarily commercial chains sharing identical brand names in identical commercial districts with omitted unit/suite numbers.
- **False Negatives (Missed Matches):** ~88% of error budget. 65% of missed true matches had completely blank pool addresses, where conservative thresholding correctly favors precision over uncertain matching (penalized heavily by $\beta = 0.5$).

---

## 6. Conclusion
By pairing multilingual semantic representation learning on GPU tensor cores with hierarchical gradient boosted ranking and strict one-to-one combinatorial assignment, our solution achieves high efficiency and competitive accuracy (0.9811 Leaderboard $F_{0.5}$). The pipeline executes end-to-end in under 45 minutes on modern GPU hardware, adapting to cold-start international jurisdictions without manual rule tuning.

---

## Appendix

### A. Code Artefacts & Reproduction
The complete self-contained pipeline is located in `code/business_entity_resolution/`:
- `src/run_pipeline.py`: Orchestrates all stages (`load`, `dict`, `norm`, `finetune`, `embed`, `block`, `feats`, `train`, `predict`).
- `src/ber/`: Modular libraries for normalization (`prep.py`), embedding (`embed.py`), GPU blocking (`knn.py`), feature generation (`feats.py`, `v3.py`), and model inference (`model.py`).
- **One-Command Execution:**
  ```bash
  cd code/business_entity_resolution
  python run_pipeline.py
  ```
  Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv` and validates formatting automatically.
