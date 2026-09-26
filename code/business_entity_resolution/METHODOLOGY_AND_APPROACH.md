# Comprehensive Methodology and Approach: Business Entity Resolution Pipeline

## Table of Contents
1. [Introduction and Problem Statement](#1-introduction-and-problem-statement)
2. [Evaluation Metrics and Objective Function](#2-evaluation-metrics-and-objective-function)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Stage 0: Pre-processing & Text Normalization](#4-stage-0-pre-processing--text-normalization)
5. [Stage 1: Blocking and Candidate Generation](#5-stage-1-blocking-and-candidate-generation)
6. [Stage 2: Advanced Feature Engineering](#6-stage-2-advanced-feature-engineering)
7. [Stage 3a: Fast Filtering with LightGBM](#7-stage-3a-fast-filtering-with-lightgbm)
8. [Stage 3b: Precision Re-ranking with Cross-Encoders](#8-stage-3b-precision-re-ranking-with-cross-encoders)
9. [Stage 4: Ensembling, Thresholding, and Global Deduplication](#9-stage-4-ensembling-thresholding-and-global-deduplication)
10. [Path to >0.98: Identified Flaws and the Optimization Strategy](#10-path-to-098-identified-flaws-and-the-optimization-strategy)
11. [Conclusion](#11-conclusion)

---

## 1. Introduction and Problem Statement

The goal of the Amazon ML Challenge 2026 is Business Entity Resolution. We are tasked with matching business entities from a primary source (Source 1) to duplicate entities present in two other distinct sources (Source 2 and Source 3). 

### Key Constraints:
- **No External Data:** The use of external databases, APIs, or geocoding services is strictly prohibited. All insights, embeddings, and matching algorithms must rely entirely on the raw strings provided (Business Name, Business Address, and Country).
- **Scale:** The datasets contain millions of potential pairs. Comparing every Source 1 entity against every Source 2 and Source 3 entity (O(N*M)) is computationally impossible. 
- **Noisy Data:** The provided names and addresses suffer from typos, differing abbreviations, missing components, varying legal suffixes (e.g., "Corp", "LLC", "S.A.R.L."), and cross-lingual differences (US vs. France vs. India).

To solve this, we have architected a multi-stage, cascading machine learning pipeline. The design philosophy of this pipeline is to start with high-recall heuristic and vector-based filtering, progressively narrowing down the search space until we apply extremely precise, compute-heavy deep learning models (Cross-Encoders) only on the most difficult boundary cases.

---

## 2. Evaluation Metrics and Objective Function

Before diving into the algorithms, it is paramount to deeply understand the scoring mechanism, as it dictates the hyperparameters and behavior of the entire pipeline. 

The challenge is scored on **Macro F0.5 (precision-weighted)**. 

### The Formula:
For a single entity query from Source 1:
- Let **P** = Precision (True Positives / Predicted Matches)
- Let **R** = Recall (True Positives / Actual Matches)

The F0.5 score is computed as:
```text
F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R)
F0.5 = (1.25 * P * R) / (0.25 * P + R)
```

### The Implications of F0.5:
1. **Precision Dominates:** In an F0.5 metric, a false merge (False Positive) penalizes the score roughly **twice as much** as a miss (False Negative). This means our models must be highly conservative. When in doubt, it is mathematically better to predict nothing than to predict a wrong match.
2. **The Singleton Rule:** Singletons are Source 1 entities that have zero true matches in Source 2 or Source 3. 
   - If the model correctly predicts an empty list, the score for that entity is `1.0`.
   - If the model predicts *even a single false match*, precision becomes `0.0`, and the F0.5 score instantly drops to `0.0`. 
   - Because the final score is the *Macro Average* across all Source 1 entities, false positives on true singletons are catastrophic to the overall score.

Because of this metric, the entirety of Stage 3 (Classification) and Stage 4 (Thresholding) is aggressively tuned to guard precision at all costs.

---

## 3. System Architecture Overview

Our solution follows a classic two-tower candidate generation and cascade re-ranking architecture. It is split into 5 distinct sequential stages:

1. **Stage 0 (Pre-processing):** Cleans the raw strings and standardizes abbreviations across languages.
2. **Stage 1 (Blocking):** Reduces the search space from billions of pairs to ~50 candidate pairs per Source 1 entity using ultra-fast retrieval methods.
3. **Stage 2 (Feature Engineering):** Generates 40 handcrafted statistical and heuristic features for every generated candidate pair.
4. **Stage 3 (Classification Cascade):**
   - **Stage 3a (LightGBM):** Scores all 50 candidates using the 40 features and trims down to the Top-5.
   - **Stage 3b (Cross-Encoder):** A DeBERTa-based transformer model scores the Top-5 candidates by processing the textual relationship between the two entities directly.
5. **Stage 4 (Inference & Post-Processing):** Ensembles the scores, applies a margin-based threshold to reject ambiguous cases, and formats the output.

---

## 4. Stage 0: Pre-processing & Text Normalization

Real-world business data is notoriously dirty. Two identical businesses might be recorded as "Acme Corporation, 123 Main Street" and "Acme Corp, 123 Main St." If we pass these raw strings into our algorithms, both our sparse TF-IDF and our dense embeddings will suffer.

We implemented a robust, country-agnostic normalization pipeline in `preprocess.py`:

### Steps Taken:
1. **Unicode Normalization (NFKD):** Accents and special characters (highly common in the French subset) are stripped down to their base ASCII equivalents.
2. **Punctuation Stripping:** Characters like `@` and `&` are replaced with "at" and "and" respectively, and all other non-alphanumeric noise is removed.
3. **Legal Suffix Standardization:** We mapped dictionaries of English, French, and Indian legal terms. For example, `\binc\b` is mapped to `incorporated`, `\bsarl\b` to `societe a responsabilite limitee`, and `\bpvt\b` to `private`. This ensures that legal suffixes do not artificially lower string similarity scores.
4. **Address Standardization:** Road types (`st` -> `street`, `r` -> `rue`, `marg` -> `marg`) are fully expanded.
5. **Component Extraction:** We implemented Regex parsers to extract the `street_number` and `postal_code` (accounting for differing lengths like 5 digits for the US/France and 6 digits for India). 

By heavily normalizing the text, we drastically improve the hit rate of our Blocking algorithms in the next stage.

---

## 5. Stage 1: Blocking and Candidate Generation

Blocking sets the absolute ceiling for our Recall. If a true match is not found in Stage 1, it can never be predicted by the model in Stage 4. 

We used a **Multi-Strategy Blocking** approach, heavily partitioned by the `country` column to avoid cross-country comparisons (which are logically impossible for brick-and-mortar entities).

### Strategy 1: Composite-Key Inverted Index
We built a fast hash-map (Inverted Index) where the keys are handcrafted combinations of entity attributes. If two entities share a key, they become candidates. Keys include:
- Country + First significant word of the name
- Country + Postal code
- Country + First name word + First address word
- Phonetic (Soundex) key of the name
*Why?* This strategy is O(1) at query time and perfectly captures entities that have exact matches on specific sub-components, even if the rest of the string is noisy.

### Strategy 2: TF-IDF Sparse Character N-Grams
We fitted a TF-IDF vectorizer using Character N-grams (length 3 to 4) on all concatenated `name_address` strings. 
*Why Character N-Grams?* They are incredibly robust to typos and missing words. "Acme Corp" and "Acmme Corp" will share a massive amount of character n-grams, despite sharing zero exact word tokens.
To compute this across millions of rows without OOM (Out of Memory) errors, we utilized `sparse_dot_topn`, which calculates the exact Top-K cosine similarities directly in C++ using Compressed Sparse Row (CSR) matrices.

### Strategy 3: Dense PyTorch ANN Retrieval
Lexical matching (TF-IDF) fails when synonyms or abbreviations are used. We utilized `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to encode all entities into 384-dimensional dense vectors. 
Using raw PyTorch Matrix Multiplication (optimized on the GPU via float16 batches), we performed exact nearest neighbor search. 
*Why?* The multilingual aspect is critical. It maps semantic concepts in French, English, and Hindi into the same vector space, catching conceptual duplicates that have zero string overlap.

**Union:** The candidates from all three strategies are merged. To prevent explosion, we cap the candidates at `50` per query, prioritizing those found by multiple strategies simultaneously.

---

## 6. Stage 2: Advanced Feature Engineering

Once we have ~50 candidates per entity, we must extract features that tell a machine learning model *why* these two entities might be a match. We handcrafted 40 features in `features.py`.

### Feature Categories:
1. **String Distance Metrics (via `rapidfuzz`):**
   - *Jaro-Winkler:* Heavily weights prefix matches. Perfect for business names where the first word is the most important (e.g., "Amazon Web Services" vs "Amazon").
   - *Levenshtein Ratio:* Standard edit distance.
   - *Token Sort Ratio & Token Set Ratio:* Alphabetically sorts tokens before comparing. This completely solves out-of-order strings like "Corp Acme" vs "Acme Corp".
   - *Longest Common Subsequence (LCS):* Catches abbreviations and dropped words.

2. **Component Specific Matching:**
   - Instead of just comparing the full string, we extract specific signals.
   - *Street Number Match (1.0, 0.0, -1.0):* If street numbers differ, it is almost certainly a different location.
   - *Postal Code Edit Distance:* If postal codes are off by 1 digit, it might be a typo. If they are entirely different, it's a strong negative signal.
   - *Numeric Overlap:* Computes the Jaccard similarity of all digits found in the address.

3. **TF-IDF & Semantic Similarity:**
   - *IDF-Weighted Token Overlap:* Words like "The", "LLC", or "Street" have low IDF. Words like "Zendesk" or "Palantir" have high IDF. We compute the overlap of tokens weighted by their global IDF. Sharing a rare word is a massive signal for a match.
   - *Embedding Cosine:* We pass the cosine similarity from the dense retrieval stage directly as a feature.

4. **Length Differentials:**
   - Ratios of string lengths and token counts. A massive disparity usually indicates one string contains extraneous metadata.

By breaking the comparison down into 40 distinct mathematical perspectives, we provide the LightGBM classifier with an incredibly rich feature space.

---

## 7. Stage 3a: Fast Filtering with LightGBM

Scoring 50 candidates per entity using a Transformer model (Cross-Encoder) is too slow. We use LightGBM as a "Cascade Filter".

### The Algorithm
LightGBM (Light Gradient Boosting Machine) is a tree-based framework that builds decision trees leaf-wise. We configured it with `objective="binary"` and `metric="binary_logloss"`.

### Training Dynamics
The dataset of candidate pairs is heavily imbalanced (most candidates generated by Blocking are false positives). We pass the 40 features into LightGBM to train it to differentiate between a true match (label 1) and a false candidate (label 0). 

*Output:* For every candidate, LightGBM outputs a probability `[0.0, 1.0]`. We sort the 50 candidates by this probability and throw away all but the **Top-5**. This reduces the workload for the Cross-Encoder by 90%.

---

## 8. Stage 3b: Precision Re-ranking with Cross-Encoders

While LightGBM relies on our handcrafted features, it cannot read the actual text. To push precision to the absolute limit, we fine-tune a Cross-Encoder.

### Bi-Encoder vs. Cross-Encoder
In Stage 1, we used a Bi-Encoder (SentenceTransformer). A Bi-Encoder processes String A and String B independently into vectors, and we compute the cosine similarity. 
A Cross-Encoder, however, concatenates String A and String B together separated by a `[SEP]` token, and feeds the entire sequence into a deep Transformer stack. This allows the self-attention mechanism to look at words in String A and String B simultaneously at every layer. It is vastly more accurate, but vastly slower.

### Model Choice and Architecture
We selected `microsoft/mdeberta-v3-base`. 
- **Multilingual:** It natively understands the French and Hindi variations in the dataset.
- **DeBERTa:** Uses disentangled attention, which handles positional relative weights much better than standard BERT/RoBERTa.

### Hard Negative Mining
To train this model effectively, we do not feed it random pairs. We use **Hard Negative Mining**.
We look at the predictions from LightGBM. Any pair that LightGBM gave a high probability to (e.g., >0.9) but is actually a False Positive is deemed a "Hard Negative". We heavily sample these confusing pairs and feed them to the Cross-Encoder during training. This forces the Cross-Encoder to learn the minute semantic differences that fooled the statistical LightGBM model.

---

## 9. Stage 4: Ensembling, Thresholding, and Global Deduplication

The final stage is where raw probabilities are turned into hard predictions. 

### 1. Weighted Ensemble
We compute a final `ensemble_score` by taking a weighted average of the LightGBM probability and the Cross-Encoder probability. 
```python
ensemble_score = (0.35 * lgbm_prob) + (0.65 * ce_prob)
```
The Cross-Encoder is given higher weight due to its superior language understanding, but LightGBM acts as a stabilizing anchor (since it understands exact string overlap and numeric matches better than sub-word tokenizers).

### 2. Margin-Based Thresholding
Due to the F0.5 metric, we cannot simply use a `0.5` threshold. We use a grid-search on the validation set to find the optimal threshold `tau` (often near `0.90` or `0.95`).
However, we also implement a **Margin Delta**. 
If a Source 1 entity has two highly scored matches, say `0.98` and `0.96`, we don't just pick the top one. Because the scores are so close, the model is clearly confused. A false merge costs twice as much as a miss. Therefore, if the difference between the Top 1 score and Top 2 score is less than `margin_delta` (e.g., 0.08), we **reject both and predict nothing**. We only predict a match if there is a clear, unambiguous winner.

### 3. Global Deduplication
Logically, if Entity A matches Entity B, no other entity should match Entity B. 
We perform global deduplication. If multiple Source 1 entities try to claim the same Source 2 entity, we give it to the S1 entity that generated the highest ensemble score, dropping the connection for the others.

---

## 10. Path to >0.98: Identified Flaws and the Optimization Strategy

Despite the robust architecture, the current codebase peaks around ~0.804 due to critical mathematical misalignments with the F0.5 metric. During analysis, we identified exactly what is suppressing the score and how to fix it:

### A. The Probability Calibration Crisis
In `train_lgbm.py`, the code dynamically balances the dataset using `scale_pos_weight = n_neg / n_pos`. Because blocking generates ~50 negatives for every 1 positive, this weight is ~50.
**The Impact:** The model is penalized 50x more for missing a match than for predicting a false positive. This forces the model to heavily prioritize Recall over Precision, which is the **exact opposite** of the F0.5 objective. The model's probabilities inflate, pushing all marginal candidates to `0.99`. This renders the `margin_delta` logic useless, leading to thousands of false positives and ruining the score for True Singletons (which drops from 1.0 to 0.0 instantly).
**The Solution:** Force `scale_pos_weight = 1.0` (or `0.8`). The model must learn the true prior. This will spread the probabilities out naturally, allowing the thresholding and margin logic to execute flawlessly.

### B. Missing Source-Specific Constraints
The problem statement matches S1 against S2 and S3. An S1 entity can logically only exist *once* in S2 and *once* in S3. 
**The Impact:** The current code allows multiple matches from the same source if they pass the threshold.
**The Solution:** Enforce a strict "Top-1 per Source" rule during inference. Group by the `S2-` and `S3-` prefixes and cull secondary matches.

### C. The City/State Feature Bug
**The Impact:** In `preprocess.py`, the function `extract_city_state()` is perfectly implemented but is **never executed**. Address matching is relying purely on string overlap, causing businesses with identical names in different cities to falsely merge.
**The Solution:** Wire `extract_city_state()` into the `preprocess_dataframe()` pipeline and expose `city_match` as a binary feature to LightGBM.

### D. Extending the Grid Search
**The Impact:** The grid search for the optimal threshold (`tau`) is hard-capped at `0.98` in `config.py`. For F0.5, the mathematical optimal threshold is frequently between `0.990` and `0.999`. 
**The Solution:** Increase `THRESHOLD_SEARCH_MAX` to `0.999` and lower the step size to allow the pipeline to dynamically find the ultra-high precision cutoff.

### E. Cross-Encoder Input Tagging
**The Impact:** The cross-encoder receives the raw concatenated string: `Acme Corp 123 Main St`. It struggles to map alignments.
**The Solution:** Restructure the input text in `train_cross_encoder.py` to use tags: `Name: {name} Address: {addr}`. This simple text formatting dramatically improves Cross-Encoder precision.

---

## 11. Conclusion

The pipeline implements state-of-the-art Entity Resolution methodologies, balancing massive scale via multi-strategy blocking with pinpoint accuracy via gradient boosting and DeBERTa cross-encoders. 

By addressing the probability calibration flaw, injecting the missing geographic features, and enforcing strict source-level constraints, the architecture is mathematically primed to heavily bias towards Precision. Securing precision protects the scores of singletons, minimizing false merge penalties, and forms the definitive blueprint for crossing the `0.98` F0.5 threshold.
