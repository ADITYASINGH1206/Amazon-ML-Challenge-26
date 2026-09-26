# SOLUTION.md: Entity Resolution Pipeline Optimization

## 1. Current Architecture & Pipeline
Your codebase implements a sophisticated, multi-stage candidate generation and reranking architecture:

- **Stage 0 (Preprocessing):** `preprocess.py` cleans text, standardizes multilingual legal/address suffixes, and extracts street numbers and postal codes.
- **Stage 1 (Blocking):** `blocking.py` generates candidates via three country-partitioned methods: 
  1. Composite-Key Inverted Index
  2. TF-IDF Sparse Dot Product (using `sparse_dot_topn`)
  3. FAISS Dense Multilingual Embeddings
  
  Candidates are unioned and capped at 50 per entity.
- **Stage 2 (Features):** `features.py` extracts 40 pairwise features (Jaro-Winkler, LCS, token subsets, semantic cosine).
- **Stage 3 (Modeling):** 
  - `train_lgbm.py` trains a LightGBM classifier with dynamically weighted positive classes. 
  - `train_cross_encoder.py` fine-tunes `mdeberta-v3-base` on a 1:1 ratio of positive to mixed negatives.
- **Stage 4 (Inference):** `inference.py` combines scores, drops candidates below a grid-searched threshold $\tau$ (capped at 0.98), and applies `_global_dedup_with_margin` to resolve conflicting S1 claims over the same S2/S3 entity.

---

## 2. Problem-by-Problem Audit & The ~0.804 Root Cause

### A. The Margin Delta Collapse (The Primary Score Killer)
- **What it does:** In `train_lgbm.py`, `scale_pos = n_neg / max(n_pos, 1)` applies a ~50x weight to positive samples. In `inference.py`, `_global_dedup_with_margin` rejects matches if `best_score - second_score < margin_delta` (0.08).
- **The Flaw:** By penalizing false negatives 50x more than false positives, LightGBM pushes all predictions (both true matches and hard negatives) toward $0.99$. Because the scores are artificially compressed at the top, the gap between the `best_score` and `second_score` shrinks below the 0.08 margin.
- **The Result:** The deduplicator erroneously marks thousands of true matches as "ambiguous" and assigns `None`, devastating your Recall and F0.5 score.

### B. Geographic Feature Neglect
- **What it does:** Address matching relies entirely on full-string overlaps.
- **The Flaw:** `extract_city_state()` is perfectly implemented in `preprocess.py` but is **never invoked** within `preprocess_dataframe()`. Consequently, `features.py` lacks `city_match` or `state_match` features.
- **The Result:** "Acme Logistics" in Paris and "Acme Logistics" in Lyon generate high string/semantic similarity. Without explicit city mismatch penalties, the model falsely merges them, generating catastrophic False Positives that instantly ruin Singleton scores.

### C. Threshold Grid Search Ceiling
- **What it does:** `THRESHOLD_SEARCH_MAX = 0.98`.
- **The Flaw:** F0.5 heavily penalizes False Positives. The mathematical optimal threshold for this dataset often rests between $0.985$ and $0.995$.
- **The Result:** The grid search stops too early, letting false positives slip through and tanking the Macro precision.

---

## 3. Blocking Optimization (Breaking the 93% Ceiling)
*If blocking recall is capped at ~93%, a perfect model can only score ~0.94.*

**Why is blocking failing to find 7% of matches?**
- **Address Dilution in TF-IDF:** In `blocking.py`, `TfidfVectorizer` fits on `name_addr`. Long addresses (e.g., "Near City Hall Building 4") dilute the character n-grams of short business names (e.g., "Zen"), causing true name matches to drop below the `top_k=30` cutoff.
- **S2/S3 Target Pooling:** `df_targets = pd.concat([df_s2, df_s3])` pools all targets into one index. If an S1 entity has 40 noisy variations in S2, they consume the entire `top_k=30` capacity, pushing out valid S3 matches.
- **Soundex Limitations:** Soundex performs terribly on French and Indian names.

**The Fix:**
- **Split TF-IDF:** Create two vectorizers—one strictly for `name_clean`, one for `addr_clean`.
- **Split Target Indices:** Run `sp_matmul_topn` independently for S2 and S3, extracting `top_k=20` from each.
- **Upgrade Phonetics:** Swap `jellyfish.soundex` for `jellyfish.metaphone`.

---

## 4. Feature / Matching Optimization

### LightGBM Calibration
Remove the dynamic `scale_pos_weight`. Set it to `1.0` (or `0.8` to slightly favor precision). This un-compresses the probabilities, allowing your `margin_delta` (0.08) to correctly differentiate a 0.95 true match from a 0.70 false positive.

### Cross-Encoder Text Tagging
In `train_cross_encoder.py`, the input is blindly concatenated: `ta = s1_text.get(...)`.
- **The Fix:** Inject semantic boundaries before tokenization: `f"[NAME] {n1} [ADDR] {a1}"`. This prevents the DeBERTa self-attention heads from aligning a business name in S1 with a street name in S2.

---

## 5. Prioritized Mitigation Plan

| Priority | Exact Problem | Proposed Solution | Expected Impact | Files to Edit |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | Probabilities compressed; `margin_delta` rejects true matches | Set `params["scale_pos_weight"] = 1.0`. | Massive jump (~0.804 $\rightarrow$ ~0.92); probabilities spread out, restoring valid matches. | `src/train_lgbm.py` |
| **P2** | Cross-city false merges | Call `extract_city_state()` in preprocessing; add `city_exact_match` feature. | Protects singletons; pushes F0.5 higher. | `src/preprocess.py`, `src/features.py` |
| **P3** | Threshold grid search capped at 0.98 | Change `THRESHOLD_SEARCH_MAX = 0.999`, `step = 0.002`. | Maximizes F0.5 mathematical peak. | `src/config.py` |
| **P4** | Address dilution drops matches out of top-K | Build separate TF-IDF indices for Name and Address. | Blocking recall pushes from 93% to >98%. | `src/blocking.py` |
| **P5** | Cross-Encoder lacks semantic boundaries | Format pairs as `[NAME] {name} [ADDR] {addr}`. | DeBERTa precision improves on edge cases. | `src/train_cross_encoder.py` |

---

## 6. Implementation Order (Step-by-Step)

### Step 1: Fix Core Configurations & Preprocessing (P2 & P3)
- **`src/config.py`**: Update `THRESHOLD_SEARCH_MAX = 0.999` and `THRESHOLD_SEARCH_STEP = 0.002`.
- **`src/preprocess.py`**: Inside `preprocess_dataframe()`, add:
  ```python
  df[['city', 'state']] = df.apply(lambda r: pd.Series(extract_city_state(r['addr_clean'], r['country_clean'])), axis=1)
  ```
- **`src/features.py`**: Update `FEATURE_NAMES` (now 42 features) and `extract_pair_features()` to include `city_exact_match` (1.0 for match, 0.0 for mismatch, -0.5 for missing) and `city_jaro_winkler`.

### Step 2: Fix the Model Calibration (P1)
- **`src/train_lgbm.py`**: Inside `train_lightgbm()`, remove `scale_pos = n_neg / max(n_pos, 1)` and hardcode `params["scale_pos_weight"] = 1.0`.
- **Action:** Run `python run_pipeline.py preprocess`, then `features`, then `train_lgbm`.

### Step 3: Upgrade Blocking (P4)
- **`src/blocking.py`**: 
  - Change `jellyfish.soundex` to `jellyfish.metaphone`.
  - In `tfidf_blocking_by_country`, create `vectorizer_name` (fitted on `name_clean`) and `vectorizer_addr` (fitted on `addr_clean`). Run `sp_matmul_topn` twice and union the candidate sets.

### Step 4: Fine-tune the Cross-Encoder (P5)
- **`src/train_cross_encoder.py`**: Modify the extraction loop in `prepare_cross_encoder_data()` to format strings:
  ```python
  t1 = f"[NAME] {row.n1} [ADDR] {row.a1} [CITY] {row.c1}"
  ```
- **Action:** Run `python run_pipeline.py train_cross_encoder`.

### Step 5: Final Evaluation
- **Action:** Run `python run_pipeline.py validate`. Observe the new optimal threshold $\tau$ (likely $\sim 0.990$) and the resulting validation F0.5.

---

## 7. Validation & Benchmarking
- **Validation Validation:** Check the logs from `evaluate_lgbm_on_val`. The `n_singletons_correct` must be $>98\%$ of `n_singletons_truth`. If singleton accuracy is dropping, your threshold is too low.
- **Blocking Validation:** Check the `evaluate_blocking_recall` logs. You should see `Blocking recall: > 0.985` before running the feature extraction.
