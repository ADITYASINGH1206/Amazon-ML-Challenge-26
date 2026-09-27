# Entity Resolution Pipeline: Optimization Analysis & Action Plan

## Context & Goal
The objective is to improve the current business entity resolution pipeline's score from **~0.804** to **> 0.98** for the Amazon ML Challenge. 

The evaluation metric is **macro F0.5 (precision-weighted)**. 
- A false merge costs approximately twice as much as a missed merge.
- Singletons (entities with no match) must receive an empty list. Predicting *any* match for a true singleton yields a score of 0 for that entity. 

To achieve a score >0.98, the pipeline must maintain near-perfect precision while preserving high recall. 

## Current Architecture Overview
The current codebase in the `src/` directory uses a robust 4-stage pipeline:
1. **Preprocessing (`preprocess.py`)**: Normalizes text and extracts basic components (postal code, street number).
2. **Blocking (`blocking.py`)**: Uses multi-strategy candidate generation (Inverted Index + Sparse TF-IDF + Dense PyTorch ANN) to reduce millions of pairs down to ~50 candidates per entity.
3. **Feature Engineering (`features.py`)**: Computes 40 similarity features (Jaro-Winkler, LCS, TF-IDF overlaps, dense cosine similarity) for the candidates.
4. **Modeling & Inference (`train_lgbm.py`, `train_cross_encoder.py`, `inference.py`)**: Uses a cascade approach. LightGBM filters candidates down to a Top-K, which are then re-ranked by a Cross-Encoder (DeBERTa). An ensemble score is computed, followed by a margin-based thresholding and global deduplication.

---

## Identified Critical Flaws & Proposed Optimizations

### 1. Probability Calibration & Class Weights (Critical Precision Leak)
**Issue:** In `src/train_lgbm.py`, `scale_pos_weight` is dynamically computed as `n_neg / n_pos` (likely around ~50). This forces the model to heavily prioritize recall and treat misses as 50x worse than false positives. This pushes all probabilities toward 1.0, destroying the `margin_delta` logic in `inference.py` because hard negatives and true matches get squashed together near `0.99`.
- **Optimization:** Remove the dynamic `scale_pos_weight` (set it to `1.0` or even `< 1.0`). This will output naturally calibrated probabilities, allowing the margin-based deduplication to properly differentiate between a true match and a confusing hard negative.
- **Cross-Encoder Note:** The training data for the cross-encoder is artificially balanced 1:1. Adjusting the sampling ratio to a more natural distribution (e.g., 1 positive to 5-10 negatives) will prevent the cross-encoder from overpredicting.

### 2. Missing Source-Specific Cardinality Constraints
**Issue:** The prompt defines matching Source 1 (S1) against Source 2 (S2) and Source 3 (S3). Logically, an S1 entity should match **at most one** entity in S2 and **at most one** entity in S3. The current global deduplication allows an S1 entity to match multiple S2 entities if they both cross the threshold.
- **Optimization:** In `src/inference.py`, enforce a strict `Top-1 per Source` constraint during thresholding. Group candidates by their source prefixes (`S2-` and `S3-`) and keep only the absolute highest-scoring candidate per source.

### 3. Untapped City & State Features (Code Bug)
**Issue:** In `src/preprocess.py`, there is a heuristic function `extract_city_state()` which is defined but **never called** inside `preprocess_dataframe()`.
- **Optimization:** Call `extract_city_state()` to generate `city` and `state` columns. Then, in `src/features.py`, add explicit features like `city_match` and `state_match`. Explicit geographic mismatch features are the strongest safeguards against false positives for businesses with identical names.

### 4. Cross-Encoder Input Formatting
**Issue:** In `src/train_cross_encoder.py`, the input to the DeBERTa model is just the concatenated string `name_clean + " " + addr_clean`. The model lacks explicit boundaries.
- **Optimization:** Format the input strings with explicit structural tags. E.g., `Name: {name_clean} Address: {addr_clean} Country: {country}` or `[NAME] {name} [ADDR] {addr}`. This significantly boosts the attention mechanism's ability to align matching components.

### 5. Threshold Grid-Search Ceiling
**Issue:** The current grid search for the optimal threshold (`THRESHOLD_SEARCH_MAX`) is capped at `0.98`. For an F0.5 metric, the optimal threshold is often extremely high to guard precision.
- **Optimization:** Extend the grid search in `src/config.py` up to `0.999` with smaller step sizes (`0.001` or `0.002`) at the upper bound.

---

## Recommended Action Plan for Execution

To the analyzing Agent, please execute the following steps:

1. **Fix `src/train_lgbm.py`**:
   - Locate `train_lightgbm()`.
   - Hardcode `params["scale_pos_weight"] = 1.0` (or `0.8` to slightly penalize FPs). Remove the dynamic `n_neg / n_pos` calculation.

2. **Fix `src/preprocess.py` & `src/features.py`**:
   - In `preprocess.py::preprocess_dataframe()`, add `df[['city', 'state']] = df.apply(lambda r: pd.Series(extract_city_state(r['addr_clean'], r['country_clean'])), axis=1)`.
   - Update `features.py` to accept `city` and `state` arguments, and compute `city_match` and `state_match`. Add them to `FEATURE_NAMES`.

3. **Enhance `src/inference.py` Thresholding**:
   - Inside `apply_thresholding()` or `_global_dedup_with_margin()`, implement the logic that ensures an S1 entity only outputs **a maximum of two matches** (one starting with `S2-` and one starting with `S3-`). Drop any secondary matches from the same source.

4. **Update `src/train_cross_encoder.py`**:
   - Change the ratio of hard/easy negatives in `prepare_cross_encoder_data()` from `1:1` to `1:5` or `1:10` to calibrate the cross-encoder probabilities.
   - Restructure the text pair format from `name_addr` to `Name: {name} Address: {addr}`.

5. **Modify `src/config.py`**:
   - Change `THRESHOLD_SEARCH_MAX` to `0.999`.
   - Change `THRESHOLD_SEARCH_STEP` to `0.002`.

6. **Re-run the Pipeline**:
   - Execute `python run_pipeline.py preprocess` and subsequent stages to rebuild features and retrain models with the new calibrated logic.
