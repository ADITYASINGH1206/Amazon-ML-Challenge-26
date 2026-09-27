# amazonMLchlng: Business Entity Resolution (Amazon ML Challenge 2026)

For each Source 1 business record, this project finds the matching records in Source 2 and Source 3. The records are noisy (names, addresses, Indic scripts) and cover the US, India and, in the test set only, France. Results are scored with macro F0.5.

## Repository layout

| Path | What it is |
|---|---|
| `student_resource/code/business_entity_resolution/` | **The pipeline.** Data in, `matching_results.tsv` and `candidate_pairs.tsv` out. See its README for how to run it. |
| `student_resource/er_lab/` | Experiment lab: normalization, blocking, features, matchers, decision rules, embeddings, cross-encoder, France probe |
| `student_resource/LAB_RESULTS.md` | What the experiments showed |
| `eda.py`, `eda_findings_and_report.md` | Exploratory analysis of the data |

The challenge data is not included. The code expects it at `student_resource/dataset/`.

## Approach

1. **Normalization.** NFKC cleanup, a native-script → Latin dictionary learned from training pairs, and canonical legal suffixes, street types and states (US, India, France).
2. **Blocking.** A fine-tuned `multilingual-e5-small` (MIT licence) embeds every record. Exact GPU nearest neighbours are searched within each country: Source 1 → pool top-10, plus pool → Source 1 top-3.
3. **Cheap re-ranker.** LightGBM narrows about 20 candidates to about 6 per Source 1 record. That set is `candidate_pairs.tsv`.
4. **Matcher.** LightGBM on name, address, number and search-context features.
5. **Second stage.** Group-aware stacking over each record's candidates.
6. **Decision.** Tuned rule and threshold, then one-to-one assignment.

## Results

| Version | Held-out F0.5 (training data) | Public leaderboard |
|---|---|---|
| v1 | 0.9796 (training density) | 0.9744 |
| v2 | 0.9826 (test-like density) | 0.9793 |
| v4 | 0.9852 (test-like density) | **0.9811** |

The held-out F0.5 is measured on slice B, folds 3–4.

v4 adds name-uniqueness and fuzzy house-number features. Error analysis showed that 66% of missed matches were pool copies with empty addresses, and that house-number typos broke exact number matching.

v2 adds:
- training density that matches the test pool;
- V2 pair features;
- second-stage cluster features;
- decision-rule selection;
- French canonicalization.
