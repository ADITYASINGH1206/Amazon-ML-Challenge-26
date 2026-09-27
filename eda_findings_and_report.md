# EDA findings and experiment plan (26 Sep 2026)

## Data facts (from eda_er.py)
- **Train:** 2.21M S1 records against a 10.3M S2+S3 pool.
- **Test:** 1.73M S1 records (India 810k, US 663k, **France 259k**) against a 10.0M pool.
- **Singletons are only 5.6%.** Non-singletons have a mean of 3.67 matches (max 11). 85% of them have matches in both S2 and S3.
- **One-to-one holds perfectly:** 0 S2/S3 records belong to more than one S1. About 26% of the pool is unmatched distractors.
- **Country agrees on 100% of true pairs**, so blocking within country is free.
- **Indic scripts:** about 45% of India S2/S3 names are in Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam). All of the hardest pairs are these.
- **House numbers are the strongest cheap signal:** shared by 96% (India) / 89% (US) of true pairs, versus 8% / 0.3% of random pairs. Postcodes rarely appear.
- **Noise seen in the data:**
  - typos and word shuffles;
  - legal-suffix swaps;
  - URL forms (vavacenter.com);
  - phone numbers or #ids appended;
  - honorifics (Smt, Dr);
  - completely different DBA names;
  - truncated or empty addresses (~3%);
  - house-number variants (0137, #21, 1287-B, 674-678);
  - state abbreviations or state names in native script.
- **France:**
  - regions and départements (Hauts-de-France / Nord, Nouvelle-Aquitaine / Gironde, Pays de la Loire / Loire-Atlantique);
  - address markers N°, R., RTE, BIS;
  - legal forms SARL, SAS, EURL, SASU, SCI, SA;
  - "(France)" noise in names.
  - Its top-1 TF-IDF similarity distribution looks like US/India.

## Blocking probes (full 10.3M pool, 2k S1 sample, raw normalisation)
- **Name-token inverted index is poor:** recall 0.66 with about 3,000 candidates.
- **TF-IDF char 2–4 on name+address:** R@1 0.25, R@5 0.83, R@10 0.91, R@50 0.95. Name only reaches R@10 of 0.55.
- **Top-1 score** separates singletons partly: median 0.80 for singletons vs 0.93 for entities with matches.

## Plan decisions so far
- Normalise text first: Indic transliteration, a learned native→Latin dictionary, then canonical forms for the US, India and France.
- **Blocking:** forward TF-IDF top-K plus **reverse** top-m (pool → S1), then a cheap LightGBM re-ranker that shrinks the list that goes into `candidate_pairs.tsv`.
- **Matcher:** LightGBM with these feature groups:
  - CTX: blocking context;
  - NAME;
  - ADDR;
  - COMP: comma-component address matching;
  - NUM: house number and postcode;
  - optionally EMB (embeddings) and CE (cross-encoder), which need the GPU (RTX 4060).
- **Decision layer:** one-to-one assignment, then a threshold or expected-F0.5-optimal lists, with optional stage-2 stacking.
- **Evaluation:**
  - universes that are nested subsamples, so results can be extrapolated to full scale;
  - 5-fold CV grouped by S1;
  - thresholds tuned on folds 0–2 and reported on folds 3–4;
  - US↔India transfer as a proxy for France.
- **Experiment lab:** `er_lab.zip`. Run `run_all.py --quick` first, then `--gpu`. Results go to `er_work/SUMMARY.md`.