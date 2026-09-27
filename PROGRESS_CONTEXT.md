# Progress context: Amazon ML Challenge 2026, Business Entity Resolution

Last updated: 27 Sep 2026, ~14:55 IST (v4 restored as current; v5 kept as backup). Read this first when resuming.

## Task
- **Goal:** for each Source 1 test record, list the matching Source 2 / Source 3 records.
- **Scoring:** macro F0.5 per Source 1 record. A singleton predicted empty scores 1; any prediction for a singleton scores 0.
- **Submission:** `output/matching_results.tsv` (scored) and `output/candidate_pairs.tsv` (the final candidate set; smaller is ranked better).
- **Final package:** a zip with `output/`, `code/business_entity_resolution/` (README, `requirements.txt`, `src/`) and the filled-in `Documentation_template.md`.
- **Rules:**
  - Models must be MIT/Apache and ≤ 8B parameters.
  - Country is an open set; France appears only in test.
- **User's target:** a leaderboard score of **≥ 0.988**.

## Machine and environment
- **Hardware:** Windows 11, i9-13900HX (32 threads), 32 GB RAM, RTX 4060 Laptop with 8 GB.
- **Power:** keep it plugged in with Legion Performance Mode (GPU limit 125 W). On battery the GPU was capped at 49 W.
- **Python:** 3.12. Key packages: torch 2.5.1+cu121, sentence-transformers 6.1, transformers 5.17, lightgbm 4.7, rapidfuzz, pyarrow, anyascii.
- **Lessons learned:**
  - **Never run two heavy jobs at once.** Every out-of-memory crash today came from parallel runs.
  - Run stages in separate processes (the pipeline does this).
  - Watch RAM with `Get-Counter '\Memory\Available MBytes'`.
- **Paths:** the project root contains the pipeline and data in `student_resource/dataset/` (or auto-detected from environment).
- **Execution:**
  - Running pipeline scripts is standalone and self-contained.
  - Multiprocessing on Windows needs a real `.py` file; scripts fed through stdin hang.

## Data facts
| | Source 1 records | Pool (Source 2 + 3) | Pool per Source 1 |
|---|---|---|---|
| Train | 2.21M | 10.32M | 4.68 |
| Test | 1.73M (France 259k, India 810k, US 663k) | 9.97M | **5.75** |

- **Train structure:**
  - 7.64M true pairs, and 26% of pool records match nothing.
  - 5.6% of Source 1 records are singletons; the others average 3.67 matches.
  - One-to-one holds: no pool record belongs to two Source 1 records.
- **The test pool has about 40% ownerless records**, as if about 19% of Source 1 were removed while their copies stayed.
  - The pipeline reproduces this in training by hiding the Source 1 slice with priority 0.30–0.49 from the search (`config.TRAIN_HIDDEN_S1`).
  - After that, training candidates are 21.2 per record against 20.7 in test.
- **Noise types:** typos, Indic-script names (about 45% of Indian pool names), truncated or empty addresses, house-number variants, legal-suffix swaps, honorifics, completely different DBA names.

## Scores so far
| Version | Held-out estimate (slice B, folds 3–4) | Public leaderboard |
|---|---|---|
| v1: fine-tuned e5 candidate search + LightGBM + second stage | 0.9796 (measured at the easier density) | 0.9744 (rank 828) |
| **v2**: v1 plus the changes listed below | **0.9826** (test-like density) | **0.9793** |
| v3: v2 + has-a-match gate | 0.9826 (no gain) | not submitted (not worth a submission) |
| **v4**: V3 features (`ber/v3.py`: name uniqueness + fuzzy house numbers) + sibling core-name feature in stage 2 | **0.9852** (stage 1: 0.9834) | **0.98111** (gap to held-out 0.0041; France likely gains less) |

**Backups:** `work/models_v2/` (v2 files), `work/models_v3/`. The v4 models are in `work/models/`.

**Error analysis (v2):**
- 66% of rejected true matches have an **empty pool address**; the names match, but the model can't rule out a namesake.
- House-number typos cause another share.
- Missed pairs: 21% not found by the candidate search, 2% removed by the cheap filter, 77% rejected by the matcher.
- The perfect-singleton ceiling is only +0.0025.

v2's changes:
- hidden-Source-1 density matching;
- V2 pair features;
- second-stage embedding-cluster and relative-to-best features;
- decision-rule selection (top-1-plus: best candidate at 0.5, others at 0.75);
- French canonicalization;
- IDF tables without once-seen tokens (so 20 workers fit in RAM).

**Current state of the files:**
- `output/` holds the **v3** files: gated rule, same expected score as v2.
- **The v2 submitted file is backed up** in `student_resource/work/models_v2/matching_results.tsv`, with the v2 models.
- The v1 models are in `work/models_v1/`.

## Where the loss is (v2, held out)
- **Candidate ceiling:** 0.9977 after candidate search, 0.9974 after the cheap filter (recall 0.991).
- **Matcher:** recall **0.962** and precision 0.994. About 0.011 of F0.5 is true matches that were in the candidates but got rejected, and about 0.004 is wrong matches.
- **Singletons** cost only about 0.0015 (2.7% get a wrong match). The has-a-match model has AUC 0.9993 but adds nothing, because stage 2 already uses that information.
- **France**, with no labels, likely scores around 0.96 and costs about 0.003 on the leaderboard.
- A rerun of the error analysis (`src/analyze.py` → `work/tune/analyze.log` and `analyze.json`) was queued behind the v3 prediction. When resuming, check its "error profile" and the example misses.

## Code
- **The pipeline** is in `student_resource/code/business_entity_resolution/src/`.
  - `run_pipeline.py` runs load → dict → norm → finetune → embed → block → feats → train → predict, each stage cached in its own process.
  - Run it from `student_resource/` with `python code/business_entity_resolution/src/run_pipeline.py [--stages ...]`.
  - Main settings: `--feat-workers 20` and `--n-jobs 24`.
  - To retrain, delete `work/models/stage2.lgb` and `decision.json`. Training reuses `cheap.lgb` and `full.lgb` if they exist, so delete those too to refit stage 1.
- **The `ber/` modules:**
  - `config` (slices, feature lists, flags `USE_V2`, `USE_S2X`, `RULES_TRY`);
  - `data`, `prep` (normalization, IDF);
  - `text` (L4 normalization plus the French rules);
  - `embed` (fine-tuning, encoding);
  - `knn` (GPU exact top-k, `block` with `rev_mask`);
  - `feats` (base features plus V2, trigram IDF);
  - `model` (LightGBM, stage-2 features, `stage2_extra`, `add_pj_features`, `entity_features`, the rules including `gated`, `macro_f05`).
- **Helpers:** `tune.py` (experiment harness) and `analyze.py` (loss decomposition, singleton ceiling, error samples, chunked-versus-one-pass check).
- **Training slices** by Source 1 priority (seed 42, the same draw as the lab):

  | Slice | Priority | Used for |
  |---|---|---|
  | A | 0–0.15 | Stage-1 training |
  | B | 0.15–0.30 | Stage 2, thresholds, estimate |
  | Hidden | 0.30–0.49 | Density simulation |
  | Encoder fine-tuning | 0.50–0.53 | Fine-tuning |
  | **Unused** | **0.53–0.60** | Free |
  | Dictionary | ≥ 0.60 | Transliteration dictionary |

- **Work directory** `student_resource/work/`:
  - `cache/` (parquet, normalized text, IDF);
  - `emb/` (fp16 embeddings, about 17 GB);
  - `block/` (candidates);
  - `feats/` (`train.parquet`, and `test_all/` with base features for all 35.8M test pairs, cached);
  - `models/`;
  - `tune/`.
- **Typical runtimes at full power:**
  - encoding: about 78 min for all 24M texts;
  - candidate search: about 20 min each for test and train;
  - training features: about 5 min;
  - training: about 10–15 min;
  - prediction: about 12 min with the cached features.
- **The experiment lab** is `student_resource/er_lab/` (results in `LAB_RESULTS.md`, `GPU_RUN_UPTO06.md`). In the lab, fine-tuned e5 reached R@10 0.9965 against 0.989 for TF-IDF, and the cross-encoder added +0.002 (with a weak model).

## Repository Configuration
- The `.gitignore` excludes data, outputs, caches, models, logs, and temporary files.
- The results table in README reflects latest leaderboard scores.

## v5: wider candidate search (started 27 Sep, 12:27 IST)
- **The user chose the wider search** as the next step.
- **v4 error analysis** (done 12:17, `work/tune/analyze.log`), held-out F0.5 0.9852:
  - The loss is dominated by partial misses (non-singletons with some matches missed), which cost 0.0082.
  - Other losses: extra matches 0.0024, non-singletons predicted empty 0.0023, singleton false merges 0.0012.
  - Missed true pairs are 3.5% of all true pairs: 22.5% not found by the search, 2.5% removed by the cheap filter, 75% rejected by the matcher.
  - 65% of the pairs rejected by the matcher have an **empty pool address**.
  - Perfect singleton detection would reach only 0.9873.
- **Candidate-search probe** (`src/probe_block.py` → `work/tune/probe_block.log`):
  - The current search finds 99.21% of slice B's true pairs; 9,063 are missed.
  - Share of those misses each option recovers:

    | Option | Recovered |
    |---|---|
    | Forward top-15 | 25% |
    | Forward top-20 | 38% |
    | Forward top-30 | 52% |
    | Forward top-50 | 65% |
    | Reverse top-5 | 8% |
    | Reverse top-10 | 19% |
    | Forward top-20 + reverse top-5 + exact core name | 60% |

  - An **exact core-name lookup is useless**: the matching names are common, and capping it to rare names (≤ 10 pool records) recovers only 7%. It was dropped.
- **France is cut off more often:**
  - At forward rank 9, France's mean cheap score is 0.0176, against 0.0051 for the US and 0.0078 for India.
  - 1.1% of France's rank-9 candidates still score ≥ 0.5, against 0.1% for the US.
  - This is the main argument for the wider search. On US/India alone the expected held-out gain is only about +0.0005.
- **The change:** `config.K_FWD` 10 → 20 and `M_REV` 3 → 5.
- **Run:** `python .../run_pipeline.py --stages block,feats,feats3,train,predict --predict-chunk 200000`, log in `student_resource/run_v5.log`, about 1 h 45 min.
- **v4 backups:**
  - `work/models_v4/`: models, decision.json, train_report and both output TSVs;
  - `work/block_v4/`: candidates;
  - `work/feats_v4/`: train.parquet and test_all.
  - To restore v4, move them back and set K_FWD=10, M_REV=3.
- **Submit v5 only if** its held-out estimate is at least v4's 0.9852. The test candidate set should stay about 6 per S1.
- **v5 training result (13:50):**
  - Candidates: 38.6 per S1 in test (66.8M pairs) and 39.4 in train (26.1M pairs).
  - Candidate-search recall on B: 0.9921 → **0.9954**. Perfect-matcher ceiling: 0.9977 → 0.9986.
  - Held-out score: stage 1 0.9835 (v4 0.9834); **stage 2 0.9851 (v4 0.9852), so flat on US/India.** The recovered pairs are the hard cases (empty address, shared names) and the matcher still rejects them.
  - Decision: rule gated(0.3, 0.5, 0.75), t_cheap 0.002.
  - v5 can only gain on France, and only the leaderboard can tell.
- **France check:** `src/diag_country.py [--old work/models_v4/matching_results.tsv]` → `work/tune/diag_country.log`. It reports per-country uncertain-probability mass, the model's self-estimated F0.5, and the share of accepted matches with dissimilar names. Run it after predict.

## Label-free France check (27 Sep, 12:50) and ideas discussed with the user
- **v4 test output by country:**

  | Country | Predicted matches per S1 | Predicted empty | Pool records per S1 | Share of pool assigned |
  |---|---|---|---|---|
  | France | 3.25 | 5.65% | 5.53 | 0.587 |
  | India | 3.33 | 5.81% | 5.82 | 0.573 |
  | US | 3.35 | 5.75% | 5.76 | 0.583 |

  For comparison, the training truth (slice B) is 3.46 matches per S1 with 5.5% singletons.
- **France doesn't look clearly worse:** it assigns the same share of its pool. The lower matches-per-S1 fits its smaller pool per S1. So the leaderboard gap of 0.0041 is **not proven to come from France**; its source is unknown.
- **Ideas ranked for the next rounds:**
  1. **Namesake-evidence features:** for a candidate, count same-core-name pool records whose address agrees with the S1 versus conflicts with it. This targets the empty-address misses. Cheap: 1–2 h.
  2. **Adversarial validation** (simulated train B against test candidates), to find what the training simulation gets wrong and explain the 0.0041 gap. About 30 min.
  3. Then, depending on 2: fix the simulation or adapt to France, and a stronger cross-encoder or encoder.
  4. More training data (cross-fitting A↔B, the unused slice) plus seed ensembles: cheap, about +0.0005–0.001.
- **Predict now saves the test scores** (qi, pj, p) to `work/feats/test_scores.parquet`, for label-free diagnostics. The v5 run picks this up.

## Findings after the user said the top leaderboard score is 0.991 (27 Sep, ~13:00)
- The top team loses 0.009 on the leaderboard against our 0.019, so much of our loss is recoverable.
- **No leak between train and test pools:**
  - Ownerless pool records do NOT match the other split's S1 names more often than owned records do.
  - Name plus house number almost never matches across splits.
- **The generator reuses business names across different businesses.** About 15% of owned US pool records carry the exact name of some other-split S1, with a different house number.
- **Empty-address copies whose core name equals the S1's** make up 2.8% of true pairs. Their share of true pairs depends on how common the name is:

  | Situation | Pairs (slice B) | True share |
  |---|---|---|
  | Name unique among S1, no same-name pool record elsewhere | 16.7k | 98.7% |
  | One same-name pool record elsewhere | 0.9k | 77% |
  | Name shared by 3+ S1 and 5+ other pool records | 227k | 3.9% (about 8.9k true pairs) |

  The last row is genuinely ambiguous for everyone. `GENERIC_NAME` is small (legal forms plus words like "group" and "service"), so the core name is close to the full name.
- **The France hypothesis fits the gap:** France at about 0.96 with US/India at 0.9852 gives 0.85 × 0.9852 + 0.15 × 0.96 ≈ 0.9814, against a leaderboard of 0.9811. The earlier check only covered France's recall, not its precision.
- **Next:**
  - Once v5 is scored, compare France's score distribution and low-name-similarity match rate with US/India using `test_scores.parquet`.
  - Save stage-2 out-of-fold predictions on B, and bucket the misses by type (ambiguous name, name noise, different-name/DBA copy), to see what is recoverable.

## Organiser rule (user, 27 Sep ~14:50): candidate-set size counts in the final ranking
- Blocking must scale to billions of records, so no all-pairs comparison.
- The organisers review `candidate_pairs.tsv` and the code that produces it. **A smaller candidate set per S1 ranks higher**, beyond the public and private leaderboards.
- Saved to memory (`candidate-set-size-counts.md`).
- **What I did (14:55):**
  - Abandoned the wider search: v5 was flat on held-out F0.5 but grew the final set from 6.09 to 7.62 per S1.
  - **Restored v4 as the current state:**
    - `output/` holds the v4 TSVs (checksums verified);
    - `work/models`, `work/block` and `work/feats` hold v4 again;
    - `config` is back to K_FWD 10, M_REV 3.
  - The v5 files are in `work/models_v5/`, `block_v5/` and `feats_v5/` (including `test_scores.parquet`).
- **v4 cheap-filter curve on B** (threshold → candidates per S1, perfect-matcher ceiling):

  | Threshold | Candidates per S1 | Ceiling |
  |---|---|---|
  | 0.0005 | 6.94 | 0.9977 |
  | 0.002 | 6.25 | 0.9976 |
  | **0.005 (used)** | **5.71** (test 6.09) | 0.9974 |
  | 0.01 | 5.23 | 0.9970 |
  | 0.02 | 4.68 | 0.9963 |

- **Measured actual held-out F0.5 by threshold** (`src/size_curve.py` → `work/tune/size_curve.json`; rule top1_plus; stage 2 retrained out of fold on each set):

  | Threshold | Candidates per S1 | F0.5 |
  |---|---|---|
  | 0.005 | 5.71 | 0.98526 |
  | 0.01 | 5.23 | 0.98513 |
  | **0.02** | **4.68** | **0.98503** |
  | 0.05 | 4.07 | 0.98439 |
  | 0.1 | 3.78 | 0.98360 |

  **The knee is 0.02:** 18% fewer candidates for −0.0002 held-out. Use it in v6 (fix t_cheap or add a size penalty to its selection). It may even help France, where v5 showed that more candidates lowered the score.
- **The user reported v5 = 0.9807 public** (v4 = 0.9811), confirming the revert.
- **The public leaderboard uses only part of the test set.** The final ranking uses the rest, plus candidate size and the code review. Choose changes by held-out score and treat the public score as a check.
- **IVF index written** (`ber/knn.py`: `topk_ivf`, `_kmeans`, `_assign`, `_search`):
  - `config.BLOCK_INDEX` ("exact" for now; switch to "ivf" after the test), `IVF_LIST_SIZE` 1024, `IVF_NPROBE` 32.
  - It uses spherical k-means lists on the GPU and list-major scanning, then re-scores exactly in fp32.
  - Test: `src/test_ivf.py` → `work/tune/test_ivf.log` (exact against IVF recall on B, per country).
  - CPU FAISS 1.15.1 is installed. The 7 GFLOP/s measurement was taken while LightGBM held every CPU core, so it isn't representative. faiss-gpu isn't available on Windows.
  - **IVF result (15:18), forward recall of B's true pairs:**

    | | Exact | IVF nprobe 8 | 16 | 32 |
    |---|---|---|---|---|
    | US | 0.9904 | 0.745 | 0.840 | 0.894 (scans 0.53%) |
    | India | 0.9838 | 0.821 | 0.882 | 0.917 (scans 0.79%) |

    Plain IVF loses too many true pairs, because true matches are often not the nearest (cosine ~0.6, rank 3–9). Speed is fine: 18 s against 113 s exact for the US test.
  - **Testing HNSW next** (FAISS `IndexHNSWFlat`, CPU): `src/test_hnsw.py` → `work/tune/test_hnsw.log`.
  - **HNSW on India** (M=32, efC=64): the build takes 331 s on 4.1M records. Forward recall by efSearch:

    | efSearch | Recall | Time per query |
    |---|---|---|
    | 32 | 0.920 | 43 µs |
    | 64 | 0.942 | 73 µs |
    | 128 | 0.955 | 126 µs |
    | 256 | 0.964 | 255 µs |

    Exact search reaches 0.984. HNSW scores visited nodes exactly, so over-fetching doesn't help: the misses are nodes the graph search never reaches.
  - **Wider IVF:**

    | | nprobe 64 | 128 | 256 |
    |---|---|---|---|
    | US forward recall | 0.923 (scans 1.1%) | 0.941 (2.1%) | 0.953 (4.2%, 29 s) |
    | India forward recall | 0.938 (1.6%) | 0.951 (3.2%) | 0.961 (6.3%) |

    Every approximate method levels off 2–4 points below exact on true-pair recall. Those are tail neighbours: near-ties at cosine ~0.6 in dense regions.
  - **Next:** `src/ann_impact.py` (nprobe 64 and 256; t_cheap 0.005 and 0.02) → `work/tune/ann_impact.json`. It measures the **final** held-out F0.5 when only candidates that IVF finds are kept. The exact-search reference is the size curve: 0.98526 at 0.005 and 0.98503 at 0.02.
  - **ann_impact result (16:03)**, final held-out F0.5 (top1_plus):

    | Search | t_cheap | Candidates per S1 | F0.5 | US | India | Union true-pair recall |
    |---|---|---|---|---|---|---|
    | Exact | 0.005 | 5.71 | 0.98526 | 0.98500 | 0.98565 | 0.9921 |
    | Exact | 0.02 | 4.68 | 0.98503 | | | |
    | IVF nprobe 64 | 0.005 | 5.48 | 0.98141 (−0.0039) | | | 0.9804 |
    | IVF nprobe 64 | 0.02 | 4.54 | 0.98116 | | | |
    | IVF nprobe 256 | 0.005 | 5.64 | 0.98438 (−0.0009) | 0.98361 | 0.98554 | 0.9894 |
    | IVF nprobe 256 | 0.02 | 4.64 | 0.98414 | | | |

    - IVF nprobe 256 took 555 s on B alone (forward plus full reverse): **no faster than exact at this scale**, because of the per-list Python loop.
    - The pairs IVF misses are ones the matcher would accept, so the loss is real.
  - **Ideas if the cost is too high:**
    - a second, name-only embedding index, since degraded copies with empty addresses sit far from the full record in the full-text space;
    - token blocking on rare name tokens and on house number plus street;
    - a larger nprobe.
- **Scalability concern:** `ber/knn.py` is an exact brute-force GPU search. It scores every S1 against every pool record of the same country, which is exactly what the organisers call not an option. Replace it with an ANN / inverted-file index (FAISS IVF, MIT licence, or a torch IVF with k-means lists and nprobe), and show that its recall matches exact search. Describe this in the README and documentation.

## v5 result and France diagnosis (27 Sep, 14:40)
- **v5 prediction** is done (47.5 min):
  - 5.77M matches (3.33 per S1); the validator passes.
  - Candidate set: 7.62 per S1 (v4: 6.09), because the wider search made `t_cheap` 0.002.
  - Files are in `output/`, backed up to `work/models_v5/`.
  - The test base features for v5's 66.8M pairs are cached in `work/feats/test_all/`.
  - Leaderboard score: **not yet reported by the user.**
- **`diag_country.py`** (v5 against v4):

  | | France | India | US |
  |---|---|---|---|
  | Uncertain candidates (0.1 ≤ p < 0.9) per S1 | **0.51** | 0.23 | 0.24 |
  | Self-estimated F0.5 | **0.979** | 0.990 | 0.989 |
  | Accepted matches with dissimilar names | 3.6% | 3.0% | 2.7% |
  | Matches changed v4 → v5 (added / removed per 1,000 S1) | +29 / −27 | +12 / −11 | +9 / −10 |

  - The self-estimate runs about 0.004 above the true held-out score on US/India, so France is probably around 0.972–0.975. That is worth about 0.002–0.003 on the leaderboard: **France confirmed as the weak spot.**
- **Uncertain France pairs** (p 0.3–0.7) are mostly true matches at the same address. Two patterns:
  1. **French descriptor-word swaps:** Amicale↔& Fils, Comite↔Collectif, Sportif↔Jeunes, Primaire↔Développement, Energie↔Ecole, Societe↔Federation, Services↔Sportive, Sport↔Section, College↔Gestion, Club dropped, "& Associés", "& Cie", "Frères". `GENERIC_NAME` has only legal forms plus a few US words (group, service, holdings, partners, center), so the model sees a name mismatch. The same kind of swap exists in the US ("Boutique Inc" → "INC-SERVICES").
  2. **Initials:** "Latelier Centre" → "LC", "Vision Ecole" → "VE", "BQ Club" → "BC". No current feature checks this.
  - Other patterns are the same as in the US and India: a random different name at the same address (e.g. "Dovalyrafaye"), zero-padded numbers ("0032"), and "R." / "AV" abbreviations.
- **v6 plan, about 3 h:**
  1. Learn the descriptor-word vocabulary per country from data: swapped tokens in labelled train pairs for US/India, and in confident same-address pairs for France.
  2. Add features: core names equal after removing descriptors, the descriptor-swap count, and an initials match. Compute them only for training plus test survivors, like V3.
  3. Retrain and predict. The test base features are cached, so predict only recomputes the survivor features.
  - Base choice: v5 or v4 blocking, depending on v5's leaderboard score.

## Current status and next steps (as of v4)
- **Current submission:** v4 has **public score 0.98111**; previous scores were 0.9744 (v1) and 0.9793 (v2).
- **The user's target is ≥ 0.988**, so the leaderboard needs +0.007.
  - Only about 70% of held-out gains have carried over to the leaderboard: v2 → v4 gained +0.0026 held-out but +0.0018 public.
  - The held-out-to-leaderboard gap widened from 0.0033 to 0.0041. France is likely the largest leak: generic French names (club, amicale, comité, SARL…) make name uniqueness weaker evidence there.
  - So roughly +0.009–0.010 is needed held-out (0.9852 → ~0.995), against a candidate ceiling of 0.9974. That calls for structural changes.
- **Running now:** the v4 error analysis (`src/analyze.py` → `work/tune/analyze.log` and `analyze.json`, console output in `student_resource/analyze_v4.log`). It uses the v4 models in `work/models/`.
- **Candidates for the next round**, to be ranked by the v4 analysis:
  1. **Wider candidate search.** 21% of v2 misses never reach the candidates. Options: forward top-20 and/or a second name-only search for empty or garbled addresses. This requires re-running the `block` stage for train and test (about 20 min each) and all later stages.
  2. **Collective / cluster voting**, via a pool↔pool nearest-neighbour graph.
  3. **A strong cross-encoder** (mDeBERTa-v3-base / XLM-R-base) as a stage-2 feature. It should help France. About 4–7 h.
  4. **France-specific name uniqueness**, counted on core names without generic words.
  5. **Cross-fitting A↔B**, plus the unused slice (priority 0.53–0.60).
- **Process rules the user cares about:**
  - Run things safely, one heavy job at a time.
  - Back up models and files before retraining (`work/models_vN/`).
  - Test on the toy data (`toy_dataset`, `toy_pipe_work`) first.
  - Push code to GitHub with descriptive commits, and record scores in the README.
  - Keep this file updated.

## Earlier plan (from before v4; still relevant)
The user wants large gains, even if they take a long time. They consider the cross-encoder's +0.002 too small as a standalone step. The proposed direction:
1. **Error analysis** of the held-out misses (queued), to decide the weights of steps 2 and 3.
2. **A strong cross-encoder:**
   - backbone: mDeBERTa-v3-base or XLM-R-base (MIT), or a larger e5;
   - training data: about 3M pairs from every slice except B and Hidden;
   - 2+ epochs, on raw and normalized text;
   - used as a stage-2 feature, with stage 2 trained on B;
   - cost: about 4–7 h of GPU time; scoring runs over the B survivors (1.9M) and the test survivors (10.5M).
3. **Collective matching:** a pool↔pool nearest-neighbour graph, with cluster votes (does this candidate's cluster belong to this Source 1 record?) as stage-2 features. This targets degraded copies such as `Usha 6roup` with no address.
4. **Cross-fitting A↔B**, so stage 2 gets twice the data, and use the unused 0.53–0.60 slice.
5. Adapt to France: pseudo-labels, plus the cross-encoder carrying over to French text.

A realistic combined gain is about +0.004–0.009 (0.984–0.988), with no guarantee.
