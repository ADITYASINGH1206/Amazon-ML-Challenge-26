# Final Execution Sequence (0.99 Target Architecture)

You have two options for running the final codebase.

## Option 1: The "Fire and Forget" Method (Recommended)
The pipeline is designed to execute all stages automatically in the correct order. You can start the entire process with a single command.

Run from repository root:
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --gpu
```
*(Or from `code/business_entity_resolution`):*
```bash
cd code/business_entity_resolution
python run_pipeline.py --gpu
```

*Note: The pipeline automatically caches completed stages under `work/`, so if your computer restarts or the process is stopped, running this exact command again safely resumes from where it left off.*

---

## Option 2: Step-by-Step Execution (For Monitoring)
If you prefer to monitor progress stage-by-stage, use the `--stages` flag:

**1. Data Loading & Text Normalization (CPU, ~5 mins)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages load,dict,norm
```

**2. e5-base Fine-Tuning & Embeddings (GPU, ~30–45 mins on RTX 5090)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages finetune,embed --gpu
```

**3. Nearest-Neighbor Blocking & Base Features (GPU + CPU, ~15–20 mins)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages block,feats,feats3,feats4 --gpu
```

**4. mDeBERTa Cross-Encoder Training & Scoring (GPU with AMP, ~1.5–2 hours on RTX 5090)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages ce_train,ce_score --gpu
```

**5. Final Seed Ensembling & Test Predictions (CPU + GPU, ~30 mins)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages train,predict
```

---

## Output Validation
Once the pipeline successfully completes the `predict` stage, the final answers will be located at:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

The script automatically executes `validate_submission.py` at the end of the `predict` stage and verifies format compliance.
