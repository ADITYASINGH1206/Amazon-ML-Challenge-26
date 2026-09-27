# Final Execution Sequence

You have two options for running the final codebase. 

## Option 1: The "Fire and Forget" Method (Recommended)
The pipeline is designed to execute all stages automatically in the correct order. You can start the entire process with a single command. 

Run this from your root folder (`e:\My Projects\Amazon_ML\amazonMLchlng-main`):
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --gpu
```
*Note: The pipeline automatically caches completed stages to disk, so if your computer restarts or the process fails, running this exact command again will safely resume from where it left off.*

---

## Option 2: Step-by-Step Execution (For Monitoring)
If you prefer to monitor the bottlenecks and execute the massive GPU/CPU tasks in chunks according to the blueprint, you can use the `--stages` flag.

Run these commands sequentially from the root directory:

**1. Data Loading & Text Normalization (CPU-bound, ~5 mins)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages load,dict,norm
```

**2. e5-base Fine-Tuning & Embeddings (GPU-heavy, ~3-4 hours)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages finetune,embed --gpu
```

**3. Nearest-Neighbor Blocking & Base Features (GPU + CPU, ~20 mins)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages block,feats,feats3,feats4 --gpu
```

**4. mDeBERTa Cross-Encoder Training & Scoring (GPU-heavy, ~5-6 hours)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages ce_train,ce_score --gpu
```

**5. Final Seed Ensembling & Test Predictions (CPU-heavy, ~1 hour)**
```bash
python student_resource/code/business_entity_resolution/src/run_pipeline.py --stages train,predict
```

---

## Output Validation
Once the pipeline successfully completes the `predict` stage, the final answers will be located at:
`student_resource/output/matching_results.tsv`

The script automatically runs the validator at the end of the `predict` stage and will print the `validator exit code 0` to confirm that the file format is perfectly matched to the competition's submission requirements.
