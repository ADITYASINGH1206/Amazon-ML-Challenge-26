# Execution Guide: Amazon ML Challenge 2026 Pipeline

This guide provides step-by-step instructions for running the Business Entity Resolution pipeline on a remote GPU instance (e.g., RTX 5090).

## 1. Prerequisites

Ensure the remote desktop has the following installed:
- **Python 3.10+**
- **NVIDIA Drivers & CUDA Toolkit** (Compatible with the GPU)
- **Git** (to clone the repository)

## 2. Setup the Environment

It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Clone the repository (if not already done)
git clone <your-github-repo-url>
cd Amazon-ML-Challenge-26/code/business_entity_resolution

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install the required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Data Directory Configuration

By default, the code expects the data to be in a specific folder structure relative to the project root.

The default expected data path is:
`../../6ab10eb3b23ba_student_resource/student_resource/dataset/`

If your data is located somewhere else on the remote machine, **you do not need to change the code**. Simply set the `STUDENT_RESOURCE_DIR` environment variable before running the script:

```bash
# On Windows (PowerShell):
$env:STUDENT_RESOURCE_DIR="C:\Path\To\student_resource"

# On Linux/macOS (Bash):
export STUDENT_RESOURCE_DIR="/path/to/student_resource"
```
*Note: Make sure the path points to the folder that CONTAINS the `dataset/` directory.*

## 4. Running the Pipeline

The pipeline is fully automated and orchestrated by `run_pipeline.py`. 

To run the entire end-to-end pipeline (Preprocessing → Blocking → Feature Extraction → LightGBM Training → Cross-Encoder Training → Inference):

```bash
python run_pipeline.py all
```

### Running Individual Stages
Since intermediate artifacts (like processed data, features, and models) are saved to disk in the `workdir/` folder, you can run or resume specific stages individually:

```bash
python run_pipeline.py preprocess    # Stage 0: Text normalization
python run_pipeline.py block         # Stage 1: Candidate generation (Inverted Index, TF-IDF, PyTorch Dense)
python run_pipeline.py features      # Stage 2: Feature engineering (Cosine, Jaro-Winkler, etc.)
python run_pipeline.py train_lgbm    # Stage 3a: Fast filter training
python run_pipeline.py train_ce      # Stage 3b: DeBERTa Cross-encoder fine-tuning
python run_pipeline.py validate      # Stage 4a: Threshold optimization on validation set
python run_pipeline.py infer         # Stage 4b: Final inference on Test data
```

## 5. Outputs

Once the pipeline completes, the final evaluation files will be generated in the output directory (default: `student_resource/output/`):

1. **`matching_results.tsv`**: The final scored entity matches to be submitted.
2. **`candidate_pairs.tsv`**: The high-recall blocking candidate pairs (audited in the final ZIP).

## 6. Hardware & Performance Notes

- **VRAM Usage**: The pipeline uses `mdeberta-v3-base` (Cross-Encoder) and `paraphrase-multilingual-MiniLM-L12-v2` (Embeddings), which will easily fit in the RTX 5090's VRAM.
- **RAM Usage**: Blocking (especially the inverted index and sparse TF-IDF matrices on millions of records) requires a significant amount of system RAM (~32GB+ recommended).
- **Time Estimate**: On an RTX 5090 with a fast SSD, expect the full pipeline to complete in roughly 4 to 6 hours.
