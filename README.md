# Bangla LLM Dataset Workspace

This workspace contains scripts, notebooks, and prepared datasets for building and validating a Bangla language model pipeline.

## Main Files
- `downloadMC4.py` and `downloadTitulm.py`: dataset download/prep scripts.
- `BanglaLLM_local.ipynb`: local workflow notebook.
- `ValidateBllm.ipynb`: model validation notebook.
- `requirements.txt`: Python dependencies.

## Key Folders
- `BanglaLLM/`: cleaned/final datasets, tokenizer assets, checkpoints, and final model files.
- `bengali_datasets/`: local working copies of source datasets.
- `bengali_datasets_originals/`: original/raw dataset snapshots.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `BanglaLLM_local.ipynb` for training/data workflow.
3. Open `ValidateBllm.ipynb` to run validation checks.