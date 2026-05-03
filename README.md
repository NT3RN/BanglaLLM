# Bangla LLM Dataset Workspace

This workspace contains scripts, notebooks, and prepared datasets for building and validating a Bangla language model pipeline.


## Main Files
- `downloadMC4.py` and `downloadTitulm.py`: dataset download/prep scripts.
- `BanglaLLM_local.ipynb`: local workflow notebook.
- `ValidateBllm.ipynb`: model validation notebook.
- `requirements.txt`: Python dependencies on fedora 43 kde plasma.
- `final_model`: Model Trained on partial dataset. 

## Key Folders
- `BanglaLLM/`: cleaned/final datasets, tokenizer assets, checkpoints, and final model files.
- `bengali_datasets/`: local working copies of source datasets.
- `bengali_datasets_originals/`: original/raw dataset snapshots.

## Quick Start
1. Create a venv and activate the environment
2. Install dependencies:
   ```bash
   pip install transformers==4.51.0 accelerate datasets sentencepiece safetensors pyarrow flask datasketch huggingface_hub tensorboard ninja packaging
   ```
3. Download dataset using the script and organize them 
4. Open `BanglaLLM_local.ipynb` for training/data workflow.
5. Open `ValidateBllm.ipynb` to run validation checks.
6. Run `appGit.py` to get a 51.7M params model with vocab 32,000
