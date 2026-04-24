import os
from datasets import load_dataset

# Configuration
SAVE_PATH = "./bengali_datasets"
os.makedirs(SAVE_PATH, exist_ok=True)

def download_and_save():
    # 1. Titulm-Bangla-Corpus (The "Exemplary" 116GB Choice)
    # This is much cleaner than mC4 and already in Parquet
    print("Starting download for Titulm-Bangla-Corpus (~116GB)...")
    titulm = load_dataset("hishab/titulm-bangla-corpus", "common_crawl", split="train")
    titulm.save_to_disk(os.path.join(SAVE_PATH, "titulm_local"))
    print("Titulm saved successfully.")


if __name__ == "__main__":
    download_and_save()
