from datasets import load_dataset

# This will download only the Bangla subset of mC4
dataset = load_dataset(
    "allenai/c4",
    "bn",
    cache_dir="./my_local_mc4_data",
    trust_remote_code=True
)

# To verify the download
print(dataset["train"][0])
