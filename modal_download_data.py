"""
Modal script to download IMDB dataset and save to scratch-transformers volume.
Run with: modal run modal_download_data.py
"""

import modal

# Create the Modal app and volume
app = modal.App("imdb-data-downloader")
volume = modal.Volume.from_name("scratch-transformers", create_if_missing=True)

# Define the image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests",
    "pandas",
    "tokenizers",
    "fastapi[standard]==0.115.0"
)


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600*10,  # 10 hours
)
def download_and_prepare_imdb(
    num_samples_per_class: int = 25000,  # Use all 50K samples (25K per class)
    vocab_size: int = 8000,
    max_length: int = None,  # Will be computed from data if None
):
    """
    Download IMDB dataset, create train/val/test splits, and train a BPE tokenizer.

    Args:
        num_samples_per_class: Number of positive and negative samples to use
        vocab_size: Vocabulary size for BPE tokenizer
        max_length: Maximum sequence length (computed from data if None)
    """
    import os
    import tarfile
    import requests
    import pandas as pd
    from pathlib import Path
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    import json

    data_dir = Path("/data")
    imdb_dir = data_dir / "aclImdb"

    # Download and extract dataset
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    target_file = data_dir / "aclImdb_v1.tar.gz"

    if not imdb_dir.exists():
        print("Downloading IMDB dataset...")
        response = requests.get(dataset_url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(target_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end="", flush=True)

        print("\nExtracting dataset...")
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall(path=data_dir)

        # Clean up tar file
        target_file.unlink()
        print("Download and extraction complete!")
    else:
        print("IMDB dataset already exists, skipping download.")

    # Load dataset into DataFrame with balanced sampling
    print(f"\nLoading dataset with {num_samples_per_class} samples per class...")

    data_frames = []
    labels = {"pos": 1, "neg": 0}

    for subset in ("train", "test"):
        for label_name, label_value in labels.items():
            path = imdb_dir / subset / label_name
            files = sorted(os.listdir(path))

            for file in files:
                with open(path / file, "r", encoding="utf-8") as infile:
                    text = infile.read()
                    data_frames.append({
                        "text": text,
                        "label": label_value,
                        "subset": subset
                    })

    df = pd.DataFrame(data_frames)
    print(f"Total samples loaded: {len(df)}")

    # Shuffle and balance the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Take equal samples from each class
    pos_samples = df[df["label"] == 1].head(num_samples_per_class)
    neg_samples = df[df["label"] == 0].head(num_samples_per_class)
    df_balanced = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset size: {len(df_balanced)}")
    print(f"  Positive: {len(df_balanced[df_balanced['label'] == 1])}")
    print(f"  Negative: {len(df_balanced[df_balanced['label'] == 0])}")

    # Split into train/val/test (70/15/15)
    total = len(df_balanced)
    train_end = int(0.95 * total)
    val_end = int(0.975 * total)

    train_df = df_balanced.iloc[:train_end]
    val_df = df_balanced.iloc[train_end:val_end]
    test_df = df_balanced.iloc[val_end:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")

    # Train BPE tokenizer on ALL data for better vocabulary coverage
    # This is safe because tokenizer only learns subword statistics, not labels
    print(f"\nTraining BPE tokenizer with vocab size {vocab_size}...")
    print(f"Using all {len(df)} samples for tokenizer training")

    # Save all texts temporarily for tokenizer training
    train_texts_file = data_dir / "train_texts.txt"
    with open(train_texts_file, "w", encoding="utf-8") as f:
        for text in df["text"]:
            f.write(text + "\n")

    # Initialize and train tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train([str(train_texts_file)], trainer)

    # Add post-processing for CLS and SEP tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # Save tokenizer (padding/truncation will be added after computing max_length)
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")

    # Clean up temporary file
    train_texts_file.unlink()

    # Compute max_length from the actual data if not provided
    if max_length is None:
        print("\nComputing max sequence length from data...")
        max_len = 0
        for text in df_balanced["text"]:
            encoded = tokenizer.encode(text)
            if len(encoded.ids) > max_len:
                max_len = len(encoded.ids)
        max_length = max_len
        print(f"Max sequence length in dataset: {max_length}")
    else:
        print(f"Using provided max_length: {max_length}")

    # Now configure padding and truncation with the computed max_length
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"),
        pad_token="[PAD]",
        length=max_length,
    )
    tokenizer.enable_truncation(max_length=max_length)

    # Save tokenizer again with padding/truncation config
    tokenizer.save(str(tokenizer_path))

    # Save datasets
    train_df[["text", "label"]].to_csv(data_dir / "train.csv", index=False)
    val_df[["text", "label"]].to_csv(data_dir / "validation.csv", index=False)
    test_df[["text", "label"]].to_csv(data_dir / "test.csv", index=False)

    print(f"\nDatasets saved to {data_dir}")

    # Save metadata
    metadata = {
        "num_samples_per_class": num_samples_per_class,
        "total_samples": len(df_balanced),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "vocab_size": vocab_size,
        "max_length": max_length,
        "pad_token_id": tokenizer.token_to_id("[PAD]"),
        "cls_token_id": tokenizer.token_to_id("[CLS]"),
        "sep_token_id": tokenizer.token_to_id("[SEP]"),
        "unk_token_id": tokenizer.token_to_id("[UNK]"),
    }

    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {data_dir / 'metadata.json'}")

    # Commit volume changes
    volume.commit()

    return metadata

@app.function(
    image=image,
    volumes={"/data": volume},
)
@modal.asgi_app()
def web():
    """
    FastAPI web app for preparing data.

    Deploy with: modal deploy modal_download_data.py
    """
    import fastapi
    import time

    web_app = fastapi.FastAPI(title="IMDB data processor")

    @web_app.post("/prepare")
    async def prepare_endpoint(
        num_samples: int = 25000,
        vocab_size: int = 8000,
        max_length: int = None,  
    ):
        call = download_and_prepare_imdb.spawn(
            num_samples_per_class=num_samples,
            vocab_size=vocab_size,
            max_length=max_length,
        )
        job_id = call.object_id
        print(f"\n✅ Processing job spawned: {job_id}")
        # Wait briefly to confirm job submission
        time.sleep(5)

        return {
            "status": "success",
            "job_id": job_id,
        }

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return web_app

@app.local_entrypoint()
def main(
    num_samples: int = 25000,  # Use all 50K samples
    vocab_size: int = 8000,
    max_length: int = None,
):
    """
    Entry point for running the data download.

    Args:
        num_samples: Number of samples per class (positive/negative)
        vocab_size: Vocabulary size for BPE tokenizer
        max_length: Maximum sequence length (computed from data if not provided)
    """
    print(f"Starting IMDB data preparation...")
    print(f"  Samples per class: {num_samples}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max length: {'auto (computed from data)' if max_length is None else max_length}")

    metadata = download_and_prepare_imdb.remote(
        num_samples_per_class=num_samples,
        vocab_size=vocab_size,
        max_length=max_length,
    )

    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("=" * 50)
    print(f"\nMetadata: {metadata}")
    print("\nData saved to Modal volume: scratch-transformers")
    print("You can now run the training script.")
