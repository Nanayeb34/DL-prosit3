"""
Modal script to train a BERT model FROM SCRATCH for IMDB sentiment classification.

Features:
- Small BERT architecture trained from random initialization
- 95% train / 2.5% val / 2.5% test split
- Training/validation loss and accuracy logging
- Matplotlib plots saved after training

Run: modal run modal_bert_finetune.py
Deploy: modal deploy modal_bert_finetune.py
"""

import modal
from pathlib import Path
import json

# Modal setup
app = modal.App("imdb-bert-trainer")
volume = modal.Volume.from_name("scratch-transformers", create_if_missing=True)

# Image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "pandas",
    "numpy",
    "matplotlib",
    "fastapi[standard]==0.115.0",
)

# ============================================================================
# Dataset
# ============================================================================
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

def get_dataset_class():
    """Return IMDbDataset class for BERT."""
    import torch
    from torch.utils.data import Dataset
    import pandas as pd

    class IMDbDataset(Dataset):
        """Dataset for IMDB sentiment classification with BERT tokenizer."""

        def __init__(self, csv_file, tokenizer, max_length=512):
            self.data = pd.read_csv(csv_file)
            self.tokenizer = tokenizer
            self.max_length = max_length

            # Pre-tokenize all texts
            self.encodings = []
            for text in self.data["text"]:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                self.encodings.append({
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                })

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings[idx]["input_ids"],
                "attention_mask": self.encodings[idx]["attention_mask"],
                "labels": torch.tensor(self.data.iloc[idx]["label"], dtype=torch.long),
            }

    return IMDbDataset


# ============================================================================
# Training Function
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=3600 * 10,
)
def train_bert(
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    eval_freq: int = 50,
    train_ratio: float = 0.95,
):
    """
    Fine-tune DistilBERT for sentiment classification.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for AdamW
        max_length: Maximum sequence length
        eval_freq: Evaluate every N steps
        train_ratio: Ratio of data for training (rest split between val/test)
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
    import pandas as pd
    import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use bfloat16 if available
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using dtype: {dtype}")

    data_dir = Path("/data")

    # Load and combine all data for re-splitting
    print("\nLoading datasets...")
    dfs = []
    for split in ["train.csv", "validation.csv", "test.csv"]:
        csv_path = data_dir / split
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))

    if not dfs:
        raise FileNotFoundError("No CSV files found in /data. Run data preparation first.")

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total samples: {len(full_df)}")

    # Split: 95% train, 2.5% val, 2.5% test
    total = len(full_df)
    train_end = int(train_ratio * total)
    val_end = train_end + int((1 - train_ratio) / 2 * total)

    train_df = full_df.iloc[:train_end]
    val_df = full_df.iloc[train_end:val_end]
    test_df = full_df.iloc[val_end:]

    # Save temporary splits
    train_df.to_csv(data_dir / "bert_train.csv", index=False)
    val_df.to_csv(data_dir / "bert_val.csv", index=False)
    test_df.to_csv(data_dir / "bert_test.csv", index=False)

    print(f"Split sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
    print(f"  Val: {len(val_df)} ({len(val_df)/total*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/total*100:.1f}%)")

    # Use BERT tokenizer (just for vocab, not pre-trained weights)
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create small BERT config for training FROM SCRATCH
    # Smaller architecture suitable for 10K samples
    print("\nInitializing BERT model FROM SCRATCH (random weights)...")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,           # Small hidden size (BERT-base uses 768)
        num_hidden_layers=4,       # 4 layers (BERT-base uses 12)
        num_attention_heads=4,     # 4 heads (BERT-base uses 12)
        intermediate_size=1024,    # FFN size (BERT-base uses 3072)
        max_position_embeddings=max_length,
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    # Initialize model with RANDOM weights (not pre-trained)
    model = BertForSequenceClassification(config)
    model.to(device=device, dtype=dtype)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Architecture: {config.num_hidden_layers} layers, {config.hidden_size} hidden, {config.num_attention_heads} heads")

    # Create datasets
    IMDbDataset = get_dataset_class()
    train_dataset = IMDbDataset(data_dir / "bert_train.csv", tokenizer, max_length)
    val_dataset = IMDbDataset(data_dir / "bert_val.csv", tokenizer, max_length)
    test_dataset = IMDbDataset(data_dir / "bert_test.csv", tokenizer, max_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        fused = True
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )

    # Training tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    steps = []

    best_val_accuracy = 0.0
    global_step = 0
    start_time = time.time()

    print("\nStarting training...")
    print(f"Total steps: {total_steps}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Evaluation
            if global_step % eval_freq == 0:
                val_loss, val_acc = evaluate(model, val_loader, device)
                train_loss_avg = epoch_loss / (batch_idx + 1)

                # Calculate train accuracy on subset
                train_loss_eval, train_acc = evaluate(model, train_loader, device, num_batches=20)

                train_losses.append(train_loss_avg)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                steps.append(global_step)

                print(f"Step {global_step:5d} | "
                      f"Train Loss: {train_loss_avg:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.2%} | "
                      f"Val Acc: {val_acc:.2%}")

                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    torch.save(model.state_dict(), data_dir / "best_bert_model.pt")

                model.train()

        # End of epoch
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch + 1}/{num_epochs} complete")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2%}")

    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)

    # Load best model
    model.load_state_dict(torch.load(data_dir / "best_bert_model.pt"))

    train_loss, train_acc = evaluate(model, train_loader, device)
    val_loss, val_acc = evaluate(model, val_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)

    elapsed_time = time.time() - start_time

    print(f"\nTrain Accuracy: {train_acc:.2%}")
    print(f"Val Accuracy: {val_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"\nTraining time: {elapsed_time / 60:.1f} minutes")

    # Generate plots
    print("\nGenerating training plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(steps, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(steps, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Steps', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(steps, [a * 100 for a in train_accs], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(steps, [a * 100 for a in val_accs], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Steps', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = data_dir / "bert_training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to: {plot_path}")

    # Save results
    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_accuracy,
        "total_steps": global_step,
        "training_time_seconds": elapsed_time,
        "model_type": "BERT from scratch",
        "architecture": {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
        },
        "num_parameters": num_params,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "history": {
            "steps": steps,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
        }
    }

    with open(data_dir / "bert_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()

    return results


def evaluate(model, data_loader, device, num_batches=None):
    """Evaluate model on a data loader."""
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item() * labels.size(0)
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


# ============================================================================
# Web Endpoint
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
)
@modal.asgi_app()
def web():
    """FastAPI web app for BERT training."""
    import fastapi
    import time

    web_app = fastapi.FastAPI(title="IMDB BERT Trainer")

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

    @web_app.post("/train")
    async def train_endpoint(
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        max_length: int = 2048,
    ):
        """Trigger BERT training from scratch."""
        call = train_bert.spawn(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
        )
        job_id = call.object_id
        print(f"\n Training job spawned: {job_id}")
        time.sleep(5)

        return {
            "status": "success",
            "job_id": job_id,
            "message": f"BERT training started: {num_epochs} epochs, batch_size={batch_size}",
        }

    @web_app.get("/status")
    async def get_status():
        """Get training status and results."""
        data_dir = Path("/data")

        response = {
            "model_trained": False,
            "results": None,
            "plot_available": False,
        }

        results_path = data_dir / "bert_training_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                response["results"] = json.load(f)
            response["model_trained"] = True

        plot_path = data_dir / "bert_training_curves.png"
        response["plot_available"] = plot_path.exists()

        return response

    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return web_app


# ============================================================================
# Local Entry Point
# ============================================================================

@app.local_entrypoint()
def main(
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
):
    """Run BERT training from scratch."""
    print("Starting BERT training FROM SCRATCH on Modal...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max length: {max_length}")
    print(f"  Architecture: 4 layers, 256 hidden, 4 heads (~8M params)")

    results = train_bert.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
    )

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {results['test_accuracy']:.2%}")
    print(f"  Training Time: {results['training_time_seconds'] / 60:.1f} minutes")
    print(f"\nPlots saved to Modal volume: bert_training_curves.png")
