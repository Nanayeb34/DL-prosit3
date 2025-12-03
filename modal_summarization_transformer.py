"""
Modal script to train an encoder-decoder transformer FROM SCRATCH for text summarization.

Dataset: CNN/DailyMail (950K samples)
Architecture: 4-layer encoder + 4-layer decoder (~40M params)
Training: From random initialization

Run: modal run modal_summarization_transformer.py
Deploy: modal deploy modal_summarization_transformer.py
"""

import modal
from pathlib import Path
import json

# Modal setup
app = modal.App("cnn-dailymail-summarization")
volume = modal.Volume.from_name("summarization-data", create_if_missing=True)

# Image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "datasets",
    "tokenizers",
    "pandas",
    "numpy",
    "matplotlib",
    "rouge-score",
    "fastapi[standard]==0.115.0",
)

# ============================================================================
# Model Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "vocab_size": 8000,
    "max_src_len": 512,        # Max source (article) length
    "max_tgt_len": 128,        # Max target (summary) length
    "d_model": 768,            # Hidden dimension (increased from 512)
    "n_heads": 12,             # Number of query heads (increased from 8)
    "n_kv_heads": 2,           # Number of key-value heads (GQA)
    "n_encoder_layers": 6,     # Encoder layers (increased from 4)
    "n_decoder_layers": 6,     # Decoder layers (increased from 4)
    "d_ff": 3072,              # Feed-forward dimension (increased from 2048)
    "dropout": 0.1,
}

# ============================================================================
# Data Preparation
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200,
)
def prepare_cnn_dailymail(
    vocab_size: int = 8000,
    max_samples: int = None,  # None = use all data
)://
    """
    Download CNN/DailyMail dataset and train tokenizer.

    Args:
        vocab_size: BPE tokenizer vocabulary size
        max_samples: Limit samples for testing (None = use all 950K)
    """
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    import pandas as pd
    import numpy as np

    data_dir = Path("/data")

    print("Downloading CNN/DailyMail dataset...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

    print(f"Dataset sizes:")
    print(f"  Train: {len(dataset['train'])}")
    print(f"  Val: {len(dataset['validation'])}")
    print(f"  Test: {len(dataset['test'])}")

    # Combine all splits and optionally limit samples
    all_data = []
    for split in ["train", "validation", "test"]:
        for item in dataset[split]:
            all_data.append({
                "article": item["article"],
                "summary": item["highlights"],
            })

    if max_samples:
        all_data = all_data[:max_samples]
        print(f"Limited to {len(all_data)} samples for testing")

    print(f"Total samples: {len(all_data)}")

    # Train tokenizer on articles and summaries
    print(f"\nTraining BPE tokenizer with vocab size {vocab_size}...")

    # Save texts for tokenizer training
    texts_file = data_dir / "texts.txt"
    with open(texts_file, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(item["article"] + "\n")
            f.write(item["summary"] + "\n")

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train([str(texts_file)], trainer)

    # Add special token handling
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # Save tokenizer
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")

    # Clean up
    texts_file.unlink()

    # Compute max lengths from data
    print("\nComputing sequence lengths...")
    import numpy as np  # Re-import to ensure availability
    src_lengths = []
    tgt_lengths = []

    for item in all_data[:10000]:  # Sample for speed
        src_enc = tokenizer.encode(item["article"])
        tgt_enc = tokenizer.encode(item["summary"])
        src_lengths.append(len(src_enc.ids))
        tgt_lengths.append(len(tgt_enc.ids))

    max_src_len = int(np.percentile(src_lengths, 95))  # 95th percentile
    max_tgt_len = int(np.percentile(tgt_lengths, 95))

    print(f"Max source length (95th percentile): {max_src_len}")
    print(f"Max target length (95th percentile): {max_tgt_len}")

    # Split data: 95% train, 2.5% val, 2.5% test
    np.random.seed(42)
    np.random.shuffle(all_data)

    total = len(all_data)
    train_end = int(0.95 * total)
    val_end = int(0.975 * total)

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")

    # Save splits
    pd.DataFrame(train_data).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(val_data).to_csv(data_dir / "val.csv", index=False)
    pd.DataFrame(test_data).to_csv(data_dir / "test.csv", index=False)

    # Save metadata
    metadata = {
        "vocab_size": vocab_size,
        "max_src_len": max_src_len,
        "max_tgt_len": max_tgt_len,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "bos_token_id": tokenizer.token_to_id("<s>"),
        "eos_token_id": tokenizer.token_to_id("</s>"),
        "unk_token_id": tokenizer.token_to_id("<unk>"),
    }

    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    volume.commit()

    return metadata


# ============================================================================
# Model Components (Encoder-Decoder Transformer)
# ============================================================================

def get_model_components():
    """Return transformer model components."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer('pe', pe)

        def forward(self, x):
            # x: (batch, seq_len, d_model)
            return x + self.pe[:x.size(1)]

    class GroupedQueryAttention(nn.Module):
        """
        Grouped Query Attention (GQA) for memory efficiency.

        Query heads: n_heads (8)
        Key-Value heads: n_kv_heads (2)
        Each KV head is shared across n_heads // n_kv_heads query heads.
        """

        def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
            super().__init__()
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

            self.d_model = d_model
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.n_rep = n_heads // n_kv_heads  # How many times to repeat KV
            self.head_dim = d_model // n_heads

            # Query projection: full n_heads
            self.q_linear = nn.Linear(d_model, n_heads * self.head_dim)
            # Key and Value projections: reduced to n_kv_heads
            self.k_linear = nn.Linear(d_model, n_kv_heads * self.head_dim)
            self.v_linear = nn.Linear(d_model, n_kv_heads * self.head_dim)

            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def _repeat_kv(self, x):
            """Repeat key/value heads to match query heads."""
            if self.n_rep == 1:
                return x
            batch_size, n_kv_heads, seq_len, head_dim = x.shape
            # Expand and reshape to repeat each KV head n_rep times
            x = x.unsqueeze(2).expand(batch_size, n_kv_heads, self.n_rep, seq_len, head_dim)
            return x.reshape(batch_size, self.n_heads, seq_len, head_dim)

        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)

            # Linear projections
            Q = self.q_linear(query)
            K = self.k_linear(key)
            V = self.v_linear(value)

            # Reshape for multi-head attention
            Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)

            # Repeat KV heads to match query heads
            K = self._repeat_kv(K)
            V = self._repeat_kv(V)

            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            context = torch.matmul(attn, V)

            # Reshape and project
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.out_proj(context)

            return output

    class FeedForward(nn.Module):
        """Position-wise feed-forward network."""

        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.linear2(self.dropout(F.gelu(self.linear1(x))))

    class EncoderLayer(nn.Module):
        """Single encoder layer with GQA."""

        def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.self_attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            # Self-attention with residual
            attn_out = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_out))

            # Feed-forward with residual
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_out))

            return x

    class DecoderLayer(nn.Module):
        """Single decoder layer with GQA cross-attention."""

        def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.self_attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
            self.cross_attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
            # Self-attention with causal mask
            attn_out = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout(attn_out))

            # Cross-attention to encoder output
            cross_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
            x = self.norm2(x + self.dropout(cross_out))

            # Feed-forward
            ff_out = self.feed_forward(x)
            x = self.norm3(x + self.dropout(ff_out))

            return x

    class Seq2SeqTransformer(nn.Module):
        """Encoder-decoder transformer for summarization."""

        def __init__(self, config: dict):
            super().__init__()
            self.config = config

            # Embeddings
            self.src_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
            self.tgt_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
            self.pos_encoding = PositionalEncoding(config["d_model"])

            # Encoder with GQA
            self.encoder_layers = nn.ModuleList([
                EncoderLayer(config["d_model"], config["n_heads"], config["n_kv_heads"],
                           config["d_ff"], config["dropout"])
                for _ in range(config["n_encoder_layers"])
            ])

            # Decoder with GQA
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(config["d_model"], config["n_heads"], config["n_kv_heads"],
                           config["d_ff"], config["dropout"])
                for _ in range(config["n_decoder_layers"])
            ])

            # Output projection
            self.output_proj = nn.Linear(config["d_model"], config["vocab_size"])

            self.dropout = nn.Dropout(config["dropout"])

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def encode(self, src, src_mask=None):
            """Encode source sequence."""
            x = self.src_embedding(src) * math.sqrt(self.config["d_model"])
            x = self.pos_encoding(x)
            x = self.dropout(x)

            for layer in self.encoder_layers:
                x = layer(x, src_mask)

            return x

        def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
            """Decode target sequence."""
            x = self.tgt_embedding(tgt) * math.sqrt(self.config["d_model"])
            x = self.pos_encoding(x)
            x = self.dropout(x)

            for layer in self.decoder_layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)

            return self.output_proj(x)

        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            """Forward pass."""
            encoder_output = self.encode(src, src_mask)
            logits = self.decode(tgt, encoder_output, src_mask, tgt_mask)
            return logits

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return {
        "Seq2SeqTransformer": Seq2SeqTransformer,
        "PositionalEncoding": PositionalEncoding,
        "GroupedQueryAttention": GroupedQueryAttention,
        "EncoderLayer": EncoderLayer,
        "DecoderLayer": DecoderLayer,
    }


# ============================================================================
# Dataset
# ============================================================================

def get_dataset_class():
    """Return dataset class."""
    import torch
    from torch.utils.data import Dataset
    import pandas as pd
    from tokenizers import Tokenizer

    class SummarizationDataset(Dataset):
        def __init__(self, csv_file, tokenizer_path, max_src_len, max_tgt_len):
            self.data = pd.read_csv(csv_file)
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.max_src_len = max_src_len
            self.max_tgt_len = max_tgt_len

            self.pad_id = self.tokenizer.token_to_id("<pad>")
            self.bos_id = self.tokenizer.token_to_id("<s>")
            self.eos_id = self.tokenizer.token_to_id("</s>")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            article = self.data.iloc[idx]["article"]
            summary = self.data.iloc[idx]["summary"]

            # Encode
            src_enc = self.tokenizer.encode(article).ids[:self.max_src_len - 2]
            tgt_enc = self.tokenizer.encode(summary).ids[:self.max_tgt_len - 2]

            # Add BOS/EOS
            src_ids = [self.bos_id] + src_enc + [self.eos_id]
            tgt_ids = [self.bos_id] + tgt_enc + [self.eos_id]

            # Pad
            src_ids += [self.pad_id] * (self.max_src_len - len(src_ids))
            tgt_ids += [self.pad_id] * (self.max_tgt_len - len(tgt_ids))

            return {
                "src": torch.tensor(src_ids[:self.max_src_len], dtype=torch.long),
                "tgt": torch.tensor(tgt_ids[:self.max_tgt_len], dtype=torch.long),
            }

    return SummarizationDataset


# ============================================================================
# Training
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    timeout=3600 * 24,  # 24 hours
)
def train_summarizer(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    warmup_steps: int = 4000,
    eval_freq: int = 1000,
):
    """Train summarization model from scratch with improved hyperparameters."""
    import torch
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import LambdaLR
    import time
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using dtype: {dtype}")

    data_dir = Path("/data")

    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Configure model
    config = DEFAULT_CONFIG.copy()
    config["vocab_size"] = metadata["vocab_size"]
    config["max_src_len"] = metadata["max_src_len"]
    config["max_tgt_len"] = metadata["max_tgt_len"]

    print("\nModel configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Initialize model
    components = get_model_components()
    Seq2SeqTransformer = components["Seq2SeqTransformer"]

    model = Seq2SeqTransformer(config).to(device=device, dtype=dtype)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Load datasets
    SummarizationDataset = get_dataset_class()
    tokenizer_path = str(data_dir / "tokenizer.json")

    train_dataset = SummarizationDataset(
        data_dir / "train.csv", tokenizer_path,
        config["max_src_len"], config["max_tgt_len"]
    )
    val_dataset = SummarizationDataset(
        data_dir / "val.csv", tokenizer_path,
        config["max_src_len"], config["max_tgt_len"]
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * num_epochs

    def lr_lambda(step):
        """Warmup + Cosine decay learning rate schedule."""
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    print(f"\nTraining schedule:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Peak LR: {learning_rate}")
    print(f"  Batch size: {batch_size}")

    # Loss (ignore padding)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=metadata["pad_token_id"])

    # Training tracking
    train_losses = []
    val_losses = []
    steps = []
    best_val_loss = float('inf')
    global_step = 0
    start_time = time.time()

    print("\nStarting training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Create causal mask for decoder
            tgt_len = tgt_input.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).to(device)
            tgt_mask = ~tgt_mask  # Invert for masking

            optimizer.zero_grad()

            logits = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, config["vocab_size"]), tgt_output.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Evaluation
            if global_step % eval_freq == 0:
                val_loss = evaluate(model, val_loader, criterion, device, metadata["pad_token_id"], config)
                train_loss_avg = epoch_loss / (batch_idx + 1)

                train_losses.append(train_loss_avg)
                val_losses.append(val_loss)
                steps.append(global_step)

                print(f"Step {global_step:6d} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), data_dir / "best_summarizer.pt")

                model.train()

        print(f"\nEpoch {epoch + 1}/{num_epochs} complete")

    # Final evaluation
    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)

    model.load_state_dict(torch.load(data_dir / "best_summarizer.pt"))
    final_val_loss = evaluate(model, val_loader, criterion, device, metadata["pad_token_id"], config)

    elapsed_time = time.time() - start_time

    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    print(f"Training time: {elapsed_time / 3600:.1f} hours")

    # Generate plots
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(steps, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(data_dir / "summarization_loss.png", dpi=150)
    plt.close()

    # Save results
    results = {
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "total_steps": global_step,
        "training_time_hours": elapsed_time / 3600,
        "num_parameters": model.count_parameters(),
        "config": config,
        "history": {
            "steps": steps,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
    }

    with open(data_dir / "summarization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()

    return results


def evaluate(model, data_loader, criterion, device, pad_id, config):
    """Evaluate model."""
    import torch

    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_len = tgt_input.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).to(device)
            tgt_mask = ~tgt_mask

            logits = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, config["vocab_size"]), tgt_output.reshape(-1))

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


# ============================================================================
# Evaluation with ROUGE
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",
    timeout=3600,
)
def evaluate_rouge(
    num_samples: int = 1000,  # Limit samples for faster eval
    max_gen_length: int = 128,
):
    """
    Evaluate model on test set using ROUGE metrics.

    Args:
        num_samples: Number of test samples to evaluate (None = all)
        max_gen_length: Maximum generation length

    Returns:
        dict: ROUGE scores and examples
    """
    import torch
    from tokenizers import Tokenizer
    from rouge_score import rouge_scorer
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path("/data")

    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load config
    config = DEFAULT_CONFIG.copy()
    config["vocab_size"] = metadata["vocab_size"]
    config["max_src_len"] = metadata["max_src_len"]
    config["max_tgt_len"] = metadata["max_tgt_len"]

    # Load model
    components = get_model_components()
    Seq2SeqTransformer = components["Seq2SeqTransformer"]

    model = Seq2SeqTransformer(config).to(device)

    # Load best model weights
    model_path = data_dir / "best_summarizer.pt"
    if not model_path.exists():
        return {"error": "No trained model found. Train the model first."}

    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"Model loaded: {model.count_parameters():,} parameters")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    pad_id = metadata["pad_token_id"]
    bos_id = metadata["bos_token_id"]
    eos_id = metadata["eos_token_id"]

    # Load test data
    test_df = pd.read_csv(data_dir / "test.csv")
    if num_samples and num_samples < len(test_df):
        test_df = test_df.head(num_samples)

    print(f"Evaluating on {len(test_df)} test samples...")

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Generate summaries and compute ROUGE
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    examples = []

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            article = row["article"]
            reference = row["summary"]

            # Encode source
            src_enc = tokenizer.encode(article).ids[:config["max_src_len"] - 2]
            src_ids = [bos_id] + src_enc + [eos_id]
            src_ids += [pad_id] * (config["max_src_len"] - len(src_ids))
            src_tensor = torch.tensor([src_ids[:config["max_src_len"]]], dtype=torch.long).to(device)

            # Generate summary (greedy decoding)
            generated = greedy_decode(
                model, src_tensor, bos_id, eos_id, pad_id,
                max_gen_length, device
            )

            # Decode
            hypothesis = tokenizer.decode(generated, skip_special_tokens=True)

            # Compute ROUGE
            scores = scorer.score(reference, hypothesis)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

            # Save examples
            if len(examples) < 5:
                examples.append({
                    "article": article[:200] + "...",
                    "reference": reference,
                    "hypothesis": hypothesis,
                })

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples")

    # Compute average scores
    results = {
        "num_samples": len(test_df),
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
        "examples": examples,
    }

    print(f"\nROUGE Scores:")
    print(f"  ROUGE-1: {results['rouge1']:.4f}")
    print(f"  ROUGE-2: {results['rouge2']:.4f}")
    print(f"  ROUGE-L: {results['rougeL']:.4f}")

    # Save results
    with open(data_dir / "rouge_results.json", "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()

    return results


def greedy_decode(model, src, bos_id, eos_id, pad_id, max_length, device):
    """Greedy decoding for generation."""
    import torch

    # Encode source
    encoder_output = model.encode(src)

    # Start with BOS token
    generated = [bos_id]

    for _ in range(max_length):
        tgt_tensor = torch.tensor([generated], dtype=torch.long).to(device)

        # Create causal mask
        tgt_len = tgt_tensor.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).to(device)
        tgt_mask = ~tgt_mask

        # Decode
        logits = model.decode(tgt_tensor, encoder_output, tgt_mask=tgt_mask)

        # Get next token (greedy)
        next_token = logits[:, -1, :].argmax(dim=-1).item()

        # Stop if EOS
        if next_token == eos_id:
            break

        generated.append(next_token)

    return generated


# ============================================================================
# Web Endpoint
# ============================================================================

@app.function(image=image, volumes={"/data": volume})
@modal.asgi_app()
def web():
    """FastAPI web app."""
    import fastapi
    import time

    web_app = fastapi.FastAPI(title="CNN/DailyMail Summarization Trainer")

    @web_app.post("/prepare")
    async def prepare_data(vocab_size: int = 8000, max_samples: int = None):
        """Prepare dataset. Set max_samples=None to use all 950K samples."""
        call = prepare_cnn_dailymail.spawn(vocab_size=vocab_size, max_samples=max_samples)
        time.sleep(5)
        return {"status": "success", "job_id": call.object_id}

    @web_app.post("/train")
    async def train(
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 5e-4,
        warmup_steps: int = 4000,
    ):
        """Start training with improved hyperparameters."""
        call = train_summarizer.spawn(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
        )
        time.sleep(5)
        return {"status": "success", "job_id": call.object_id}

    @web_app.post("/evaluate")
    async def evaluate(num_samples=None, max_gen_length: int = 128):
        """Evaluate model on test set with ROUGE metrics."""
        call = evaluate_rouge.spawn(num_samples=num_samples, max_gen_length=max_gen_length)
        time.sleep(5)
        return {
            "status": "success",
            "job_id": call.object_id,
            "message": f"Evaluation started on {num_samples} samples",
        }

    @web_app.get("/results")
    async def get_results():
        """Get ROUGE evaluation results."""
        data_dir = Path("/data")
        rouge_path = data_dir / "rouge_results.json"

        if not rouge_path.exists():
            return {"error": "No evaluation results found. Run /evaluate first."}

        with open(rouge_path, "r") as f:
            results = json.load(f)

        return results

    @web_app.get("/health")
    async def health():
        return {"status": "healthy"}

    return web_app


# ============================================================================
# Local Entry Points
# ============================================================================

@app.local_entrypoint()
def main(task: str = "prepare", num_epochs: int = 10):
    """
    Run tasks locally.

    Args:
        task: "prepare" or "train"
        num_epochs: Number of epochs for training (default: 10)
    """
    if task == "prepare":
        print("Preparing CNN/DailyMail dataset (all 950K samples)...")
        metadata = prepare_cnn_dailymail.remote(vocab_size=8000, max_samples=None)
        print(f"\nDataset ready: {metadata}")

    elif task == "train":
        print("Starting summarization training with improved hyperparameters...")
        print(f"  Model: 6-layer encoder + 6-layer decoder (768 hidden, ~100M params)")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: 32")
        print(f"  Learning rate: 5e-4 with warmup")
        results = train_summarizer.remote(num_epochs=num_epochs)
        print(f"\nTraining complete!")
        print(f"  Best val loss: {results['best_val_loss']:.4f}")
        print(f"  Time: {results['training_time_hours']:.1f} hours")
