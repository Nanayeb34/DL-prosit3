"""
Modal-compatible transformer training script for IMDB sentiment classification.

Features:
- Grouped Query Attention (GQA) for memory efficiency
- Custom BPE tokenizer with smaller vocab (8K instead of 50K)
- Web endpoint for triggering training
- Structured and clean code organization

Run training: modal run modal_train_transformer.py
Deploy as web endpoint: modal deploy modal_train_transformer.py
"""

import modal
from pathlib import Path
import json

# Modal setup
app = modal.App("imdb-transformer-trainer")
volume = modal.Volume.from_name("scratch-transformers", create_if_missing=True)

# Image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "pandas",
    "numpy",
    "tokenizers",
    "fastapi[standard]==0.115.0"
)

# ============================================================================
# Model Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "vocab_size": 8000,        # Smaller vocab for custom BPE tokenizer
    "block_size": 256,         # Context length (max sequence length)
    "emb_dim": 256,            # Embedding dimension (reduced for smaller dataset)
    "n_heads": 8,              # Number of query heads
    "n_kv_heads": 2,           # Number of key-value heads for GQA (memory efficient)
    "n_layers": 4,             # Number of transformer layers
    "drop_rate": 0.1,          # Dropout rate
    "num_classes": 2,          # Binary classification
}

# ============================================================================
# Model Components
# ============================================================================

def get_model_components():
    """Import and return model components. Must be called within Modal function."""
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    class LayerNorm(nn.Module):
        """Layer normalization with learnable parameters."""

        def __init__(self, emb_dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift

    class FeedForward(nn.Module):
        """Position-wise feed-forward network with GELU activation."""

        def __init__(self, cfg: dict):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                nn.GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
                nn.Dropout(cfg["drop_rate"]),
            )

        def forward(self, x):
            return self.net(x)

    class GroupedQueryAttention(nn.Module):
        """
        Grouped Query Attention (GQA) for memory efficiency.

        Instead of having separate K and V projections for each head,
        GQA shares K and V across groups of query heads.

        Memory savings: (n_heads - n_kv_heads) * 2 * head_dim parameters
        """

        def __init__(self, cfg: dict):
            super().__init__()
            emb_dim = cfg["emb_dim"]
            n_heads = cfg["n_heads"]
            n_kv_heads = cfg["n_kv_heads"]

            assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
            assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.n_rep = n_heads // n_kv_heads  # Number of times to repeat KV
            self.head_dim = emb_dim // n_heads

            # Query projection: full n_heads
            self.W_query = nn.Linear(emb_dim, n_heads * self.head_dim, bias=False)
            # Key and Value projections: reduced to n_kv_heads
            self.W_key = nn.Linear(emb_dim, n_kv_heads * self.head_dim, bias=False)
            self.W_value = nn.Linear(emb_dim, n_kv_heads * self.head_dim, bias=False)

            self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
            self.dropout = nn.Dropout(cfg["drop_rate"])

            # Causal mask
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(cfg["block_size"], cfg["block_size"]), diagonal=1)
            )

        def _repeat_kv(self, x):
            """Repeat key/value heads to match query heads."""
            if self.n_rep == 1:
                return x
            batch, n_kv_heads, seq_len, head_dim = x.shape
            # Expand and reshape to repeat each KV head n_rep times
            x = x.unsqueeze(2).expand(batch, n_kv_heads, self.n_rep, seq_len, head_dim)
            return x.reshape(batch, self.n_heads, seq_len, head_dim)

        def forward(self, x):
            batch, seq_len, _ = x.shape

            # Project queries, keys, values
            queries = self.W_query(x)  # (batch, seq_len, n_heads * head_dim)
            keys = self.W_key(x)       # (batch, seq_len, n_kv_heads * head_dim)
            values = self.W_value(x)   # (batch, seq_len, n_kv_heads * head_dim)

            # Reshape for multi-head attention
            queries = queries.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            keys = keys.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            values = values.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

            # Repeat KV heads to match query heads
            keys = self._repeat_kv(keys)
            values = self._repeat_kv(values)

            # Scaled dot-product attention
            attn_scores = (queries @ keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

            # Apply causal mask
            mask = self.mask[:seq_len, :seq_len].bool()
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Compute output
            context = attn_weights @ values  # (batch, n_heads, seq_len, head_dim)
            context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)

            return self.out_proj(context)

    class TransformerBlock(nn.Module):
        """Transformer block with GQA and feed-forward network."""

        def __init__(self, cfg: dict):
            super().__init__()
            self.ln1 = LayerNorm(cfg["emb_dim"])
            self.attn = GroupedQueryAttention(cfg)
            self.ln2 = LayerNorm(cfg["emb_dim"])
            self.ff = FeedForward(cfg)

        def forward(self, x):
            # Pre-norm architecture with residual connections
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

    class TransformerClassifier(nn.Module):
        """
        Transformer model for sequence classification.

        Uses:
        - Token and positional embeddings
        - GQA-based transformer blocks
        - Classification head on [CLS] token or mean pooling
        """

        def __init__(self, cfg: dict):
            super().__init__()
            self.cfg = cfg

            # Embeddings
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["block_size"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            # Transformer blocks
            self.blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            )

            # Output layers
            self.ln_final = LayerNorm(cfg["emb_dim"])
            self.classifier = nn.Linear(cfg["emb_dim"], cfg["num_classes"])

            # Initialize weights
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, x, pooling: str = "cls"):
            """
            Forward pass.

            Args:
                x: Input token IDs (batch, seq_len)
                pooling: "cls" for [CLS] token, "mean" for mean pooling

            Returns:
                logits: Classification logits (batch, num_classes)
            """
            batch, seq_len = x.shape

            # Token + positional embeddings
            tok_emb = self.tok_emb(x)
            pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))
            x = self.drop_emb(tok_emb + pos_emb)

            # Transformer blocks
            x = self.blocks(x)
            x = self.ln_final(x)

            # Pooling for classification
            if pooling == "cls":
                # Use first token ([CLS])
                pooled = x[:, 0, :]
            else:
                # Mean pooling over sequence
                pooled = x.mean(dim=1)

            return self.classifier(pooled)

        def count_parameters(self):
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return {
        "TransformerClassifier": TransformerClassifier,
        "LayerNorm": LayerNorm,
        "FeedForward": FeedForward,
        "GroupedQueryAttention": GroupedQueryAttention,
        "TransformerBlock": TransformerBlock,
    }

# ============================================================================
# Dataset
# ============================================================================

def get_dataset_class():
    """Return IMDbDataset class. Must be called within Modal function."""
    import torch
    from torch.utils.data import Dataset
    import pandas as pd
    from tokenizers import Tokenizer

    class IMDbDataset(Dataset):
        """Dataset for IMDB sentiment classification using custom tokenizer."""

        def __init__(self, csv_path: str, tokenizer_path: str, max_length: int = 256):
            self.data = pd.read_csv(csv_path)
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.max_length = max_length

            # Pre-tokenize all texts
            self.encoded_texts = []
            for text in self.data["text"]:
                encoding = self.tokenizer.encode(text)
                self.encoded_texts.append(encoding.ids)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            input_ids = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
            label = torch.tensor(self.data.iloc[idx]["label"], dtype=torch.long)
            return input_ids, label

    return IMDbDataset

# ============================================================================
# Training Functions
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L40S",  # or "A10G" for more memory
    timeout=3600*10,  # 10 hours
)
def train_model(
    num_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    pooling: str = "cls",
    eval_freq: int = 100,
    config_overrides: dict = None,
):
    """
    Train the transformer classifier.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: Weight decay for regularization
        pooling: "cls" or "mean" for sequence pooling
        eval_freq: Evaluate every N steps
        config_overrides: Optional dict to override default config

    Returns:
        dict: Training results including final metrics
    """
    import torch
    from torch.utils.data import DataLoader
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use bfloat16 for memory efficiency (better than fp16 for training stability)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using dtype: {dtype}")

    # Load metadata
    data_dir = Path("/data")
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Configure model
    config = DEFAULT_CONFIG.copy()
    config["vocab_size"] = metadata["vocab_size"]
    config["block_size"] = metadata["max_length"]

    if config_overrides:
        config.update(config_overrides)

    print(f"\nModel configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Get model class and instantiate
    components = get_model_components()
    TransformerClassifier = components["TransformerClassifier"]

    model = TransformerClassifier(config).to(device=device, dtype=dtype)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Load datasets
    IMDbDataset = get_dataset_class()
    tokenizer_path = str(data_dir / "tokenizer.json")

    train_dataset = IMDbDataset(
        str(data_dir / "train.csv"),
        tokenizer_path,
        max_length=config["block_size"]
    )
    val_dataset = IMDbDataset(
        str(data_dir / "validation.csv"),
        tokenizer_path,
        max_length=config["block_size"]
    )
    test_dataset = IMDbDataset(
        str(data_dir / "test.csv"),
        tokenizer_path,
        max_length=config["block_size"]
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    global_step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, pooling=pooling)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Evaluation
            if global_step % eval_freq == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device, pooling, dtype)
                train_losses.append(loss.item())
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                print(f"Step {global_step:5d} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2%}")

                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    # Save best model
                    torch.save(model.state_dict(), data_dir / "best_model.pt")

                model.train()

        # End of epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, pooling, dtype)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Avg Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2%}")

    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)

    # Load best model
    model.load_state_dict(torch.load(data_dir / "best_model.pt"))

    train_loss, train_acc = evaluate(model, train_loader, criterion, device, pooling, dtype)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device, pooling, dtype)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, pooling, dtype)

    elapsed_time = time.time() - start_time

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
        "model_parameters": model.count_parameters(),
    }

    print(f"\nTrain Accuracy: {train_acc:.2%}")
    print(f"Val Accuracy: {val_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"\nTraining time: {elapsed_time / 60:.1f} minutes")

    # Save results
    with open(data_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()

    return results


def evaluate(model, data_loader, criterion, device, pooling, dtype=None):
    """Evaluate model on a data loader."""
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids, pooling=pooling)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# ============================================================================
# Web Endpoint (FastAPI with async job spawning)
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
)
@modal.asgi_app()
def web():
    """
    FastAPI web app for triggering training jobs asynchronously.

    Deploy with: modal deploy modal_train_transformer.py
    """
    import fastapi
    import time

    web_app = fastapi.FastAPI(title="IMDB Transformer Trainer")

    @web_app.post("/train")
    async def train_endpoint(
        num_epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        pooling: str = "cls",
    ):
        """
        Trigger model training asynchronously.

        Returns job_id immediately - training runs in background.
        """
        call = train_model.spawn(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pooling=pooling,
        )
        job_id = call.object_id
        print(f"\n✅ Training job spawned: {job_id}")

        # Wait briefly to confirm job submission
        time.sleep(5)

        return {
            "status": "success",
            "job_id": job_id,
            "message": f"Training started with {num_epochs} epochs, batch_size={batch_size}",
        }

    @web_app.get("/status")
    async def get_status():
        """
        Get training status and results if available.
        """
        data_dir = Path("/data")

        response = {
            "data_prepared": False,
            "model_trained": False,
            "metadata": None,
            "results": None,
        }

        # Check for metadata
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                response["metadata"] = json.load(f)
            response["data_prepared"] = True

        # Check for training results
        results_path = data_dir / "training_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                response["results"] = json.load(f)
            response["model_trained"] = True

        return response

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return web_app


# ============================================================================
# Local Entry Point
# ============================================================================

@app.local_entrypoint()
def main(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    pooling: str = "cls",
):
    """
    Local entry point for training.

    Args:
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        pooling: Pooling strategy ("cls" or "mean")
    """
    print("Starting transformer training on Modal...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Pooling: {pooling}")

    results = train_model.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pooling=pooling,
    )

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {results['test_accuracy']:.2%}")
    print(f"  Training Time: {results['training_time_seconds'] / 60:.1f} minutes")
