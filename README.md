# Modal.com Machine Learning Training Scripts

This repository contains a collection of Modal.com scripts for training various machine learning models. Each script is designed to execute on Modal's cloud infrastructure with GPU acceleration support.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Scripts Overview](#scripts-overview)
- [Usage Guide](#usage-guide)
  - [1. IMDB Data Preparation](#1-imdb-data-preparation)
  - [2. BERT Training (From Scratch)](#2-bert-training-from-scratch)
  - [3. Decoder-only Transformer Training](#3-custom-transformer-training)
  - [4. Encoder Decoder Text Summarization Transformer](#4-text-summarization-transformer)
  - [5. Whisper Fine-tuning](#5-whisper-fine-tuning)
- [Web Endpoints](#web-endpoints)
- [Volumes and Data Persistence](#volumes-and-data-persistence)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. **Modal.com Account**: Create an account at [modal.com](https://modal.com)
2. **Modal CLI**: Install the Modal command-line interface
   ```bash
   pip install modal
   ```
3. **Authentication**: Authenticate with Modal using the CLI
   ```bash
   modal token new
   ```
4. **HuggingFace Token** (required for Whisper fine-tuning): Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and configure it as a Modal secret:
   ```bash
   modal secret create huggingface-secret HF_TOKEN=your_token_here
   ```

## Setup

1. Clone this repository or navigate to the project directory
2. Install dependencies locally (optional, for local development and testing):
   ```bash
   pip install -r requirements.txt
   ```

## Scripts Overview

| Script | Purpose | Dataset | Model Type |
|--------|---------|---------|------------|
| `modal_download_data.py` | Download and prepare IMDB dataset | IMDB Reviews | Data preparation |
| `modal_bert_finetune.py` | Train BERT from scratch | IMDB Reviews | BERT (sentiment classification) |
| `decoder_classifier.py` | Train decoder-only transformer with GQA | IMDB Reviews | Custom Transformer (sentiment classification) |
| `modal_summarization_transformer.py` | Train encoder-decoder transformer | CNN/DailyMail | Seq2Seq Transformer (summarization) |
| `whisper_finetune.py` | Fine-tune Whisper for speech recognition | AfriSpeech-200 | Whisper (ASR) |

## Usage Guide

### 1. IMDB Data Preparation

**File**: `modal_download_data.py`

Downloads the IMDB dataset, creates train/val/test splits, and trains a BPE tokenizer.

#### Run Locally:
```bash
modal run modal_download_data.py
```

#### With Custom Parameters:
```bash
modal run modal_download_data.py --num-samples 10000 --vocab-size 8000 --max-length 256
```

#### Parameters:
- `num_samples`: Number of samples per class (positive/negative). Default: 5000
- `vocab_size`: Vocabulary size for BPE tokenizer. Default: 8000
- `max_length`: Maximum sequence length (auto-computed if None). Default: None

#### Output:
- Creates Modal volume: `scratch-transformers`
- Saves: `train.csv`, `validation.csv`, `test.csv`, `tokenizer.json`, `metadata.json`

#### Deploy as Web Endpoint:
```bash
modal deploy modal_download_data.py
```

Then access the endpoint:
- `POST /prepare` - Trigger data preparation
- `GET /health` - Health check

---

### 2. BERT Training (From Scratch)

**File**: `modal_bert_finetune.py`

Trains a small BERT model from random initialization (not pre-trained) for IMDB sentiment classification.

**Prerequisites**: Execute `modal_download_data.py` first to prepare the dataset.

#### Run Locally:
```bash
modal run modal_bert_finetune.py
```

#### With Custom Parameters:
```bash
modal run modal_bert_finetune.py --num-epochs 5 --batch-size 16 --learning-rate 2e-5 --max-length 256
```

#### Parameters:
- `num_epochs`: Number of training epochs. Default: 3
- `batch_size`: Batch size. Default: 16
- `learning_rate`: Learning rate for AdamW. Default: 2e-5
- `max_length`: Maximum sequence length. Default: 256
- `eval_freq`: Evaluate every N steps. Default: 50

#### Model Architecture:
- 4 transformer layers
- 256 hidden dimensions
- 4 attention heads
- ~8M parameters

#### Output:
- Model checkpoint: `best_bert_model.pt`
- Training curves: `bert_training_curves.png`
- Results: `bert_training_results.json`

#### Deploy as Web Endpoint:
```bash
modal deploy modal_bert_finetune.py
```

Endpoints:
- `POST /train` - Start training
- `GET /status` - Get training status and results
- `GET /health` - Health check

---

### 3. Custom Transformer Training

**File**: `decoder_classifier.py`

Trains a custom transformer classifier with Grouped Query Attention (GQA) for memory efficiency.

**Prerequisites**: Execute `modal_download_data.py` first to prepare the dataset.

#### Run Locally:
```bash
modal run decoder_classifier.py
```

#### With Custom Parameters:
```bash
modal run decoder_classifier.py --num-epochs 20 --batch-size 8 --learning-rate 3e-4 --pooling cls
```

#### Parameters:
- `num_epochs`: Number of training epochs. Default: 1
- `batch_size`: Batch size. Default: 8
- `learning_rate`: Learning rate. Default: 3e-4
- `pooling`: Pooling strategy ("cls" or "mean"). Default: "cls"
- `eval_freq`: Evaluate every N steps. Default: 100

#### Model Features:
- Grouped Query Attention (GQA) for memory efficiency
- Custom BPE tokenizer (8K vocab)
- Pre-norm architecture with residual connections

#### Output:
- Model checkpoint: `best_model.pt`
- Results: `training_results.json`

#### Deploy as Web Endpoint:
```bash
modal deploy decoder_classifier.py
```

Endpoints:
- `POST /train` - Start training
- `GET /status` - Get training status
- `GET /health` - Health check

---

### 4. Text Summarization Transformer

**File**: `modal_summarization_transformer.py`

Trains an encoder-decoder transformer from scratch for text summarization on CNN/DailyMail dataset.

#### Step 1: Prepare Dataset
```bash
modal run modal_summarization_transformer.py --task prepare
```

This downloads CNN/DailyMail, trains a tokenizer, and creates train/val/test splits.

#### Step 2: Train Model
```bash
modal run modal_summarization_transformer.py --task train --num-epochs 3
```

#### Parameters:
- `task`: "prepare" or "train". Default: "prepare"
- `num_epochs`: Number of epochs (for training). Default: 3
- `batch_size`: Batch size. Default: 16
- `learning_rate`: Learning rate. Default: 1e-4
- `eval_freq`: Evaluate every N steps. Default: 1000

#### Model Architecture:
- 4-layer encoder + 4-layer decoder
- Grouped Query Attention (GQA)
- ~40M parameters
- Max source length: 512 tokens
- Max target length: 128 tokens

#### Optional: Evaluate with ROUGE
The script includes an evaluation function that computes ROUGE scores. Access it via the web endpoint.

#### Output:
- Model checkpoint: `best_summarizer.pt`
- Training curves: `summarization_loss.png`
- Results: `summarization_results.json`

#### Deploy as Web Endpoint:
```bash
modal deploy modal_summarization_transformer.py
```

Endpoints:
- `POST /prepare` - Prepare dataset
- `POST /train` - Start training
- `POST /evaluate` - Evaluate with ROUGE metrics
- `GET /results` - Get ROUGE evaluation results
- `GET /health` - Health check

---

### 5. Whisper Fine-tuning

**File**: `whisper_finetune.py`

Fine-tunes a Whisper model (distil-whisper/distil-large-v3) for speech recognition on the AfriSpeech-200 dataset, supporting Twi, Akan, and Zulu languages.

**Prerequisites**: 
- HuggingFace access token configured as a Modal secret (see [Prerequisites](#prerequisites))
- Modal secret name must be: `huggingface-secret`

#### Step 1: Download and Preprocess Dataset
```bash
modal run whisper_finetune.py::download_dataset
```

This downloads AfriSpeech-200, extracts audio files, and preprocesses the dataset.

#### Step 2: Train Model
```bash
modal run whisper_finetune.py::train
```

The model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

#### Step 3: Generate Training Curves
```bash
modal run whisper_finetune.py::plot_training_curves
```

#### Step 4: Evaluate on Test Set
```bash
modal run whisper_finetune.py::evaluate_test_set
```

#### Training Configuration:
- Model: `distil-whisper/distil-large-v3`
- Fine-tuning: LoRA (r=16, alpha=32)
- Batch size: 8 (with gradient accumulation: 4)
- Learning rate: 1e-3
- Epochs: 3
- Max audio length: 30 seconds
- Max label length: 448 tokens

#### Output:
- Model checkpoints: `/my_vol/checkpoints/`
- Final model: `/my_vol/final_model/`
- Training metrics: `/my_vol/training_metrics.json`
- Training curves: `/my_vol/training_curves.png`
- Test results: `/my_vol/test_evaluation_results.json`
- Model pushed to HuggingFace Hub: `sirsam01/afrispeech-twi-zulu-akan`

#### Deploy as Web Endpoint:
```bash
modal deploy whisper_finetune.py
```

Endpoints:
- `POST /prepare-dataset` - Download and preprocess dataset
- `POST /train` - Start training
- `POST /plot-curves` - Generate training curves
- `POST /evaluate` - Evaluate on test set

---

## Web Endpoints

All scripts can be deployed as web endpoints to enable remote access via HTTP:

1. **Deploy Application**:
   ```bash
   modal deploy <script_name>.py
   ```

2. **Retrieve Endpoint URL**:
   After deployment, Modal provides a URL in the following format:
   ```
   https://<username>--<app-name>-web.modal.run
   ```

3. **Access Endpoints**:
   - Use any HTTP client such as `curl`, Python `requests`, or similar tools
   - Example request:
     ```bash
     curl -X POST https://<url>/train \
       -H "Content-Type: application/json" \
       -d '{"num_epochs": 5, "batch_size": 16}'
     ```

---

## Volumes and Data Persistence

Modal volumes are used to persist data across execution runs:

| Volume Name | Used By | Contents |
|-------------|---------|----------|
| `scratch-transformers` | IMDB scripts | IMDB dataset, tokenizers, models, results |
| `summarization-data` | Summarization script | CNN/DailyMail dataset, tokenizer, models |
| `afrispeech-train-volume` | Whisper script | AfriSpeech dataset, preprocessed data, checkpoints |

**Note**: Volumes persist data between runs. To reset and start fresh, delete and recreate volumes via the Modal dashboard or CLI.

---

## Troubleshooting

### Common Issues

1. **Error: "No CSV files found"**
   - **Solution**: Execute the data preparation script first (`modal_download_data.py`)

2. **Error: "Volume not found"**
   - **Solution**: Volumes are created automatically upon first use. If issues persist, verify volume status in the Modal dashboard

3. **Error: "Out of memory"**
   - **Solution**: Reduce the `batch_size` parameter or configure a larger GPU type (modify `gpu="A100"` to `gpu="A10G"` or `gpu="L40S"` in the function decorator)

4. **Error: "HuggingFace authentication error"** (Whisper script)
   - **Solution**: Verify that the `huggingface-secret` Modal secret is properly configured with your HF_TOKEN

5. **Error: "Job timeout"**
   - **Solution**: Increase the `timeout` parameter value in the function decorator

### Checking Job Status

1. **Via Modal Dashboard**: Access the applications page at [modal.com/apps](https://modal.com/apps)
2. **Via Command Line Interface**: 
   ```bash
   modal app list
   modal app logs <app-name>
   ```

### Viewing Logs

To view real-time logs for a specific application:
```bash
modal app logs <app-name> --tail
```

---

## File Structure

```
.
├── modal_download_data.py          # IMDB data preparation
├── modal_bert_finetune.py          # BERT training from scratch
├── decoder_classifier.py      # Custom transformer training
├── modal_summarization_transformer.py  # Summarization transformer
├── whisper_finetune.py             # Whisper fine-tuning
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Notes

- All scripts execute on Modal's cloud infrastructure with GPU acceleration
- Training jobs execute asynchronously when deployed as web endpoints
- Models and data are persisted in Modal volumes for subsequent access
- Monitor job status and logs via the Modal dashboard
- GPU types can be modified in function decorators (options include `gpu="A100"`, `gpu="A10G"`, `gpu="T4"`, etc.)

---

