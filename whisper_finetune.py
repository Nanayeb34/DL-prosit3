import modal
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

app = modal.App("whisper-preprocess")

# Create Modal image with all required packages
image = (modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
.apt_install("ffmpeg", "libsndfile1")  # FFmpeg for audio processing
.pip_install("fastapi[standard]==0.115.0","requests","inflect","pandas","matplotlib")
.pip_install_from_requirements("requirements.txt")
)

# Persistent volume to store dataset and checkpoints
VOLUME_NAME = "afrispeech-train-volume"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Whisper max audio length
MAX_DURATION_IN_SECONDS = 30.0
MAX_INPUT_LENGTH = MAX_DURATION_IN_SECONDS * 16000
MAX_LABEL_LENGTH = 448

def _preprocess_text(text):
    import re
    import inflect
    """
    Convert numbers in text to words using the inflect library.
    Handles:
    - Regular numbers (e.g., "123" -> "one hundred and twenty-three")
    - Ordinals (e.g., "1st" -> "first", "2nd" -> "second")
    - Decimals (e.g., "3.14" -> "three point one four")
    """
    p = inflect.engine()

    # Handle ordinals first (1st, 2nd, 3rd, etc.)
    def replace_ordinal(match):
        num_str = match.group(0)
        try:
            num = int(re.sub(r'(st|nd|rd|th)', '', num_str))
            return p.number_to_words(p.ordinal(num))
        except:
            return num_str

    text = re.sub(r'\b\d+(st|nd|rd|th)\b', replace_ordinal, text, flags=re.IGNORECASE)

    # Handle decimals (e.g., 3.14 -> "three point one four")
    def replace_decimal(match):
        parts = match.group(0).split('.')
        try:
            integer_part = p.number_to_words(int(parts[0]))
            decimal_part = " ".join([p.number_to_words(int(d)) for d in parts[1]])
            return f"{integer_part} point {decimal_part}"
        except:
            return match.group(0)

    text = re.sub(r'\b\d+\.\d+\b', replace_decimal, text)

    # Handle regular numbers (including years)
    def replace_number(match):
        num_str = match.group(0)
        try:
            num = int(num_str)
            return p.number_to_words(num)
        except:
            return num_str

    text = re.sub(r'\b\d+\b', replace_number, text)

    return text


# Preprocess text: normalize and filter
def _normalize_text(batch):
    """Lowercase, remove punctuation, convert numbers to words, and strip extra spaces."""
    import re
    import string

    text = batch["transcript"]

    # Convert numbers to words FIRST (before lowercasing)
    text = _preprocess_text(text)

    # Then lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r"\s+", " ", text).strip()

    batch["transcript"] = text
    return batch

def prepare_dataset(batch, feature_extractor, tokenizer):
  # load and resample audio data from 48 to 16kHz
  audio = batch["audio"]

  # compute log-Mel input features from input audio array
  batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

  # encode target text to label ids (use "transcript" not "transcription")
  batch["labels"] = tokenizer(batch["transcript"]).input_ids
  batch["input_length"] = len(audio["array"])
  batch["labels_length"] = len(tokenizer(batch["transcript"], add_special_tokens=False).input_ids)
  return batch

def filter_inputs(input_length):
    """Filter inputs with zero input length or longer than 30s"""
    return 0 < input_length < MAX_INPUT_LENGTH

def filter_labels(labels_length):
    """Filter label sequences longer than max length (448)"""
    return labels_length < MAX_LABEL_LENGTH

def compute_metrics(pred, tokenizer, metric):
    """
    Compute WER metric for generated predictions.
    With predict_with_generate=True, predictions are already token IDs from .generate()
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id in labels
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    # pred_ids are already token IDs from .generate(), not logits
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Convert input_features to fp16 to match model dtype
        batch["input_features"] = batch["input_features"].to(torch.float16)

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save( self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs,):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

@app.function(
    image=image,
    volumes={"/my_vol": volume},
    timeout=3600 * 5,  # 5 hours timeout for large files
)
def download_dataset():
    from huggingface_hub import snapshot_download
    import tarfile
    import pandas as pd
    from transformers import WhisperFeatureExtractor
    from transformers import WhisperTokenizer
    from transformers import WhisperProcessor
    from datasets import Audio

    output_path = Path("/my_vol") / 'afrispeech200data'
    os.makedirs(output_path,exist_ok=True)

    snapshot_download(
    repo_id="intronhealth/afrispeech-200",
    repo_type="dataset",
    allow_patterns=["audio/twi*", "transcripts/twi*","audio/akan*", "transcripts/akan*", "audio/akan(fante)*", "transcripts/akan(fante)*",
                    "audio/zulu*", "transcripts/zulu*",'accents.json','accent_stats.py', 'afrispeech-200.py'],
    local_dir=output_path
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained("distil-whisper/distil-large-v3", do_normalize=True)
    tokenizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3", task="transcribe")
    processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3", task="transcribe")
    
    
    splits = ['train', 'dev', 'test']
    languages = ['twi', 'akan', 'akan-fante', 'zulu']
    audio_root = Path(output_path) / "audio"

    for lang in languages:
        for split in splits:
            tar_path = f"{audio_root}/{lang}/{split}/{split}_{lang.replace('(', '_').replace(')', '')}_0.tar.gz"
            extract_dir = f"{audio_root}/{lang}/{split}"
            os.makedirs(extract_dir, exist_ok=True)
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=extract_dir)
                print(f"Extracted {tar_path} to {extract_dir}")
            except FileNotFoundError:
                print(f"Warning: {tar_path} not found. Skipping extraction for this split/language.")
            except Exception as e:
                print(f"Error extracting {tar_path}: {e}")

    # Load transcripts for all languages
    csv_root =  Path(output_path) / "transcripts"
    dfs = {split: [] for split in splits}

    for lang in languages:
        for split in splits:
            csv_path = f"{csv_root}/{lang}/{split}.csv"
            try:
                df = pd.read_csv(csv_path)
                dfs[split].append(df)
            except FileNotFoundError:
                print(f"Warning: {csv_path} not found. Skipping.")

    # Concatenate dataframes for each split
    for split in splits:
        if dfs[split]:
            dfs[split] = pd.concat(dfs[split], ignore_index=True)
        else:
            dfs[split] = pd.DataFrame() 

    # Update audio paths
    for split_name in splits:
        dfs[split_name]['audio_path'] = dfs[split_name].apply(
            lambda row: f"{audio_root}/{row['accent'].replace('akan (fante)', 'akan-fante')}/{split_name}/data/data/intron/{'/'.join(row['audio_paths'].split('/')[-2:])}",
            axis=1 # Apply the lambda function row-wise
        )
        dfs[split_name] = dfs[split_name].drop(columns=['audio_paths'])

    for split_name in splits:
        print(split_name)
        print(dfs[split_name]['audio_path'].iloc[0])

    for split in splits:
        dfs[split] = dfs[split].apply(_normalize_text, axis=1)

    print(f"Train samples: {len(dfs['train'])}")
    print(f"Dev samples: {len(dfs['dev'])}")
    print(f"Test samples: {len(dfs['test'])}")
    print(f"\nSample transcript: {dfs['train'].iloc[0]['transcript']}")

    # Transform dataframes into HuggingFace datasets
    from datasets import Dataset, DatasetDict

    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(dfs['train']),
        'dev': Dataset.from_pandas(dfs['dev']),
        'test': Dataset.from_pandas(dfs['test'])
    })

    # Cast audio column to Audio type with 16kHz sampling
    dataset_dict = dataset_dict.cast_column("audio_path", Audio(sampling_rate=16000))
    dataset_dict = dataset_dict.rename_column("audio_path", "audio")

    # Apply preprocessing with feature extractor and tokenizer
    # Keep only the columns we need for training
    columns_to_remove = [col for col in dataset_dict['train'].column_names if col not in ['audio', 'transcript']]
    dataset_cln = dataset_dict.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=columns_to_remove,
        num_proc=1
    )

    # Filter by length constraints
    dataset_cln = dataset_cln.filter(filter_inputs, input_columns=["input_length"])
    dataset_cln = dataset_cln.filter(filter_labels, input_columns=["labels_length"])
    dataset_cln = dataset_cln.remove_columns(['labels_length', 'input_length'])

    # Save preprocessed dataset to volume
    save_path = Path("/my_vol") / "preprocessed_dataset"
    dataset_cln.save_to_disk(str(save_path))

    # Commit changes to the volume
    from modal import Volume
    vol = Volume.from_name(VOLUME_NAME)
    vol.commit()

    print(f"\n✅ Preprocessed dataset saved to {save_path}")
    print(f"Train: {len(dataset_cln['train'])} samples")
    print(f"Dev: {len(dataset_cln['dev'])} samples")
    print(f"Test: {len(dataset_cln['test'])} samples")

    return str(save_path)

@app.function(
    image=image,
    volumes={"/my_vol": volume},
    gpu="L40S",  
    timeout=3600 * 10,  # 10 hours for training
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    import pandas as pd
    from transformers import WhisperForConditionalGeneration
    from peft import prepare_model_for_kbit_training
    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
    from datasets import load_from_disk
    from transformers import Seq2SeqTrainingArguments
    from transformers import WhisperFeatureExtractor
    from transformers import WhisperTokenizer
    from transformers import WhisperProcessor
    import torch
    import evaluate
    from transformers import Seq2SeqTrainer
    from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    dataset = load_from_disk("/my_vol/preprocessed_dataset")
    metric = evaluate.load("wer")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("distil-whisper/distil-large-v3", do_normalize=True)
    tokenizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3", task="transcribe")
    processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3", task="transcribe")
    
    # Load model in full precision (fp16 for memory efficiency)
    import torch
    model = WhisperForConditionalGeneration.from_pretrained(
        "distil-whisper/distil-large-v3",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Must be False for gradient checkpointing

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Configure LoRA
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    # Apply LoRA to the model
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir="/my_vol/checkpoints",
        per_device_train_batch_size=8,  # Reduced to fit without gradient checkpointing
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size of 16
        learning_rate=1e-3,
        warmup_steps=100,
        num_train_epochs=3,  # Set number of epochs (default is 3)
        # max_steps=1000,  # Use either num_train_epochs OR max_steps, not both
        gradient_checkpointing=False,  # Disabled to avoid gradient issues with LoRA
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=4,  # Increased from 1 for faster evaluation
        predict_with_generate=True,  # Use .generate() during evaluation for accurate WER
        generation_max_length=128,
        save_steps=200,
        eval_steps=50,  # Evaluate at same frequency as saving
        logging_steps=50,
        report_to=["tensorboard"],
        logging_dir="/my_vol/logs",  # Explicit logging directory
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        optim='adamw_torch_fused',
        push_to_hub=True,
        hub_token=os.environ.get("HF_TOKEN"),  # Get from environment
        hub_model_id="sirsam01/afrispeech-twi-zulu-akan",  # Repository to push to
        hub_strategy="checkpoint",  # Push checkpoints during training, not just at end
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, metric),
        callbacks=[SavePeftModelCallback()],
        processing_class=processor.feature_extractor  # Updated from 'tokenizer' to avoid deprecation
    )
    processor.save_pretrained(training_args.output_dir)

    # Train the model
    trainer.train()

    # Save final model locally
    final_model_path = "/my_vol/final_model"
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)

    # Save training metrics and history
    import json
    metrics_path = "/my_vol/training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"✅ Training complete!")
    print(f"  - Model saved to volume: {final_model_path}")
    print(f"  - Metrics saved to: {metrics_path}")
    print(f"  - TensorBoard logs: /my_vol/logs")
    print(f"  - Model pushed to HuggingFace Hub")

    # Commit volume changes to persist everything
    from modal import Volume
    vol = Volume.from_name(VOLUME_NAME)
    vol.commit()

    return {
        "status": "success",
        "model_id": "sirsam01/afrispeech-twi-zulu-akan",
        "final_model_path": final_model_path,
        "metrics_path": metrics_path,
        "logs_path": "/my_vol/logs"
    }

@app.function(
    image=image,
    volumes={"/my_vol": volume},
    gpu="L40S",
    timeout=3600 * 10,  # 10 hours for training
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def resume_training(additional_epochs=3, new_model_name="afrispeech-twi-zulu-akan-extended"):
    """Resume training from the last checkpoint for additional epochs"""
    import pandas as pd
    from transformers import WhisperForConditionalGeneration
    from peft import PeftModel, LoraConfig
    from datasets import load_from_disk
    from transformers import Seq2SeqTrainingArguments
    from transformers import WhisperFeatureExtractor
    from transformers import WhisperTokenizer
    from transformers import WhisperProcessor
    import torch
    import evaluate
    from transformers import Seq2SeqTrainer
    import os
    import glob

    print("Loading preprocessed dataset...")
    dataset = load_from_disk("/my_vol/preprocessed_dataset")
    metric = evaluate.load("wer")

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3", task="transcribe")

    print("Loading base model...")
    # Load the base model
    model = WhisperForConditionalGeneration.from_pretrained(
        "distil-whisper/distil-large-v3",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Loading PEFT adapter from checkpoint...")
    # Load the trained PEFT adapter from the final model
    model = PeftModel.from_pretrained(model, "/my_vol/final_model")

    # IMPORTANT: Re-enable training for the adapter parameters
    # When loading a PEFT model, parameters aren't automatically set to trainable
    for param in model.parameters():
        param.requires_grad = False  # First freeze everything

    # Then unfreeze only the LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    # Configure model
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    print("Model loaded successfully!")
    model.print_trainable_parameters()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # New output directory and hub model ID
    new_output_dir = f"/my_vol/checkpoints_{new_model_name}"
    new_hub_id = f"sirsam01/{new_model_name}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=new_output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,  # Slightly lower LR for continued training
        warmup_steps=50,  # Fewer warmup steps since already trained
        num_train_epochs=additional_epochs,
        gradient_checkpointing=False,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=128,
        save_steps=200,
        eval_steps=50,
        logging_steps=50,
        report_to=["tensorboard"],
        logging_dir=f"/my_vol/logs_{new_model_name}",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        optim='adamw_torch_fused',
        push_to_hub=True,
        hub_token=os.environ.get("HF_TOKEN"),
        hub_model_id=new_hub_id,
        hub_strategy="checkpoint",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer, metric),
        callbacks=[SavePeftModelCallback()],
        processing_class=processor.feature_extractor
    )

    processor.save_pretrained(training_args.output_dir)

    print(f"\n{'='*60}")
    print(f"Resuming training for {additional_epochs} more epochs")
    print(f"New model will be saved as: {new_hub_id}")
    print(f"{'='*60}\n")

    # Resume training (no resume_from_checkpoint needed since model is already loaded)
    trainer.train()

    # Save final model
    final_model_path = f"/my_vol/final_model_{new_model_name}"
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)

    # Save training metrics
    import json
    metrics_path = f"/my_vol/training_metrics_{new_model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Resumed training complete!")
    print(f"{'='*60}")
    print(f"  - Model saved to volume: {final_model_path}")
    print(f"  - Metrics saved to: {metrics_path}")
    print(f"  - TensorBoard logs: /my_vol/logs_{new_model_name}")
    print(f"  - Model pushed to HuggingFace Hub: {new_hub_id}")
    print(f"{'='*60}\n")

    # Commit volume changes
    from modal import Volume
    vol = Volume.from_name(VOLUME_NAME)
    vol.commit()

    return {
        "status": "success",
        "model_id": new_hub_id,
        "final_model_path": final_model_path,
        "metrics_path": metrics_path,
        "logs_path": f"/my_vol/logs_{new_model_name}"
    }

@app.function(
    image=image,
    volumes={"/my_vol": volume},
)
def plot_training_curves():
    """Generate and save training/validation curves as images"""
    import json
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Load metrics
    metrics_path = "/my_vol/training_metrics_afrispeech-twi-zulu-akan-extended.json"
    with open(metrics_path, "r") as f:
        log_history = json.load(f)

    # Extract training and eval metrics
    train_loss = [(log['step'], log['loss']) for log in log_history if 'loss' in log]
    eval_loss = [(log['step'], log['eval_loss']) for log in log_history if 'eval_loss' in log]
    eval_wer = [(log['step'], log['eval_wer']) for log in log_history if 'eval_wer' in log]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training loss
    if train_loss:
        steps, losses = zip(*train_loss)
        axes[0].plot(steps, losses, 'b-', label='Train Loss')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)

    # Validation loss
    if eval_loss:
        steps, losses = zip(*eval_loss)
        axes[1].plot(steps, losses, 'r-', label='Validation Loss')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].legend()
        axes[1].grid(True)

    # WER
    if eval_wer:
        steps, wer = zip(*eval_wer)
        axes[2].plot(steps, wer, 'g-', label='WER')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('WER (%)')
        axes[2].set_title('Word Error Rate')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = "/my_vol/extended_training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Commit volume
    from modal import Volume
    vol = Volume.from_name(VOLUME_NAME)
    vol.commit()

    print(f"✅ Training curves saved to: {plot_path}")
    return plot_path

@app.function(
    image=image,
    volumes={"/my_vol": volume},
    gpu="A10G",
    timeout=3600 * 2,  # 2 hours for evaluation
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_test_set():
    """Evaluate the trained model on the test set"""
    from datasets import load_from_disk
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel
    import torch
    import evaluate
    from tqdm import tqdm
    import json

    print("Loading preprocessed dataset...")
    dataset = load_from_disk("/my_vol/preprocessed_dataset")
    test_dataset = dataset['test']

    print("Loading trained model...")
    # Load the final model from volume
    model = WhisperForConditionalGeneration.from_pretrained(
        "distil-whisper/distil-large-v3",
        device_map="auto"
    )
    # Load PEFT adapter
    model = PeftModel.from_pretrained(model, "/my_vol/final_model_afrispeech-twi-zulu-akan-extended")
    model.eval()

    # Load processor
    processor = WhisperProcessor.from_pretrained("/my_vol/final_model_afrispeech-twi-zulu-akan-extended")

    # Load WER metric
    metric = evaluate.load("wer")

    print(f"Evaluating on {len(test_dataset)} test samples...")

    predictions = []
    references = []

    # Run inference on test set
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_dataset)):
            # Get input features
            input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(model.device)

            # Generate prediction
            predicted_ids = model.generate(input_features, max_length=448)

            # Decode prediction
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # Get reference (decode labels, replacing -100)
            labels = sample["labels"]
            labels = [l if l != -100 else processor.tokenizer.pad_token_id for l in labels]
            reference = processor.tokenizer.decode(labels, skip_special_tokens=True)

            predictions.append(transcription)
            references.append(reference)

            # Print some examples
            if i < 5:
                print(f"\nExample {i+1}:")
                print(f"  Reference: {reference}")
                print(f"  Prediction: {transcription}")

    # Calculate WER
    wer = 100 * metric.compute(predictions=predictions, references=references)

    # Save detailed results
    results = {
        "test_wer": wer,
        "num_samples": len(test_dataset),
        "examples": [
            {"reference": ref, "prediction": pred}
            for ref, pred in list(zip(references, predictions))[:20]  # Save first 20 examples
        ]
    }

    results_path = "/my_vol/extended_test_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Test Set Evaluation Complete!")
    print(f"{'='*50}")
    print(f"Test WER: {wer:.2f}%")
    print(f"Number of samples: {len(test_dataset)}")
    print(f"Results saved to: {results_path}")

    # Commit volume
    from modal import Volume
    vol = Volume.from_name(VOLUME_NAME)
    vol.commit()

    return {
        "test_wer": wer,
        "num_samples": len(test_dataset),
        "results_path": results_path
    }

@app.function(image=image)
@modal.asgi_app()
def web():
    import fastapi
    import time
    web_app = fastapi.FastAPI()

    @web_app.post("/prepare-dataset")
    async def prepare_dataset_endpoint():
        call = download_dataset.spawn()
        job_id = call.object_id
        print(f"\n✅ Dataset preparation job spawned: {job_id}")
        print("Waiting 30 seconds to confirm job submission...")

        # Wait to ensure Modal has registered the job
        time.sleep(30)

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Dataset preparation started"
        }

    @web_app.post("/train")
    async def train_endpoint():
        call = train.spawn()
        job_id = call.object_id
        print(f"\n✅ Training job spawned: {job_id}")
        print("Waiting 30 seconds to confirm job submission...")

        # Wait to ensure Modal has registered the job
        time.sleep(30)

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Training started"
        }

    @web_app.post("/resume-training")
    async def resume_training_endpoint(epochs: int = 3, model_name: str = "afrispeech-twi-zulu-akan-extended"):
        """Resume training from the saved checkpoint for additional epochs"""
        call = resume_training.spawn(additional_epochs=epochs, new_model_name=model_name)
        job_id = call.object_id
        print(f"\n✅ Resume training job spawned: {job_id}")
        print(f"Training for {epochs} additional epochs as: {model_name}")
        print("Waiting 30 seconds to confirm job submission...")

        time.sleep(30)

        return {
            "status": "success",
            "job_id": job_id,
            "message": f"Resumed training for {epochs} more epochs. New model: {model_name}"
        }

    @web_app.post("/plot-curves")
    async def plot_curves_endpoint():
        """Generate training curves plot after training completes"""
        call = plot_training_curves.spawn()
        job_id = call.object_id
        print(f"\n✅ Plot generation job spawned: {job_id}")

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Generating training curves plot"
        }

    @web_app.post("/evaluate")
    async def evaluate_endpoint():
        """Evaluate trained model on test set"""
        call = evaluate_test_set.spawn()
        job_id = call.object_id
        print(f"\n✅ Test evaluation job spawned: {job_id}")
        print("This will evaluate the model on the test set and calculate WER...")

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Test set evaluation started"
        }

    return web_app