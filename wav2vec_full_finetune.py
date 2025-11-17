import modal
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np

app = modal.App("wav2vec-full-finetune")

# Create Modal image with all required packages
image = (modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
.apt_install("ffmpeg", "libsndfile1")
.pip_install("fastapi[standard]==0.115.0","requests","pandas","matplotlib")
.pip_install_from_requirements("requirements.txt")
)

# Persistent volume
VOLUME_NAME = "afrispeech-train-volume"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator for CTC training"""
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        import torch

        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


def compute_metrics(pred, processor, metric):
    """Compute WER for Wav2Vec2"""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 with pad_token_id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@app.function(
    image=image,
    volumes={"/my_vol": volume},
    gpu="L40S",
    timeout=3600 * 10,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_wav2vec_full():
    """Full fine-tuning of Wav2Vec2 (no LoRA)"""
    from datasets import load_from_disk
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
    import torch
    import evaluate
    import json

    print("Loading preprocessed dataset...")
    dataset = load_from_disk("/my_vol/preprocessed_dataset")

    # Use pre-trained vocabulary from facebook model
    print("Loading pre-trained Wav2Vec2 processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    print(f"Vocabulary size: {len(processor.tokenizer)}")

    # Prepare dataset for Wav2Vec2
    def prepare_dataset(batch):
        audio = batch["audio"]

        # Extract audio input values
        batch["input_values"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_values[0]

        # Encode text to label ids using the new API (text parameter instead of as_target_processor)
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids

        return batch

    print("Reprocessing dataset for Wav2Vec2...")
    # Don't remove all columns - keep the ones we create (input_values, labels)
    columns_to_remove = [col for col in dataset['train'].column_names if col not in ['audio', 'transcript','input_values','labels']]
    wav2vec_dataset = dataset.map(
        prepare_dataset,
        remove_columns=columns_to_remove,
        num_proc=1
    )
    print(f"Sample from train after map:{wav2vec_dataset['train'][0]}")

    print("Loading Wav2Vec2 model for full fine-tuning...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        torch_dtype=torch.float16,
        device_map="auto",
        ignore_mismatched_sizes=True
    )

    # Freeze feature encoder
    model.freeze_feature_encoder()

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}%")

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Load metric
    metric = evaluate.load("wer")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/my_vol/wav2vec_full_checkpoints",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 32
        eval_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=False,
        # max_grad_norm=1.0,  # Explicitly set gradient clipping norm
        save_steps=200,
        eval_steps=200,
        logging_steps=50,
        learning_rate=1e-4,  # Lower LR for full fine-tuning
        warmup_steps=500,
        save_total_limit=2,
        logging_dir="/my_vol/wav2vec_full_logs",
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        hub_token=os.environ.get("HF_TOKEN"),
        hub_model_id="sirsam01/wav2vec2-full-afrispeech",
        hub_strategy="checkpoint",
        group_by_length=True,
        # Fix for FP16 + gradient checkpointing issue
        optim="adamw_torch",  # Use PyTorch Adam optimizer instead of fused
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=wav2vec_dataset["train"],
        eval_dataset=wav2vec_dataset["dev"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric),
        tokenizer=processor.feature_extractor,
    )

    print(f"\nTraining Configuration:")
    print(f"  - Model: Full fine-tuning (NO LoRA)")
    print(f"  - Vocab: Pre-trained (facebook/wav2vec2-base-960h)")
    print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Epochs: {training_args.num_train_epochs}\n")

    # Save processor
    processor.save_pretrained(training_args.output_dir)

    # Train
    print("Starting full fine-tuning...")
    trainer.train()

    # Save final model
    final_model_path = "/my_vol/wav2vec_full_final_model"
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)

    # Save metrics
    metrics_path = "/my_vol/wav2vec_full_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\n✅ Full fine-tuning complete!")
    print(f"  - Model saved to: {final_model_path}")
    print(f"  - Metrics saved to: {metrics_path}")

    # Commit volume
    from modal import Volume
    vol = Volume.from_name(VOLUME_NAME)
    vol.commit()

    return {
        "status": "success",
        "model_id": "sirsam01/wav2vec2-full-afrispeech",
        "final_model_path": final_model_path,
        "metrics_path": metrics_path
    }


@app.function(
    image=image,
    volumes={"/my_vol": volume},
)
def plot_wav2vec_training_curves():
    """Generate and save training/validation curves for Wav2Vec2"""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Load metrics
    metrics_path = "/my_vol/wav2vec_full_metrics.json"
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
        axes[0].set_title('Training Loss (Wav2Vec2)')
        axes[0].legend()
        axes[0].grid(True)

    # Validation loss
    if eval_loss:
        steps, losses = zip(*eval_loss)
        axes[1].plot(steps, losses, 'r-', label='Validation Loss')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss (Wav2Vec2)')
        axes[1].legend()
        axes[1].grid(True)

    # WER
    if eval_wer:
        steps, wer = zip(*eval_wer)
        axes[2].plot(steps, wer, 'g-', label='WER')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('WER (%)')
        axes[2].set_title('Word Error Rate (Wav2Vec2)')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = "/my_vol/wav2vec_training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Commit volume
    from modal import Volume
    vol = Volume.from_name(VOLUME_NAME)
    vol.commit()

    print(f"✅ Wav2Vec2 training curves saved to: {plot_path}")
    return plot_path


@app.function(
    image=image,
    volumes={"/my_vol": volume},
    gpu="A10G",
    timeout=3600 * 2,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_wav2vec_test():
    """Evaluate trained Wav2Vec2 model on test set"""
    from datasets import load_from_disk
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    import torch
    import evaluate
    from tqdm import tqdm
    import json

    print("Loading test dataset...")
    dataset = load_from_disk("/my_vol/preprocessed_dataset")
    test_dataset = dataset['test']

    print("Loading trained model...")
    processor = Wav2Vec2Processor.from_pretrained("/my_vol/wav2vec_full_final_model")
    model = Wav2Vec2ForCTC.from_pretrained("/my_vol/wav2vec_full_final_model", device_map="auto")
    model.eval()

    # Load metric
    metric = evaluate.load("wer")

    print(f"Evaluating on {len(test_dataset)} test samples...")

    predictions = []
    references = []

    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_dataset)):
            # Process audio
            input_values = torch.tensor(sample["input_values"]).unsqueeze(0).to(model.device)

            # Get prediction
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)

            # Decode
            pred_str = processor.batch_decode(pred_ids)[0]
            label_str = processor.tokenizer.decode(sample["labels"], group_tokens=False)

            predictions.append(pred_str)
            references.append(label_str)

            if i < 5:
                print(f"\nExample {i+1}:")
                print(f"  Reference: {label_str}")
                print(f"  Prediction: {pred_str}")

    # Calculate WER
    wer = 100 * metric.compute(predictions=predictions, references=references)

    results = {
        "test_wer": wer,
        "num_samples": len(test_dataset),
        "examples": [
            {"reference": ref, "prediction": pred}
            for ref, pred in list(zip(references, predictions))[:20]
        ]
    }

    results_path = "/my_vol/wav2vec_full_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Wav2Vec2 Full Fine-tuning Test Evaluation Complete!")
    print(f"{'='*50}")
    print(f"Test WER: {wer:.2f}%")
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

    @web_app.post("/train-wav2vec-full")
    async def train_endpoint():
        call = train_wav2vec_full.spawn()
        job_id = call.object_id
        print(f"\n✅ Wav2Vec2 full fine-tuning job spawned: {job_id}")
        time.sleep(30)

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Wav2Vec2 full fine-tuning started"
        }

    @web_app.post("/evaluate-wav2vec-full")
    async def evaluate_endpoint():
        call = evaluate_wav2vec_test.spawn()
        job_id = call.object_id
        print(f"\n✅ Wav2Vec2 evaluation job spawned: {job_id}")

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Wav2Vec2 test evaluation started"
        }

    @web_app.post("/plot-wav2vec-curves")
    async def plot_curves_endpoint():
        """Generate training curves plot for Wav2Vec2"""
        call = plot_wav2vec_training_curves.spawn()
        job_id = call.object_id
        print(f"\n✅ Wav2Vec2 plot generation job spawned: {job_id}")

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Generating Wav2Vec2 training curves plot"
        }

    return web_app
