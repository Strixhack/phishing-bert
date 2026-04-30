# ============================================================
# train.py
# ------------------------------------------------------------
# PURPOSE: Fine-tunes DistilBERT on the phishing email dataset.
#
# WHAT IS FINE-TUNING?
#   DistilBERT is pre-trained on massive English text (Wikipedia +
#   BookCorpus), so it already understands English grammar, context,
#   and semantics. Fine-tuning adds a small classification head on
#   top and trains the whole model on *our specific task* (phishing
#   detection), adapting those general language skills to our domain.
#
# TRAINING LOOP (simplified):
#   For each batch of emails:
#     1. Forward pass: model predicts class probabilities
#     2. Compute loss: how wrong were the predictions?
#     3. Backward pass: calculate gradients
#     4. Optimizer step: nudge weights to reduce loss
#   Repeat until validation F1 stops improving.
#
# USAGE:
#   python src/train.py
#   python src/train.py --epochs 6 --batch_size 32 --lr 3e-5
# ============================================================

import os
import json
import logging
import argparse
import numpy as np
import torch

from transformers import (
    DistilBertForSequenceClassification,  # DistilBERT + classification head
    Trainer,                              # HuggingFace high-level training loop
    TrainingArguments,                    # Hyperparameter configuration
    EarlyStoppingCallback,                # Stop training when val metric plateaus
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# Import our custom dataset loader
from dataset import get_tokenizer, get_datasets

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Class label configuration ─────────────────────────────────
# Must match LABEL_MAP in preprocess.py
NUM_LABELS = 3
ID2LABEL = {0: "benign", 1: "suspicious", 2: "phishing"}
LABEL2ID = {"benign": 0, "suspicious": 1, "phishing": 2}


def compute_metrics(eval_pred) -> dict:
    """
    Computes evaluation metrics from model predictions.

    Called by HuggingFace Trainer after each validation epoch.
    The Trainer passes raw logits (unnormalised scores per class),
    and we compute the final metrics from those.

    WHY F1 AND NOT JUST ACCURACY?
      In security, false negatives (missed phishing) are expensive.
      F1 balances precision and recall — it penalises the model for
      both missing phishing emails AND crying wolf on safe emails.
      We also report per-class F1 to see where the model struggles.

    Args:
        eval_pred: NamedTuple(predictions=logits, label_ids=true_labels)
                   logits shape: [n_samples, n_classes]
                   label_ids shape: [n_samples]

    Returns:
        dict: Metric name → value (all floats, rounded to 4 decimal places)
    """
    logits, label_ids = eval_pred

    # argmax converts logits → predicted class index
    # e.g., [0.1, 0.3, 0.6] → 2 (phishing)
    predictions = np.argmax(logits, axis=-1)

    # Overall metrics (weighted = accounts for class imbalance)
    acc     = accuracy_score(label_ids, predictions)
    f1      = f1_score(label_ids, predictions, average="weighted", zero_division=0)
    prec    = precision_score(label_ids, predictions, average="weighted", zero_division=0)
    recall  = recall_score(label_ids, predictions, average="weighted", zero_division=0)

    # Per-class F1 — tells us if the model is weak on any specific class
    f1_per_class = f1_score(label_ids, predictions, average=None, zero_division=0)

    return {
        "accuracy":             round(float(acc), 4),
        "f1":                   round(float(f1), 4),
        "precision":            round(float(prec), 4),
        "recall":               round(float(recall), 4),
        "f1_benign":            round(float(f1_per_class[0]), 4),
        "f1_suspicious":        round(float(f1_per_class[1]), 4),
        "f1_phishing":          round(float(f1_per_class[2]), 4),
    }


def build_model(model_name: str = "distilbert-base-uncased") -> DistilBertForSequenceClassification:
    """
    Loads DistilBERT with a fresh 3-class classification head.

    DistilBertForSequenceClassification = DistilBERT encoder +
    a linear layer on top of the [CLS] token embedding.

    The [CLS] token is a special token prepended to every input.
    After passing through DistilBERT's transformer layers, the
    [CLS] embedding summarises the whole email — we classify from that.

    id2label and label2id are stored in the model config so that
    model.config.id2label gives human-readable class names during inference.

    Args:
        model_name (str): HuggingFace model identifier.

    Returns:
        DistilBertForSequenceClassification: Model ready for fine-tuning.
    """
    logger.info(f"Loading model: {model_name}")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        # ignore_mismatched_sizes: allows loading weights even when the
        # classification head size changes (from 2 → 3 labels, for example)
        ignore_mismatched_sizes=True,
    )
    return model


def build_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
) -> TrainingArguments:
    """
    Configures all training hyperparameters.

    KEY HYPERPARAMETERS EXPLAINED:
      - learning_rate (lr): How large each weight update step is.
        Too high → training diverges. Too low → very slow convergence.
        2e-5 is the standard starting point for BERT fine-tuning.

      - warmup_steps: Gradually ramp up LR from 0 to lr over N steps.
        Prevents instability at the start of training.

      - weight_decay: L2 regularisation — penalises large weights to
        prevent overfitting (especially important with small datasets).

      - fp16: Half-precision (16-bit) training. Uses less GPU memory
        and runs ~2x faster with negligible accuracy loss.
        Set to False if your GPU doesn't support fp16.

      - load_best_model_at_end: After training, automatically restore
        the checkpoint with the highest validation F1.

    Args:
        output_dir (str): Where to save model checkpoints.
        epochs (int): Number of full passes over the training data.
        batch_size (int): Number of emails per gradient update step.
        lr (float): Peak learning rate.
        warmup_steps (int): Linear LR warmup steps.
        weight_decay (float): L2 regularisation coefficient.

    Returns:
        TrainingArguments: Fully configured hyperparameter object.
    """
    return TrainingArguments(
        output_dir=output_dir,

        # ── Training schedule ──────────────────────────────
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # Can use larger batch for eval (no gradients)

        # ── Optimiser settings ─────────────────────────────
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,

        # ── Evaluation & checkpointing ─────────────────────
        evaluation_strategy="epoch",    # Run validation at end of each epoch
        save_strategy="epoch",          # Save checkpoint at end of each epoch
        load_best_model_at_end=True,    # Restore best checkpoint when done
        metric_for_best_model="f1",     # Use weighted F1 to select best model
        greater_is_better=True,

        # ── Performance ────────────────────────────────────
        fp16=torch.cuda.is_available(), # Use fp16 only if GPU supports it

        # ── Logging ────────────────────────────────────────
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,               # Log training loss every 50 steps
        report_to="none",               # Change to "wandb" for W&B tracking

        # ── Reproducibility ────────────────────────────────
        seed=42,
    )


def train(
    processed_dir: str = "data/processed",
    output_dir: str = "models/checkpoints",
    model_name: str = "distilbert-base-uncased",
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_steps: int = 200,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 2,
) -> None:
    """
    Main training function — orchestrates the full fine-tuning pipeline.

    Steps:
      1. Detect available hardware (GPU / CPU)
      2. Load tokenizer and datasets
      3. Build DistilBERT model
      4. Configure training arguments
      5. Create Trainer and run training
      6. Save final model + tokenizer

    Args:
        processed_dir (str): Directory with train/val/test CSVs.
        output_dir (str): Directory to save model checkpoints.
        model_name (str): Base DistilBERT model to fine-tune from.
        epochs (int): Training epochs.
        batch_size (int): Per-device batch size.
        lr (float): Learning rate.
        warmup_steps (int): LR warmup steps.
        weight_decay (float): L2 regularisation.
        early_stopping_patience (int): Stop if val F1 doesn't improve
                                        for this many epochs.
    """
    # ── Hardware detection ────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device.upper()}")
    if device == "cpu":
        logger.warning(
            "No GPU detected. Training on CPU will be slow (~10x slower).\n"
            "Consider using Google Colab (free T4 GPU) for faster training."
        )

    # ── Load tokenizer and datasets ───────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(model_name)

    logger.info("Loading datasets...")
    train_dataset, val_dataset, _ = get_datasets(processed_dir, tokenizer)
    logger.info(
        f"Dataset sizes — Train: {len(train_dataset):,} | "
        f"Val: {len(val_dataset):,}"
    )

    # ── Build model ───────────────────────────────────────────
    model = build_model(model_name)

    # ── Configure training ────────────────────────────────────
    training_args = build_training_args(
        output_dir, epochs, batch_size, lr, warmup_steps, weight_decay
    )

    # ── Early stopping callback ───────────────────────────────
    # Stops training if val F1 hasn't improved for `patience` epochs.
    # Prevents overfitting and saves compute time.
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience
    )

    # ── Create Trainer ────────────────────────────────────────
    # HuggingFace Trainer handles:
    #   - Distributed training (multi-GPU)
    #   - Mixed precision (fp16)
    #   - Gradient accumulation
    #   - Checkpoint saving
    #   - Logging
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    # ── Run training ──────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train()

    # ── Save best model and tokenizer ────────────────────────
    # Saves model weights, config, and tokenizer vocab to disk.
    # These files are what you push to GitHub / HuggingFace Hub.
    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    logger.info(f"\n✅ Training complete. Model saved to: {final_model_dir}")

    # ── Save training config for reproducibility ─────────────
    # Storing hyperparameters alongside the model lets anyone
    # reproduce your results exactly — important for GitHub README.
    config = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "num_labels": NUM_LABELS,
        "label_map": LABEL2ID,
    }
    with open(f"{final_model_dir}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Training config saved.")


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for phishing detection")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir",    type=str, default="models/checkpoints")
    parser.add_argument("--model_name",    type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs",        type=int, default=4)
    parser.add_argument("--batch_size",    type=int, default=16)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--warmup_steps",  type=int, default=200)
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--patience",      type=int, default=2)
    args = parser.parse_args()

    train(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
    )
