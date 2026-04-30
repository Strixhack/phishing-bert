# ============================================================
# evaluate.py
# ------------------------------------------------------------
# PURPOSE: Evaluates the fine-tuned model on the held-out test set
#          and produces a comprehensive security-focused report.
#
# WHY A SEPARATE EVALUATION SCRIPT?
#   The test set is NEVER touched during training or validation.
#   Running evaluation only at the end gives an honest, unbiased
#   measure of how the model will perform in the real world.
#   This is what you report in your GitHub README / model card.
#
# OUTPUTS:
#   - Classification report (precision, recall, F1 per class)
#   - Confusion matrix (visualised as heatmap)
#   - Security-focused metrics (FNR, FPR for phishing class)
#   - Saved plots to models/checkpoints/evaluation/
#
# USAGE:
#   python src/evaluate.py
#   python src/evaluate.py --model_dir models/checkpoints/final_model
# ============================================================

import os
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from dataset import PhishingEmailDataset

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Class names (must match training label order) ─────────────
CLASS_NAMES = ["benign", "suspicious", "phishing"]


def load_model_and_tokenizer(model_dir: str) -> tuple:
    """
    Loads the fine-tuned model and its tokenizer from disk.

    The saved directory contains:
      - config.json: model architecture and label mappings
      - pytorch_model.bin: trained weight tensors
      - tokenizer.json / vocab.txt: tokenizer vocabulary

    Args:
        model_dir (str): Path to saved model directory.

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading model from: {model_dir}")

    model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

    # Move model to GPU if available — faster inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout layers)

    logger.info(f"Model loaded successfully. Running on: {device}")
    return model, tokenizer


def get_predictions(model, test_dataset, batch_size: int = 32) -> tuple:
    """
    Runs inference on the entire test dataset and returns predictions.

    We use HuggingFace Trainer for prediction — it handles batching,
    device placement, and mixed precision automatically.

    Args:
        model: Fine-tuned DistilBERT model.
        test_dataset: PhishingEmailDataset instance for test split.
        batch_size (int): Batch size for inference.

    Returns:
        tuple: (true_labels, predicted_labels, confidence_scores)
               All are numpy arrays of shape [n_samples].
    """
    # Minimal TrainingArguments needed just to run prediction
    pred_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=batch_size,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=pred_args)

    # predict() returns a PredictionOutput namedtuple:
    # .predictions = raw logits [n_samples, n_classes]
    # .label_ids   = true labels [n_samples]
    logger.info("Running inference on test set...")
    prediction_output = trainer.predict(test_dataset)

    logits     = prediction_output.predictions          # shape: [n, 3]
    true_labels = prediction_output.label_ids.astype(int)

    # Convert logits → probabilities → predicted class
    # softmax normalises logits to sum to 1.0 (probability distribution)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()  # shape: [n, 3]
    predicted_labels = np.argmax(probs, axis=-1)                 # shape: [n]
    confidence_scores = np.max(probs, axis=-1)                   # shape: [n]

    return true_labels, predicted_labels, confidence_scores


def compute_security_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict:
    """
    Computes security-specific metrics beyond standard ML metrics.

    In a SOC context, not all errors are equal:
      - False Negative (FN): Missed phishing → attacker gets through ⚠️ HIGH COST
      - False Positive (FP): Benign flagged as phishing → analyst time wasted

    False Negative Rate (FNR) = FN / (FN + TP)
      → What % of actual phishing emails did we miss?
      → Target: < 5% (ideally < 2%)

    False Positive Rate (FPR) = FP / (FP + TN)
      → What % of safe emails did we incorrectly flag?
      → Target: < 10%

    These map directly to NIS2 incident detection requirements —
    you can reference this in EU job interviews.

    Args:
        true_labels (np.ndarray): Ground truth class indices.
        predicted_labels (np.ndarray): Model predicted class indices.

    Returns:
        dict: Security-specific metric dictionary.
    """
    # Treat class 2 (phishing) as the positive class for binary metrics
    phishing_class = 2

    # Build binary arrays: 1 = phishing, 0 = not phishing
    true_binary = (true_labels == phishing_class).astype(int)
    pred_binary = (predicted_labels == phishing_class).astype(int)

    # Confusion matrix components for phishing class
    TP = int(np.sum((true_binary == 1) & (pred_binary == 1)))  # Correctly caught phishing
    FN = int(np.sum((true_binary == 1) & (pred_binary == 0)))  # Missed phishing (dangerous)
    FP = int(np.sum((true_binary == 0) & (pred_binary == 1)))  # False alarm on safe email
    TN = int(np.sum((true_binary == 0) & (pred_binary == 0)))  # Correctly safe

    # Rate calculations (guard against division by zero)
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0  # Phishing miss rate
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # Safe email false alarm rate
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Phishing detection rate (recall)

    return {
        "phishing_TP":  TP,
        "phishing_FN":  FN,
        "phishing_FP":  FP,
        "phishing_TN":  TN,
        "phishing_TPR_detection_rate":   round(tpr, 4),
        "phishing_FNR_miss_rate":        round(fnr, 4),
        "phishing_FPR_false_alarm_rate": round(fpr, 4),
    }


def plot_confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray, output_dir: str) -> None:
    """
    Generates and saves a confusion matrix heatmap.

    A confusion matrix shows counts of:
      - Correct predictions (diagonal)
      - Misclassifications (off-diagonal)

    Reading the matrix:
      Row = true class, Column = predicted class.
      Off-diagonal entries = mistakes.
      High numbers on the diagonal = good model.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Predicted labels.
        output_dir: Directory to save the plot.
    """
    cm = confusion_matrix(true_labels, predicted_labels)

    # Normalise to percentages for easier reading
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PhishBERT — Confusion Matrix", fontsize=14, fontweight="bold")

    # Raw counts heatmap
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Percentage heatmap
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1]
    )
    axes[1].set_title("Row-Normalised (%)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to: {save_path}")


def evaluate(
    model_dir: str = "models/checkpoints/final_model",
    test_csv: str = "data/processed/test.csv",
    output_dir: str = "models/checkpoints/evaluation",
    batch_size: int = 32,
) -> dict:
    """
    Full evaluation pipeline — runs on the held-out test set and
    produces a comprehensive security-focused evaluation report.

    Args:
        model_dir (str): Path to saved fine-tuned model.
        test_csv (str): Path to test CSV file.
        output_dir (str): Directory to save evaluation results and plots.
        batch_size (int): Inference batch size.

    Returns:
        dict: All computed metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model and tokenizer ──────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # ── Load test dataset ─────────────────────────────────────
    test_dataset = PhishingEmailDataset(test_csv, tokenizer)
    logger.info(f"Test set size: {len(test_dataset):,} samples")

    # ── Run inference ─────────────────────────────────────────
    true_labels, predicted_labels, confidence_scores = get_predictions(
        model, test_dataset, batch_size
    )

    # ── Standard classification metrics ──────────────────────
    report = classification_report(
        true_labels, predicted_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    logger.info("\n" + classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES))

    # ── Security-specific metrics ─────────────────────────────
    security_metrics = compute_security_metrics(true_labels, predicted_labels)
    logger.info("\n=== Security Metrics ===")
    for k, v in security_metrics.items():
        logger.info(f"  {k}: {v}")

    # ── Overall accuracy and F1 ───────────────────────────────
    overall = {
        "accuracy": round(accuracy_score(true_labels, predicted_labels), 4),
        "f1_weighted": round(f1_score(true_labels, predicted_labels, average="weighted"), 4),
        "mean_confidence": round(float(np.mean(confidence_scores)), 4),
    }

    # ── Plot confusion matrix ─────────────────────────────────
    plot_confusion_matrix(true_labels, predicted_labels, output_dir)

    # ── Assemble full results dict ────────────────────────────
    results = {
        "overall": overall,
        "per_class": report,
        "security_metrics": security_metrics,
    }

    # ── Save results to JSON ──────────────────────────────────
    # This JSON is what you copy into your GitHub README / model card
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Evaluation complete. Results saved to: {results_path}")
    logger.info(
        f"\n{'='*50}\n"
        f"  Accuracy:              {overall['accuracy']:.4f}\n"
        f"  Weighted F1:           {overall['f1_weighted']:.4f}\n"
        f"  Phishing Detection:    {security_metrics['phishing_TPR_detection_rate']:.4f}\n"
        f"  Phishing Miss Rate:    {security_metrics['phishing_FNR_miss_rate']:.4f}\n"
        f"{'='*50}"
    )

    return results


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PhishBERT on test set")
    parser.add_argument("--model_dir",  type=str, default="models/checkpoints/final_model")
    parser.add_argument("--test_csv",   type=str, default="data/processed/test.csv")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    evaluate(
        model_dir=args.model_dir,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
