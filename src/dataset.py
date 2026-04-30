# ============================================================
# dataset.py
# ------------------------------------------------------------
# PURPOSE: Converts preprocessed CSV data into PyTorch Dataset
#          objects that the Trainer can consume during training.
#
# WHY A CUSTOM DATASET?
#   PyTorch's DataLoader expects data in a specific format.
#   The PhishingEmailDataset class acts as an adapter —
#   it reads our CSV, tokenizes the text with DistilBERT's
#   tokenizer, and returns tensors that the model understands.
#
# DATA FLOW:
#   CSV file  →  PhishingEmailDataset  →  DataLoader  →  Trainer
# ============================================================

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


class PhishingEmailDataset(Dataset):
    """
    PyTorch Dataset for phishing email classification.

    Inherits from torch.utils.data.Dataset, which requires
    implementing __len__ and __getitem__ — PyTorch's DataLoader
    calls these automatically during training.

    What this class does:
      1. Reads clean email text from a CSV file
      2. Tokenizes each email using DistilBERT's tokenizer
         (converts words → token IDs the model understands)
      3. Returns a dict of tensors for each email:
           - input_ids: token ID sequence
           - attention_mask: 1 for real tokens, 0 for padding
           - labels: integer class (0=benign, 1=suspicious, 2=phishing)

    Args:
        csv_path (str): Path to processed CSV (train/val/test).
        tokenizer (DistilBertTokenizerFast): Tokenizer instance.
        max_length (int): Maximum token sequence length.
                          DistilBERT's hard limit is 512 tokens.
                          Emails longer than this will be truncated.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = 512,
    ):
        # ── Load the preprocessed CSV ─────────────────────────
        self.df = pd.read_csv(csv_path)

        # Ensure required columns exist
        assert "clean_text" in self.df.columns, "Missing 'clean_text' column — run preprocess.py first"
        assert "label_int" in self.df.columns, "Missing 'label_int' column — run preprocess.py first"

        # Store tokenizer and config
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Extract texts and labels as Python lists for fast access
        self.texts = self.df["clean_text"].tolist()
        self.labels = self.df["label_int"].tolist()

    def __len__(self) -> int:
        """
        Returns the total number of samples in this dataset.
        DataLoader uses this to know when one epoch is complete.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single training sample at the given index.

        Called by DataLoader internally during training —
        you don't call this directly.

        Tokenization details:
          - truncation=True: Cut emails longer than max_length tokens.
          - padding="max_length": Pad shorter emails with [PAD] tokens
            so all sequences in a batch are the same length.
            (Batching requires uniform tensor shapes.)
          - return_tensors="pt": Return PyTorch tensors, not Python lists.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict with keys: input_ids, attention_mask, labels
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the email text
        # The tokenizer converts words → subword tokens → integer IDs
        encoding = self.tokenizer(
            text,
            truncation=True,        # Cut at max_length if too long
            padding="max_length",   # Pad to max_length if too short
            max_length=self.max_length,
            return_tensors="pt",    # Return PyTorch tensors
        )

        # .squeeze(0) removes the batch dimension added by return_tensors="pt"
        # Shape: [1, 512] → [512]
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_tokenizer(model_name: str = "distilbert-base-uncased") -> DistilBertTokenizerFast:
    """
    Loads and returns the DistilBERT tokenizer.

    The 'Fast' tokenizer is implemented in Rust — significantly faster
    than the Python version for batch tokenization.

    We use 'distilbert-base-uncased' which lowercases all input text.
    This is appropriate for email content where "URGENT" and "urgent"
    carry the same meaning.

    Args:
        model_name (str): HuggingFace model identifier.

    Returns:
        DistilBertTokenizerFast: Tokenizer instance.
    """
    return DistilBertTokenizerFast.from_pretrained(model_name)


def get_datasets(
    processed_dir: str = "data/processed",
    tokenizer: DistilBertTokenizerFast = None,
    max_length: int = 512,
) -> tuple:
    """
    Convenience function that loads all three dataset splits at once.

    Args:
        processed_dir (str): Directory containing train.csv, val.csv, test.csv.
        tokenizer: DistilBERT tokenizer. If None, loads default.
        max_length (int): Max token sequence length.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    train_dataset = PhishingEmailDataset(f"{processed_dir}/train.csv", tokenizer, max_length)
    val_dataset   = PhishingEmailDataset(f"{processed_dir}/val.csv",   tokenizer, max_length)
    test_dataset  = PhishingEmailDataset(f"{processed_dir}/test.csv",  tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset
