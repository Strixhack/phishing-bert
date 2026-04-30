# ============================================================
# preprocess.py
# ------------------------------------------------------------
# PURPOSE: Cleans and prepares raw email data for model training.
#
# Raw emails contain HTML tags, URLs, special characters, and
# inconsistent formatting — all of which hurt model performance.
# This module strips noise and normalises text so DistilBERT
# sees clean, meaningful content rather than HTML boilerplate.
#
# PIPELINE:
#   Raw CSV  →  clean_email()  →  build_dataset()  →  Saved CSVs
# ============================================================

import re
import os
import logging
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

# Configure logging so we can track preprocessing progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Label mapping ────────────────────────────────────────────
# We convert string labels to integers for PyTorch compatibility.
# 0 = benign (safe email)
# 1 = suspicious (uncertain — low-confidence phishing signals)
# 2 = phishing (confirmed malicious)
LABEL_MAP = {
    "benign": 0,
    "safe": 0,
    "ham": 0,
    "legitimate": 0,
    "suspicious": 1,
    "spam": 1,           # Spam is treated as suspicious, not confirmed phishing
    "phishing": 2,
    "malicious": 2,
}


def clean_email(raw_text: str) -> str:
    """
    Cleans a single raw email string into normalised plain text.

    Steps performed:
      1. Strip HTML tags using BeautifulSoup (many phishing emails use HTML)
      2. Replace URLs with a [URL] token — the presence of URLs is
         captured by the token itself; the specific domain is less important
         and can cause the model to overfit to specific domains.
      3. Replace email addresses with [EMAIL] token for the same reason.
      4. Remove non-ASCII characters (mojibake, encoding artefacts)
      5. Collapse repeated whitespace into single spaces

    Args:
        raw_text (str): The raw email body, possibly containing HTML.

    Returns:
        str: Cleaned plain-text version of the email.
    """
    # Guard against non-string inputs (NaN values from pandas read_csv)
    if not isinstance(raw_text, str):
        return ""

    # Step 1 — Strip HTML tags
    # BeautifulSoup parses HTML/XML and .get_text() extracts visible text only.
    # separator=" " ensures words aren't concatenated across tags.
    soup = BeautifulSoup(raw_text, "lxml")
    text = soup.get_text(separator=" ")

    # Step 2 — Replace hyperlinks with placeholder token
    # Regex matches http:// or https:// followed by any non-whitespace chars.
    text = re.sub(r"https?://\S+", "[URL]", text)

    # Step 3 — Replace email addresses with placeholder token
    # Regex matches standard email format: localpart@domain.tld
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.\w+\b", "[EMAIL]", text)

    # Step 4 — Remove non-ASCII characters
    # encode → decode trick drops anything outside the ASCII range cleanly.
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Step 5 — Collapse multiple spaces/newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads the raw CSV dataset and validates required columns exist.

    Expected CSV format (Kaggle phishing dataset):
        text_combined  |  label
        "Dear user..." |  phishing
        "Hi John..."   |  benign

    Args:
        csv_path (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Raw dataframe with at least 'text_combined' and 'label' columns.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist at the given path.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Download from: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset"
        )

    logger.info(f"Loading raw dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate columns — different Kaggle versions use different column names
    # Try to auto-detect the text and label columns
    text_col_candidates = ["text_combined", "body", "text", "email_text", "message"]
    label_col_candidates = ["label", "class", "category", "type"]

    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    label_col = next((c for c in label_col_candidates if c in df.columns), None)

    if not text_col or not label_col:
        raise ValueError(
            f"Could not find required columns. Found: {list(df.columns)}\n"
            f"Expected one of {text_col_candidates} for text and {label_col_candidates} for labels."
        )

    # Rename to standard names for rest of pipeline
    df = df.rename(columns={text_col: "text_combined", label_col: "label"})
    logger.info(f"Loaded {len(df):,} rows. Label distribution:\n{df['label'].value_counts()}")

    return df


def build_dataset(
    csv_path: str,
    output_dir: str = "data/processed",
    test_size: float = 0.15,
    val_size: float = 0.10,
    random_seed: int = 42,
) -> dict:
    """
    Full preprocessing pipeline: load → clean → encode labels → split → save.

    Splits the data into three sets:
      - Train (75%): Used to update model weights during training.
      - Validation (10%): Monitored during training to detect overfitting.
      - Test (15%): Held out completely; only used for final evaluation.

    Args:
        csv_path (str): Path to raw CSV file.
        output_dir (str): Directory to save processed train/val/test CSVs.
        test_size (float): Fraction of data for the test set.
        val_size (float): Fraction of remaining data for validation.
        random_seed (int): Seed for reproducible splits.

    Returns:
        dict: {"train": df_train, "val": df_val, "test": df_test}
    """
    # ── Load raw data ─────────────────────────────────────────
    df = load_raw_dataset(csv_path)

    # ── Clean email text ──────────────────────────────────────
    logger.info("Cleaning email text (this may take a minute)...")
    df["clean_text"] = df["text_combined"].apply(clean_email)

    # Drop rows where cleaning produced an empty string
    # (e.g., emails that were entirely HTML images with no alt text)
    original_len = len(df)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
    logger.info(f"Dropped {original_len - len(df)} empty rows after cleaning.")

    # ── Encode string labels to integers ─────────────────────
    # .str.lower() normalises "Phishing" → "phishing" before mapping
    df["label_int"] = df["label"].map({0: 0, 1: 2})

    # Warn about any labels that didn't match our LABEL_MAP
    unmapped = df["label_int"].isna().sum()
    if unmapped > 0:
        unknown_labels = df[df["label_int"].isna()]["label"].unique()
        logger.warning(f"{unmapped} rows have unrecognised labels: {unknown_labels}. Dropping them.")
        df = df.dropna(subset=["label_int"]).reset_index(drop=True)

    df["label_int"] = df["label_int"].astype(int)

    # ── Train / Test split ────────────────────────────────────
    # stratify=df["label_int"] ensures class distribution is preserved
    # across all splits (important for imbalanced datasets)
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=df["label_int"]
    )

    # ── Train / Validation split ──────────────────────────────
    # val_size is relative to the remaining data after test split
    relative_val_size = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val, test_size=relative_val_size, random_state=random_seed, stratify=df_train_val["label_int"]
    )

    # ── Save processed splits to disk ────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    save_cols = ["clean_text", "label", "label_int"]

    df_train[save_cols].to_csv(f"{output_dir}/train.csv", index=False)
    df_val[save_cols].to_csv(f"{output_dir}/val.csv", index=False)
    df_test[save_cols].to_csv(f"{output_dir}/test.csv", index=False)

    logger.info(
        f"\n✅ Dataset saved to {output_dir}/\n"
        f"   Train:      {len(df_train):>6,} rows\n"
        f"   Validation: {len(df_val):>6,} rows\n"
        f"   Test:       {len(df_test):>6,} rows"
    )

    return {"train": df_train, "val": df_val, "test": df_test}


# ── Script entry point ────────────────────────────────────────
# Run directly: python src/preprocess.py
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess phishing email dataset")
    parser.add_argument("--input", type=str, default="data/raw/phishing_emails.csv",
                        help="Path to raw CSV file")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Directory to save processed splits")
    args = parser.parse_args()

    build_dataset(csv_path=args.input, output_dir=args.output)
