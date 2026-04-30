# ============================================================
# notebooks/EDA.py  (run as: jupyter notebook or python EDA.py)
# ------------------------------------------------------------
# PURPOSE: Exploratory Data Analysis of the phishing email dataset.
#
# WHAT IS EDA?
#   Before training any model, you must understand your data:
#   - How many samples per class? (class imbalance)
#   - How long are the emails? (determines max_length setting)
#   - What words are most common in phishing vs benign emails?
#   - Are there data quality issues? (duplicates, nulls)
#
# EDA findings directly inform preprocessing decisions.
# Include EDA plots in your GitHub README to show thoroughness.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
import sys

# Add src/ to path for our preprocessing functions
sys.path.insert(0, "../src")
from preprocess import clean_email

# ── Style configuration ────────────────────────────────────────
plt.style.use("dark_background")
PALETTE = ["#2ecc71", "#f39c12", "#e74c3c"]  # green, orange, red

# ── Load dataset ───────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("../data/raw/phishing_emails.csv")

# Standardise column names (adjust if your CSV uses different names)
# This assumes columns: text_combined, label
print(f"Dataset shape: {df.shape}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")
print(f"\nNull values:\n{df.isnull().sum()}")


# ================================================================
# PLOT 1: Class Distribution Bar Chart
# ================================================================
# Shows whether the dataset is balanced or skewed.
# Imbalanced classes require stratified splitting (already handled).
# ================================================================

fig, ax = plt.subplots(figsize=(8, 5))
label_counts = df['label'].value_counts()
bars = ax.bar(label_counts.index, label_counts.values, color=PALETTE[:len(label_counts)], edgecolor='white', linewidth=0.5)

# Add count labels on top of each bar
for bar, count in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f'{count:,}', ha='center', va='bottom', fontsize=11, color='white')

ax.set_title("Class Distribution", fontsize=14, pad=15)
ax.set_xlabel("Email Class")
ax.set_ylabel("Count")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
print("Saved: class_distribution.png")
plt.close()


# ================================================================
# PLOT 2: Email Length Distribution (character count)
# ================================================================
# Helps decide the max_length parameter for tokenization.
# If 95% of emails are < 512 tokens, truncation affects few samples.
# ================================================================

# Compute character counts per email per class
df["text_len"] = df["text_combined"].fillna("").apply(len)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Email Length Distribution", fontsize=14)

# Histogram of all emails
axes[0].hist(df["text_len"].clip(upper=5000), bins=50, color="#3498db", edgecolor='none', alpha=0.8)
axes[0].axvline(df["text_len"].median(), color='yellow', linestyle='--', label=f'Median: {df["text_len"].median():.0f}')
axes[0].set_title("Character Length (all emails)")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Count")
axes[0].legend()

# Box plot by class
classes = df['label'].unique()
data_by_class = [df[df['label'] == c]['text_len'].clip(upper=5000).values for c in classes]
bp = axes[1].boxplot(data_by_class, labels=classes, patch_artist=True)
for patch, color in zip(bp['boxes'], PALETTE[:len(classes)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title("Length by Class")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Characters")

plt.tight_layout()
plt.savefig("email_length_distribution.png", dpi=150)
print("Saved: email_length_distribution.png")
plt.close()


# ================================================================
# PLOT 3: Top Phishing vs Benign Keywords
# ================================================================
# Visualises what words are most characteristic of each class.
# This validates that the model has meaningful signal to learn from.
# ================================================================

def get_top_words(texts: pd.Series, n: int = 30) -> list:
    """
    Extracts the most frequent words from a collection of emails.
    Filters out common English stop words that carry no signal.

    Args:
        texts (pd.Series): Series of cleaned email text strings.
        n (int): Number of top words to return.

    Returns:
        list: [(word, count), ...] sorted by frequency descending.
    """
    # Common English stop words to exclude (they appear in all classes equally)
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "this",
        "that", "these", "those", "it", "its", "we", "you", "your", "our",
        "i", "he", "she", "they", "them", "my", "his", "her", "their",
        "not", "no", "if", "so", "as", "all", "any", "more", "also",
    }

    word_counter = Counter()
    for text in texts.fillna(""):
        # Extract all lowercase words (no punctuation)
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        # Filter stop words
        words = [w for w in words if w not in stop_words]
        word_counter.update(words)

    return word_counter.most_common(n)


# Clean all emails first
print("Cleaning emails for keyword analysis (may take ~1 minute)...")
df["clean_text"] = df["text_combined"].apply(clean_email)

# Get top words for phishing vs benign
phishing_texts = df[df['label'].str.lower().isin(['phishing', 'malicious'])]['clean_text']
benign_texts   = df[df['label'].str.lower().isin(['benign', 'ham', 'legitimate'])]['clean_text']

phishing_words = get_top_words(phishing_texts, n=20)
benign_words   = get_top_words(benign_texts,   n=20)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Top 20 Keywords: Phishing vs Benign", fontsize=14)

# Phishing keywords
ph_words, ph_counts = zip(*phishing_words)
axes[0].barh(ph_words[::-1], ph_counts[::-1], color="#e74c3c", alpha=0.8)
axes[0].set_title("🚨 Phishing Emails", color="#e74c3c")
axes[0].set_xlabel("Frequency")

# Benign keywords
bn_words, bn_counts = zip(*benign_words)
axes[1].barh(bn_words[::-1], bn_counts[::-1], color="#2ecc71", alpha=0.8)
axes[1].set_title("✅ Benign Emails", color="#2ecc71")
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.savefig("top_keywords.png", dpi=150)
print("Saved: top_keywords.png")
plt.close()

# ── Summary statistics ─────────────────────────────────────────
print("\n" + "="*50)
print("EDA SUMMARY")
print("="*50)
print(f"Total emails:    {len(df):,}")
print(f"Null values:     {df['text_combined'].isnull().sum()}")
print(f"Duplicate rows:  {df.duplicated(subset='text_combined').sum()}")
print(f"\nEmail length stats:")
print(df['text_len'].describe().to_string())
print("\nTop 10 phishing keywords:")
for word, count in phishing_words[:10]:
    print(f"  {word}: {count:,}")
print("\n✅ EDA complete. Plots saved to notebooks/")
