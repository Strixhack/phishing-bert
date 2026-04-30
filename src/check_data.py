import pandas as pd
import matplotlib.pyplot as plt

# ── Load the raw dataset ──────────────────────────────────────
df = pd.read_csv("data/raw/phishing_emails.csv")

# ── Map integer labels to readable names ──────────────────────
label_names = {0: "Benign / Safe", 1: "Phishing"}
df["label_name"] = df["label"].map(label_names)

# ── Print counts and percentages ──────────────────────────────
total = len(df)
counts = df["label_name"].value_counts()

print("\n" + "="*40)
print("  DATASET BREAKDOWN")
print("="*40)
for label, count in counts.items():
    pct = (count / total) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<20} {count:>6,}  ({pct:.1f}%)  {bar}")
print(f"\n  Total emails:        {total:>6,}")
print("="*40)

# ── Bar chart ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PhishBERT — Dataset Breakdown", fontsize=14, fontweight="bold")

colors = ["#2ecc71", "#e74c3c"]  # green = safe, red = phishing

# Count bar chart
axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
axes[0].set_title("Email Count by Class")
axes[0].set_ylabel("Number of Emails")
for i, (label, count) in enumerate(counts.items()):
    axes[0].text(i, count + 300, f"{count:,}", ha="center", fontsize=11, fontweight="bold")

# Pie chart
axes[1].pie(
    counts.values,
    labels=counts.index,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
axes[1].set_title("Proportion of Classes")

plt.tight_layout()
plt.savefig("data_breakdown.png", dpi=150)
print("\n  Chart saved → data_breakdown.png")
plt.show()