# 🛡️ PhishBERT — AI-Powered Phishing Email Triage

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MITRE ATT&CK](https://img.shields.io/badge/MITRE-ATT%26CK-red)](https://attack.mitre.org/)

> Fine-tuned DistilBERT model for real-time phishing email classification with IOC extraction, MITRE ATT&CK tagging, and NIS2-aligned severity scoring. Built as a SOC L1 portfolio project targeting EU cybersecurity roles.

---

## 📸 Demo

![PhishBERT Demo UI](docs/demo_screenshot.png)

```
Input:  Raw email (paste or upload)
Output: Verdict | Confidence | IOCs | MITRE Tags | NIS2 Severity
```

---

## 🎯 Use Case

SOC L1 analysts spend significant time triaging phishing emails — determining if an email is benign, suspicious, or a confirmed phishing attempt. PhishBERT automates the initial triage step:

| Without PhishBERT | With PhishBERT |
|---|---|
| Manual keyword scanning | Automated ML classification |
| ~5–10 min per email | < 1 second per email |
| No structured IOC output | Structured IOC + MITRE tags |
| Inconsistent severity ratings | NIS2-aligned severity scoring |

---

## 🧠 Model Architecture

```
Raw Email (HTML + text)
        │
        ▼
  Preprocessing Pipeline
  (strip HTML, normalise URLs, clean text)
        │
        ▼
  DistilBERT Tokenizer
  (subword tokenisation, max 512 tokens)
        │
        ▼
  DistilBERT Base Uncased
  (6 transformer layers, 66M parameters)
        │
  [CLS] token embedding
        │
        ▼
  Linear Classifier Head
  (768 → 3 classes)
        │
        ▼
  Softmax → {benign, suspicious, phishing}
```

**Why DistilBERT?**
- 40% smaller than BERT-base, 60% faster inference
- Retains 97% of BERT's language understanding
- Fits in free-tier GPU memory (Google Colab T4)
- Pre-trained on Wikipedia + BookCorpus = strong English understanding

---

## 📊 Evaluation Results

> Results on held-out test set (15% of full dataset, never seen during training)

| Metric | Score |
|---|---|
| **Accuracy** | 0.9412 |
| **Weighted F1** | 0.9389 |
| **Precision (weighted)** | 0.9401 |
| **Recall (weighted)** | 0.9412 |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| ✅ Benign | 0.96 | 0.97 | 0.965 |
| ⚠️ Suspicious | 0.88 | 0.85 | 0.865 |
| 🚨 Phishing | 0.95 | 0.96 | 0.955 |

### Security-Specific Metrics

| Metric | Value | Interpretation |
|---|---|---|
| Phishing Detection Rate (TPR) | 96.1% | % of phishing caught |
| Phishing Miss Rate (FNR) | 3.9% | % of phishing missed ← critical |
| False Alarm Rate (FPR) | 4.2% | % of safe emails wrongly flagged |

> **Note:** Replace these with your actual results after training. These are representative targets.

---

## 🔍 Features

### ML Classification
- **3-class output**: benign / suspicious / phishing
- **Calibrated confidence scores** per class
- **Early stopping** to prevent overfitting

### IOC Extraction
Extracts from raw email text using regex:
- 🔗 URLs (with shortener detection)
- 🖥️ IPv4 addresses
- 📧 Email addresses
- 📎 Attachment filenames by extension

### MITRE ATT&CK Tagging
Automatically tags emails with relevant techniques:
- `T1566.001` — Spearphishing Attachment
- `T1566.002` — Spearphishing Link
- `T1598.003` — Spearphishing via Service
- `T1078` — Valid Accounts (credential harvesting)
- `T1204.001/002` — User Execution

### NIS2 Severity (EU Compliance)
Maps verdict to NIS2 Directive (EU 2022/2555) severity:
- **None** → Benign
- **Low** → Suspicious — log and monitor
- **Significant** → Phishing — 72h notification obligation if confirmed

---

## 🗂️ Project Structure

```
phishing-bert/
├── data/
│   ├── raw/                    # Raw downloaded dataset (not committed to git)
│   └── processed/              # Cleaned train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── preprocess.py           # Email cleaning + dataset splitting
│   ├── dataset.py              # PyTorch Dataset class + tokenizer loader
│   ├── train.py                # Fine-tuning loop (HuggingFace Trainer)
│   ├── evaluate.py             # Test set evaluation + confusion matrix
│   └── predict.py              # Inference + IOC extraction + MITRE tagging
├── models/
│   └── checkpoints/
│       ├── final_model/        # Saved model weights + tokenizer
│       └── evaluation/         # Confusion matrix plots + metrics JSON
├── app/
│   └── app.py                  # Gradio demo UI
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/phishing-bert.git
cd phishing-bert
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **Phishing Email Dataset** from Kaggle:
```
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
```
Save the CSV to `data/raw/phishing_emails.csv`.

### 3. Preprocess

```bash
python src/preprocess.py --input data/raw/phishing_emails.csv --output data/processed
```

### 4. Train (Google Colab recommended for GPU)

```bash
python src/train.py --epochs 4 --batch_size 16 --lr 2e-5
```

**Using Google Colab:**
1. Upload the repo to Google Drive
2. Mount Drive in Colab
3. Run `!python src/train.py --batch_size 32` (T4 GPU allows larger batch)
4. Download `models/checkpoints/final_model/` when done

### 5. Evaluate

```bash
python src/evaluate.py
```
Outputs confusion matrix to `models/checkpoints/evaluation/`.

### 6. Run Demo UI

```bash
python app/app.py
# Opens at http://localhost:7860

# Share publicly (Gradio shareable link):
python app/app.py --share
```

---

## ⚙️ Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base model | `distilbert-base-uncased` | Balance of speed and accuracy |
| Epochs | 4 | With early stopping (patience=2) |
| Batch size | 16 | Fits Colab T4 GPU memory |
| Learning rate | 2e-5 | Standard BERT fine-tuning range |
| Warmup steps | 200 | Prevents instability at start |
| Weight decay | 0.01 | L2 regularisation |
| Max token length | 512 | DistilBERT's hard limit |
| FP16 | Auto (if GPU) | 2x speedup with negligible accuracy loss |

---

## 🛠️ Inference API

```python
from src.predict import PhishingPredictor

# Load model
predictor = PhishingPredictor("models/checkpoints/final_model")

# Triage an email
result = predictor.predict("""
Dear user, your account has been suspended.
Verify immediately: http://bit.ly/verify-acc
""")

print(result.verdict)        # "phishing"
print(result.confidence)     # 0.9734
print(result.iocs["urls"])   # ["http://bit.ly/verify-acc"]
print(result.mitre_tags)     # ["T1566.002", "T1204.001"]
print(result.to_dict())      # Full JSON-serialisable result
```

---

## 🔐 MITRE ATT&CK Mapping

| ATT&CK ID | Technique | Trigger |
|---|---|---|
| T1566.001 | Spearphishing Attachment | Attachment extensions detected |
| T1566.002 | Spearphishing Link | URLs in email body |
| T1598.003 | Spearphishing via Service | Brand impersonation |
| T1078 | Valid Accounts | Credential harvesting language |
| T1204.001 | User Execution: Link | User prompted to click |
| T1204.002 | User Execution: File | User prompted to open file |

---

## 🇪🇺 NIS2 Compliance Notes

This tool aligns with the **NIS2 Directive (EU) 2022/2555** incident classification framework:

- **Significant incidents** (confirmed phishing leading to breach): 24h early warning + 72h notification to national CSIRT
- **Non-significant incidents** (suspicious / blocked phishing): internal logging recommended

PhishBERT's severity tiers map directly to this framework, supporting faster analyst decision-making within NIS2 reporting windows.

---

## 📈 Roadmap

- [ ] Add SHAP explanations (model interpretability)
- [ ] Header analysis module (SPF/DKIM/DMARC parsing)
- [ ] Batch processing mode (process full mailbox CSVs)
- [ ] Docker containerisation
- [ ] HuggingFace Hub upload
- [ ] Integration with TheHive (SOAR platform)

---

## 📚 Dataset

**Phishing Email Dataset** — Kaggle (Naser Abdullah Alam)
- ~82,000 labeled emails
- Labels: phishing / legitimate
- Source: Multiple public phishing corpora

---

## 🤝 Author

Built by **[Your Name]** as part of an EU SOC analyst portfolio.

- 🔗 LinkedIn: [your-linkedin]
- 🐙 GitHub: [your-github]
- 🛡️ TryHackMe: [your-thm-profile]

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

MITRE ATT&CK® is a registered trademark of The MITRE Corporation.
