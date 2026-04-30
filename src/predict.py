# ============================================================
# predict.py
# ------------------------------------------------------------
# PURPOSE: Runs inference on a single email and returns a
#          structured triage verdict with IOC extraction.
#
# This is the inference module used by the Gradio demo UI.
# It combines:
#   1. Model prediction (benign / suspicious / phishing)
#   2. IOC (Indicator of Compromise) extraction via regex
#   3. MITRE ATT&CK technique tagging
#   4. NIS2-aligned incident severity classification
#
# USAGE (Python):
#   from predict import PhishingPredictor
#   predictor = PhishingPredictor("models/checkpoints/final_model")
#   result = predictor.predict("Dear user, click here to verify...")
#   print(result)
#
# USAGE (CLI):
#   python src/predict.py --email "paste email text here"
#   python src/predict.py --file path/to/email.txt
# ============================================================

import re
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from preprocess import clean_email

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Class configuration ───────────────────────────────────────
CLASS_NAMES  = ["benign", "suspicious", "phishing"]
CLASS_EMOJIS = {"benign": "✅", "suspicious": "⚠️", "phishing": "🚨"}

# ── MITRE ATT&CK technique mapping ───────────────────────────
# Maps phishing signal patterns to relevant ATT&CK techniques.
# This lets analysts immediately understand the threat vector.
MITRE_TECHNIQUE_MAP = {
    "T1566.001": "Spearphishing Attachment — email contains suspicious attachment indicators",
    "T1566.002": "Spearphishing Link — email contains suspicious URLs",
    "T1598.003": "Spearphishing via Service — impersonation of legitimate service",
    "T1204.001": "User Execution: Malicious Link — user is prompted to click a link",
    "T1204.002": "User Execution: Malicious File — user is prompted to open a file",
    "T1078":     "Valid Accounts — credential harvesting attempt detected",
    "T1534":     "Internal Spearphishing — may target internal users",
}

# ── NIS2 severity mapping ─────────────────────────────────────
# Maps model verdict to NIS2 incident severity classification.
# Relevant for EU SOC roles — demonstrates regulatory awareness.
NIS2_SEVERITY_MAP = {
    "benign":     {"level": "None",     "reporting": "No reporting required"},
    "suspicious": {"level": "Low",      "reporting": "Log and monitor; report if confirmed"},
    "phishing":   {"level": "Significant", "reporting": "72h NIS2 notification if confirmed incident"},
}


@dataclass
class TriageResult:
    """
    Structured container for a single email triage result.

    Using a dataclass makes the result easy to serialise to JSON
    (for API responses) and display in the Gradio UI.

    Fields:
        verdict:        "benign" | "suspicious" | "phishing"
        confidence:     Float 0.0-1.0 — model's confidence in verdict
        probabilities:  Dict of class → probability for all 3 classes
        iocs:           Extracted Indicators of Compromise
        mitre_tags:     Relevant MITRE ATT&CK technique IDs
        nis2_severity:  NIS2 incident severity classification
        signals:        Human-readable list of detected phishing signals
        clean_text:     Cleaned version of the input email
    """
    verdict: str
    confidence: float
    probabilities: dict
    iocs: dict
    mitre_tags: list
    nis2_severity: dict
    signals: list
    clean_text: str

    def to_dict(self) -> dict:
        """Converts the result to a plain dict for JSON serialisation."""
        return {
            "verdict":       self.verdict,
            "confidence":    round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "iocs":          self.iocs,
            "mitre_tags":    self.mitre_tags,
            "nis2_severity": self.nis2_severity,
            "signals":       self.signals,
        }


def extract_iocs(raw_text: str) -> dict:
    """
    Extracts Indicators of Compromise (IOCs) from email text using regex.

    IOCs are artefacts that indicate potential malicious activity.
    In phishing emails, common IOCs include:
      - Suspicious URLs (especially shortened or non-HTTPS links)
      - Embedded IP addresses (instead of domain names)
      - Harvested email addresses (target / sender spoofing)
      - File attachment references (.exe, .zip, .doc, etc.)

    These are extracted BEFORE cleaning so we preserve the original URLs.

    Args:
        raw_text (str): Original (uncleaned) email text.

    Returns:
        dict: {
            "urls": [...],       # All HTTP(S) URLs found
            "ips": [...],        # IPv4 addresses found
            "emails": [...],     # Email addresses found
            "attachments": [...] # Attachment filenames found
        }
    """
    # Extract all HTTP/HTTPS URLs
    urls = re.findall(r"https?://[^\s<>\"']+", raw_text)

    # Extract IPv4 addresses (e.g., 192.168.1.1)
    # Pattern: four groups of 1-3 digits separated by dots
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw_text)

    # Filter out false positives in IP extraction (e.g., version numbers like 1.2.3.4)
    ips = [ip for ip in ips if all(0 <= int(octet) <= 255 for octet in ip.split("."))]

    # Extract email addresses
    emails = re.findall(r"\b[\w.+%-]+@[\w-]+\.[a-zA-Z]{2,}\b", raw_text)

    # Extract potential attachment references
    # Covers common malicious attachment types
    attachment_pattern = r"\b[\w\s-]+\.(exe|zip|rar|doc|docx|xls|xlsx|pdf|js|vbs|bat|cmd|ps1)\b"
    attachments = re.findall(attachment_pattern, raw_text, re.IGNORECASE)

    # Deduplicate while preserving order
    return {
        "urls":        list(dict.fromkeys(urls)),
        "ips":         list(dict.fromkeys(ips)),
        "emails":      list(dict.fromkeys(emails)),
        "attachments": list(dict.fromkeys(attachments)),
    }

from typing import Tuple
def detect_phishing_signals(clean_text: str, iocs: dict) -> tuple[list, list]:
    """
    Applies rule-based heuristics to identify specific phishing signals.

    These rules complement the ML model — the model catches subtle
    language patterns, while rules catch specific known-bad patterns.

    Args:
        clean_text (str): Cleaned email body text.
        iocs (dict): Extracted IOCs from extract_iocs().

    Returns:
        tuple: (signals: list of str, mitre_tags: list of str)
               signals = human-readable descriptions of what was found
               mitre_tags = relevant ATT&CK technique IDs
    """
    signals    = []
    mitre_tags = set()

    text_lower = clean_text.lower()

    # ── Urgency language patterns ─────────────────────────────
    # Phishing emails often create artificial time pressure
    urgency_keywords = [
        "urgent", "immediately", "action required", "account suspended",
        "verify now", "within 24 hours", "limited time", "expire",
    ]
    found_urgency = [kw for kw in urgency_keywords if kw in text_lower]
    if found_urgency:
        signals.append(f"Urgency language detected: {', '.join(found_urgency)}")
        mitre_tags.add("T1566.002")

    # ── Credential harvesting language ────────────────────────
    credential_keywords = [
        "password", "username", "login", "sign in", "verify your account",
        "confirm your identity", "update your credentials",
    ]
    found_cred = [kw for kw in credential_keywords if kw in text_lower]
    if found_cred:
        signals.append(f"Credential harvesting language: {', '.join(found_cred)}")
        mitre_tags.add("T1078")

    # ── URL-based signals ─────────────────────────────────────
    if iocs["urls"]:
        signals.append(f"{len(iocs['urls'])} URL(s) found in email body")
        mitre_tags.add("T1566.002")
        mitre_tags.add("T1204.001")

        # Check for URL shorteners (common in phishing to hide real domain)
        shorteners = ["bit.ly", "tinyurl", "t.co", "ow.ly", "goo.gl", "short.link"]
        short_urls = [u for u in iocs["urls"] if any(s in u for s in shorteners)]
        if short_urls:
            signals.append(f"Shortened URLs detected (common phishing vector): {short_urls}")

        # Check for non-HTTPS links (insecure)
        http_only = [u for u in iocs["urls"] if u.startswith("http://")]
        if http_only:
            signals.append(f"Non-HTTPS (HTTP) links found — data may be transmitted insecurely")

    # ── IP address instead of domain ──────────────────────────
    if iocs["ips"]:
        signals.append(f"Raw IP address(es) found: {iocs['ips']} — legitimate services use domain names")
        mitre_tags.add("T1566.002")

    # ── Suspicious attachment types ───────────────────────────
    high_risk_exts = ["exe", "js", "vbs", "bat", "cmd", "ps1"]
    risky = [a for a in iocs["attachments"] if a.lower() in high_risk_exts]
    if risky:
        signals.append(f"High-risk attachment type(s): {risky}")
        mitre_tags.add("T1566.001")
        mitre_tags.add("T1204.002")
    elif iocs["attachments"]:
        signals.append(f"Attachment reference(s) found: {iocs['attachments']}")
        mitre_tags.add("T1566.001")

    # ── Impersonation patterns ────────────────────────────────
    impersonation_brands = [
        "microsoft", "paypal", "apple", "amazon", "google", "netflix",
        "facebook", "instagram", "linkedin", "dhl", "fedex", "irs", "hmrc",
    ]
    found_brands = [b for b in impersonation_brands if b in text_lower]
    if found_brands:
        signals.append(f"Brand impersonation indicators: {', '.join(found_brands)}")
        mitre_tags.add("T1598.003")

    return signals, sorted(mitre_tags)


class PhishingPredictor:
    """
    Inference class that wraps the fine-tuned DistilBERT model.

    Provides a single .predict() method that takes raw email text
    and returns a structured TriageResult with verdict, IOCs,
    MITRE tags, and NIS2 severity.

    Usage:
        predictor = PhishingPredictor("models/checkpoints/final_model")
        result = predictor.predict(email_text)
    """

    def __init__(self, model_dir: str):
        """
        Loads model and tokenizer from disk.

        Args:
            model_dir (str): Path to fine-tuned model directory.
        """
        logger.info(f"Loading PhishBERT predictor from: {model_dir}")

        # Detect GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and send to device
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()  # Disable dropout — we're doing inference, not training

        # Load tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

        logger.info(f"Predictor ready on: {self.device}")

    def predict(self, email_text: str, max_length: int = 512) -> TriageResult:
        """
        Triages a single email and returns a structured verdict.

        Pipeline:
          1. Extract IOCs from raw text (before cleaning removes URLs)
          2. Clean email text
          3. Tokenize for DistilBERT
          4. Run model forward pass
          5. Detect phishing signals via heuristics
          6. Assemble and return TriageResult

        Args:
            email_text (str): Raw email body text (may contain HTML).
            max_length (int): Max tokens for DistilBERT (hard limit: 512).

        Returns:
            TriageResult: Structured triage verdict with all supporting data.
        """
        # ── Step 1: Extract IOCs from raw text ────────────────
        # Must happen BEFORE cleaning — cleaning removes URLs
        iocs = extract_iocs(email_text)

        # ── Step 2: Clean email text ──────────────────────────
        clean_text = clean_email(email_text)

        if not clean_text:
            # Empty email after cleaning — treat as suspicious
            return TriageResult(
                verdict="suspicious", confidence=0.0,
                probabilities={c: 0.0 for c in CLASS_NAMES},
                iocs=iocs, mitre_tags=[], signals=["Email body was empty after cleaning"],
                nis2_severity=NIS2_SEVERITY_MAP["suspicious"], clean_text=""
            )

        # ── Step 3: Tokenize ──────────────────────────────────
        inputs = self.tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        # Move input tensors to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ── Step 4: Forward pass ──────────────────────────────
        # torch.no_grad() disables gradient computation — not needed for inference
        # and saves significant memory
        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.logits shape: [1, 3] — one prediction, 3 classes
        logits = outputs.logits[0]  # shape: [3]

        # Convert logits to probability distribution
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # shape: [3]

        # Get predicted class and confidence
        predicted_idx = int(probs.argmax())
        verdict       = CLASS_NAMES[predicted_idx]
        confidence    = float(probs[predicted_idx])

        # Build probability dict for display
        probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(3)}

        # ── Step 5: Detect signals & MITRE tags ──────────────
        signals, mitre_tags = detect_phishing_signals(clean_text, iocs)

        # ── Step 6: Assemble result ───────────────────────────
        return TriageResult(
            verdict=verdict,
            confidence=confidence,
            probabilities=probabilities,
            iocs=iocs,
            mitre_tags=mitre_tags,
            nis2_severity=NIS2_SEVERITY_MAP[verdict],
            signals=signals,
            clean_text=clean_text,
        )


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PhishBERT inference on an email")
    parser.add_argument("--model_dir", type=str, default="models/checkpoints/final_model")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--email", type=str, help="Email text as a string")
    group.add_argument("--file",  type=str, help="Path to a .txt file containing the email")

    args = parser.parse_args()

    # Load email text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            email_text = f.read()
    else:
        email_text = args.email

    # Run prediction
    predictor = PhishingPredictor(args.model_dir)
    result    = predictor.predict(email_text)

    # Pretty-print result as JSON
    print("\n" + "="*60)
    print(f"  VERDICT: {CLASS_EMOJIS[result.verdict]} {result.verdict.upper()}")
    print(f"  CONFIDENCE: {result.confidence:.1%}")
    print("="*60)
    print(json.dumps(result.to_dict(), indent=2))
