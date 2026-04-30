# ============================================================
# app/app.py
# ------------------------------------------------------------
# PURPOSE: Gradio-based demo UI for PhishBERT.
#
# Gradio creates a browser-based web interface automatically —
# no HTML/CSS/JS required. It's the fastest way to build a
# shareable demo for a ML model.
#
# FEATURES:
#   - Paste any email → instant triage verdict
#   - Confidence scores for all 3 classes (bar chart)
#   - Extracted IOCs listed in a structured table
#   - MITRE ATT&CK tags with technique descriptions
#   - NIS2 severity and reporting guidance
#   - Example phishing emails pre-loaded for demo
#
# USAGE:
#   python app/app.py
#   python app/app.py --model_dir models/checkpoints/final_model
#   python app/app.py --share  (creates a public Gradio link)
# ============================================================

import os
import sys
import argparse

import gradio as gr

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from predict import PhishingPredictor, CLASS_EMOJIS, MITRE_TECHNIQUE_MAP

# ── Example emails for the demo ───────────────────────────────
# Pre-loaded examples so recruiters/reviewers can try immediately
EXAMPLE_EMAILS = [
    # Example 1: Clear phishing
    ["""Subject: URGENT: Your account will be suspended in 24 hours

Dear Valued Customer,

We have detected suspicious activity on your PayPal account.
Your account has been temporarily limited.

To restore full access, please verify your identity immediately:
http://bit.ly/payp4l-verify-account

You must confirm your account details within 24 hours or your
account will be permanently suspended.

Click here to verify: http://192.168.99.1/login

Thank you,
PayPal Security Team"""],

    # Example 2: Legitimate email
    ["""Hi Sarah,

Just following up on our meeting yesterday regarding the Q3
budget review. I've attached the spreadsheet we discussed.

The numbers look good — I think we're on track to hit our
targets this quarter. Let me know if you have any questions
before the presentation on Friday.

Best,
Marcus"""],

    # Example 3: Suspicious (ambiguous)
    ["""Important notice from IT Department

All employees are required to update their network credentials
before end of business Friday.

Please sign in to the employee portal and update your password:
https://company-portal.updatecredentials.net/login

Failure to do so may result in loss of network access.

IT Support Team"""],
]


def build_ui(predictor: PhishingPredictor) -> gr.Blocks:
    """
    Constructs the Gradio UI layout and wires up event handlers.

    Gradio uses a Blocks-based API for complex layouts.
    Components are arranged in Rows and Columns.
    .click() / .submit() wire UI events to Python functions.

    Args:
        predictor (PhishingPredictor): Loaded model predictor instance.

    Returns:
        gr.Blocks: Complete Gradio app, ready to launch.
    """

    def run_triage(email_text: str) -> tuple:
        """
        Called when user clicks "Analyse Email" button.

        Takes raw email text → runs PhishingPredictor → formats
        results for each Gradio output component.

        Args:
            email_text (str): Raw email pasted by user.

        Returns:
            tuple: (verdict_html, confidence_dict, ioc_text, mitre_text, nis2_text)
        """
        if not email_text or not email_text.strip():
            error_html = "<p style='color:gray'>⬅️ Paste an email on the left and click Analyse.</p>"
            return error_html, None, "", "", ""

        # ── Run model prediction ──────────────────────────────
        result = predictor.predict(email_text)

        # ── Format verdict banner ─────────────────────────────
        # Colour-coded based on verdict
        verdict_colors = {
            "benign":     ("#1a472a", "#2d6a4f", "✅"),
            "suspicious": ("#7b3f00", "#d4a017", "⚠️"),
            "phishing":   ("#4a0000", "#c0392b", "🚨"),
        }
        bg, border, emoji = verdict_colors[result.verdict]
        verdict_html = f"""
        <div style="background:{bg}; border-left:5px solid {border};
                    padding:16px; border-radius:8px; font-family:monospace;">
            <h2 style="margin:0; color:white;">{emoji} {result.verdict.upper()}</h2>
            <p style="margin:4px 0 0; color:#ccc;">
                Confidence: <strong>{result.confidence:.1%}</strong>
            </p>
        </div>
        """

        # ── Format confidence scores for bar chart ────────────
        # Gradio Label component accepts {label: score} dicts
        confidence_dict = {
            f"{CLASS_EMOJIS[k]} {k}": round(v, 4)
            for k, v in result.probabilities.items()
        }

        # ── Format IOCs ───────────────────────────────────────
        ioc_lines = []
        if result.iocs["urls"]:
            ioc_lines.append(f"🔗 URLs ({len(result.iocs['urls'])}):")
            for url in result.iocs["urls"][:10]:  # Cap at 10 for readability
                ioc_lines.append(f"   {url}")

        if result.iocs["ips"]:
            ioc_lines.append(f"\n🖥️  IP Addresses:")
            for ip in result.iocs["ips"]:
                ioc_lines.append(f"   {ip}")

        if result.iocs["emails"]:
            ioc_lines.append(f"\n📧 Email Addresses:")
            for email in result.iocs["emails"][:5]:
                ioc_lines.append(f"   {email}")

        if result.iocs["attachments"]:
            ioc_lines.append(f"\n📎 Attachment Types:")
            for att in result.iocs["attachments"]:
                ioc_lines.append(f"   .{att}")

        if not any(result.iocs.values()):
            ioc_lines.append("No IOCs extracted.")

        ioc_text = "\n".join(ioc_lines)

        # ── Format MITRE ATT&CK tags ──────────────────────────
        if result.mitre_tags:
            mitre_lines = []
            for tid in result.mitre_tags:
                desc = MITRE_TECHNIQUE_MAP.get(tid, "No description available")
                mitre_lines.append(f"[{tid}] {desc}")
            mitre_text = "\n".join(mitre_lines)
        else:
            mitre_text = "No MITRE ATT&CK techniques identified."

        # ── Format detected signals ───────────────────────────
        if result.signals:
            signals_text = "\n".join(f"• {s}" for s in result.signals)
        else:
            signals_text = "• No specific phishing signals detected."

        # ── Format NIS2 severity ──────────────────────────────
        nis2 = result.nis2_severity
        nis2_text = (
            f"Severity Level: {nis2['level']}\n"
            f"Reporting Guidance: {nis2['reporting']}\n\n"
            f"NIS2 Directive (EU) 2022/2555 requires significant incidents\n"
            f"to be reported within 24h (early warning) and 72h (full notification)."
        )

        return verdict_html, confidence_dict, ioc_text, signals_text, mitre_text, nis2_text

    # ── Build UI layout ───────────────────────────────────────
    with gr.Blocks(
        title="PhishBERT — Phishing Email Triage",
        theme=gr.themes.Monochrome(),
        css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        .header { text-align: center; padding: 20px 0; }
        """
    ) as demo:

        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="font-family:monospace; font-size:2em;">
                🛡️ PhishBERT
            </h1>
            <p style="color:gray; font-size:0.95em;">
                AI-powered phishing email triage using fine-tuned DistilBERT |
                MITRE ATT&CK tagged | NIS2-aligned severity
            </p>
        </div>
        """)

        # Main layout: two columns
        with gr.Row():

            # ── Left column: Input ────────────────────────────
            with gr.Column(scale=1):
                email_input = gr.Textbox(
                    label="📧 Paste Email Body Here",
                    placeholder="Paste raw email content (headers + body)...",
                    lines=20,
                    max_lines=30,
                )
                analyse_btn = gr.Button("🔍 Analyse Email", variant="primary", size="lg")

                # Pre-loaded examples
                gr.Examples(
                    examples=EXAMPLE_EMAILS,
                    inputs=[email_input],
                    label="Load Example Email",
                )

            # ── Right column: Results ─────────────────────────
            with gr.Column(scale=1):
                verdict_display = gr.HTML(
                    label="Verdict",
                    value="<p style='color:gray; font-family:monospace;'>Awaiting analysis...</p>"
                )

                confidence_chart = gr.Label(
                    label="Class Probabilities",
                    num_top_classes=3,
                )

                with gr.Tabs():
                    with gr.Tab("🔍 Detected Signals"):
                        signals_output = gr.Textbox(
                            label="Phishing Signals",
                            lines=6,
                            interactive=False,
                        )

                    with gr.Tab("🔗 IOCs"):
                        ioc_output = gr.Textbox(
                            label="Indicators of Compromise",
                            lines=8,
                            interactive=False,
                        )

                    with gr.Tab("🗺️ MITRE ATT&CK"):
                        mitre_output = gr.Textbox(
                            label="MITRE ATT&CK Techniques",
                            lines=6,
                            interactive=False,
                        )

                    with gr.Tab("🇪🇺 NIS2 Severity"):
                        nis2_output = gr.Textbox(
                            label="NIS2 Incident Severity",
                            lines=6,
                            interactive=False,
                        )

        # Footer
        gr.HTML("""
        <div style="text-align:center; margin-top:20px; color:gray; font-size:0.8em;">
            PhishBERT | Fine-tuned DistilBERT | Built for EU SOC L1 Portfolio |
            MITRE ATT&CK® is a registered trademark of The MITRE Corporation
        </div>
        """)

        # ── Wire button click to run_triage function ──────────
        analyse_btn.click(
            fn=run_triage,
            inputs=[email_input],
            outputs=[
                verdict_display,
                confidence_chart,
                ioc_output,
                signals_output,
                mitre_output,
                nis2_output,
            ],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch PhishBERT Gradio demo")
    parser.add_argument(
        "--model_dir", type=str,
        default="models/checkpoints/final_model",
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public shareable Gradio link"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Local port to run the server on"
    )
    args = parser.parse_args()

    # Load predictor (loads model weights into memory)
    predictor = PhishingPredictor(args.model_dir)

    # Build and launch UI
    demo = build_ui(predictor)
    demo.launch(
        server_name="0.0.0.0",   # Accessible on local network
        server_port=args.port,
        share=args.share,        # --share creates a public gradio.live URL
    )
