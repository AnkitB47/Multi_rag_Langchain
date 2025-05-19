import streamlit as st
import tempfile
import os
import requests
from datetime import datetime
import logging
from typing import Optional

# ─── Configuration ───────────────────────────────────────
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8001")
API_TOKEN        = os.getenv("API_AUTH_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ui():
    st.set_page_config(
        page_title="📄 PDF Q&A",
        page_icon="📄",
        layout="wide"
    )
    st.title("📄 PDF Question-Answering Service")

def upload_pdf() -> Optional[str]:
    """Allow user to upload a PDF and return its local path."""
    uploaded = st.file_uploader("Upload PDF", type="pdf")
    if not uploaded:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name

def ask_pdf(pdf_path: str, question: str, lang: str) -> Optional[str]:
    """POST the PDF + question to the PDF service and return answer."""
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                f"{PDF_SERVICE_URL}/process",
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                files={"file": f},
                data={"query": question, "lang": lang},
                timeout=60
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("output") or data.get("answer") or "No answer returned."
    except Exception as e:
        logger.error(f"PDF service error: {e}", exc_info=True)
        st.error("❌ Failed to process PDF. See logs.")
        return None

def main():
    setup_ui()

    question = st.text_input("Your question about the PDF:")
    lang     = st.selectbox("Language", ["en", "de", "hi", "fr"], index=0)
    pdf_path = upload_pdf()

    if st.button("Get Answer"):
        if not pdf_path or not question:
            st.warning("Please upload a PDF and enter a question.")
            return

        with st.spinner("Processing PDF…"):
            answer = ask_pdf(pdf_path, question, lang)
            if answer:
                st.markdown("### Answer")
                st.write(answer)

    # Clean up
    if pdf_path and os.path.exists(pdf_path):
        os.unlink(pdf_path)

if __name__ == "__main__":
    main()
