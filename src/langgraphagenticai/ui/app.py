import os
import tempfile
import streamlit as st
import requests
import logging

from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf

# ─────────────────────────────────────────────────────────────────────────────
#   CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PDF_API_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8001").rstrip("/")
API_TOKEN   = os.getenv("API_AUTH_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_app")

# ─────────────────────────────────────────────────────────────────────────────
#   UI SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF-only RAG Agent",
    page_icon="📄",
    layout="wide"
)
st.title("📄 PDF RAG Agent")
st.markdown("Upload a PDF, index it, then ask free-form questions.")

# ─────────────────────────────────────────────────────────────────────────────
#   STATE
# ─────────────────────────────────────────────────────────────────────────────
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# ─────────────────────────────────────────────────────────────────────────────
#   HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_upload(uploader, suffix):
    """Save uploaded file to a temp path."""
    f = uploader
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.getvalue())
    tmp.close()
    return tmp.name

def call_service(endpoint: str, files=None, data=None):
    """Call CPU-side API if you prefer remote ingestion/query."""
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    resp = requests.post(f"{PDF_API_URL}{endpoint}", files=files, data=data, headers=headers)
    resp.raise_for_status()
    return resp.json()

# ─────────────────────────────────────────────────────────────────────────────
#   PDF UPLOAD & INGEST
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Step 1 – Upload PDF", type=["pdf"])
if uploaded:
    path = save_upload(uploaded, ".pdf")
    with st.spinner("Indexing PDF into Pinecone…"):
        # local ingestion
        ingest_pdf(path)
        # or if you have a separate service:
        # call_service("/ingest", files={"file": open(path,"rb")})
    st.success("✅ Indexed!")
    st.session_state.pdf_path = path

# ─────────────────────────────────────────────────────────────────────────────
#   QUESTION
# ─────────────────────────────────────────────────────────────────────────────
query = st.text_input("Step 2 – Ask your question…")

if st.button("Run PDF RAG") and query:
    if not st.session_state.pdf_path:
        st.warning("Please upload & index a PDF first.")
    else:
        with st.spinner("Running RetrievalQA…"):
            # local query
            answer = query_pdf(query)
            # or if remote:
            # answer = call_service("/query", data={"query": query})
        st.subheader("🎯 Answer")
        st.write(answer)
