import streamlit as st
import tempfile
import os
import requests
from PIL import Image
import io
from datetime import datetime
import logging
from typing import Optional  # Added this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service Configuration
SERVICES = {
    "pdf": os.getenv("PDF_SERVICE_URL", "http://localhost:8001"),
    "image": os.getenv("GPU_SERVICE_URL", "http://localhost:8000")
}
API_TOKEN = os.getenv("API_AUTH_TOKEN")

def setup_ui():
    st.set_page_config(
        page_title="Multi-RAG Agent",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 Multi-RAG LangGraph Agent")
    st.markdown("""
        <style>
            .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; }
            .stFileUploader>div>div>div>div { color: #4CAF50; }
        </style>
    """, unsafe_allow_html=True)

def handle_file_upload(file_type: str, extensions: list) -> Optional[str]:
    uploaded_file = st.file_uploader(
        f"Upload {file_type.upper()}",
        type=extensions,
        key=f"{file_type}_uploader"
    )
    if uploaded_file:
        suffix = f".{extensions[0]}" if extensions else ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_file.getvalue())
            return f.name
    return None

def process_pdf(pdf_path: str, query: str, lang: str):
    try:
        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"{SERVICES['pdf']}/process",
                files={"file": f},
                data={"query": query, "lang": lang},
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                timeout=60
            )
        return response.json() if response.ok else None
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return None

def image_search(image_path: str, top_k: int = 3):
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{SERVICES['image']}/search",
                files={"file": f},
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                params={"top_k": top_k},
                timeout=30
            )
        return response.json() if response.ok else None
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        return None

def cleanup_files(*file_paths):
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"Couldn't delete {path}: {e}")

def main():
    setup_ui()
    
    # Input section
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_input("Ask something...", placeholder="Enter your question")
    with col2:
        lang = st.selectbox("Response Language", ["en", "de", "hi", "fr"], index=0)
    
    # File upload
    pdf_path = handle_file_upload("PDF", ["pdf"])
    image_path = handle_file_upload("Image", ["png", "jpg", "jpeg"])

    if st.button("Process", type="primary"):
        if not query and not pdf_path and not image_path:
            st.warning("Please provide at least a question or upload a file")
            return
            
        with st.spinner("Processing..."):
            try:
                result = {}
                
                # Process PDF if uploaded
                if pdf_path:
                    pdf_result = process_pdf(pdf_path, query, lang)
                    if pdf_result:
                        result.update(pdf_result)
                
                # Process image if uploaded
                if image_path:
                    img_result = image_search(image_path)
                    if img_result:
                        result["image_matches"] = img_result.get("matches", [])
                
                # Display results
                st.success(result.get("output", "Processing complete"))
                
                if image_path and "image_matches" in result:
                    st.subheader("Similar Images")
                    cols = st.columns(3)
                    for i, match in enumerate(result["image_matches"][:3]):
                        with cols[i]:
                            img_url = match["image_url"]
                            if img_url.startswith("/"):
                                img_url = f"{SERVICES['image']}{img_url}"
                            img_data = requests.get(img_url).content
                            st.image(Image.open(io.BytesIO(img_data)), 
                                     caption=f"Score: {match['similarity']:.2f}")
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
            finally:
                cleanup_files(pdf_path, image_path)

if __name__ == "__main__":
    main()