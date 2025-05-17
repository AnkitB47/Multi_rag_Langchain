# app.py

import streamlit as st
import tempfile
import os
from langgraphagenticai.graph.chatbot_graph import create_graph
from typing import Optional, Tuple
import logging
from datetime import datetime
import requests  
from PIL import Image
import io

GPU_API_URL = os.getenv("GPU_API_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = os.getenv("API_AUTH_TOKEN")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ui():
    """Configure Streamlit UI settings"""
    st.set_page_config(
        page_title="Multi-RAG Agent",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 Multi-RAG LangGraph Agent")
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            .stFileUploader>div>div>div>div {
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)
    
# --- New GPU Search Function ---
def gpu_image_search(image_path: str, top_k: int = 3) -> list:
    """Call GPU API for similarity search"""
    try:
        with open(image_path, "rb") as img_file:
            response = requests.post(
+               f"{GPU_API_URL}/search",
                files={"file": img_file},
                timeout=30
            )
        if response.status_code == 200:
            return response.json().get("indices", [])
        return []
    except Exception as e:
        logger.error(f"GPU search failed: {e}")
        return []

def handle_file_upload(file_type: str, extensions: list) -> Optional[str]:
    """Handle file upload and return temp file path"""
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

def cleanup_files(*file_paths):
    """Safely remove temporary files with error handling"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
                logger.debug(f"Cleaned up temp file: {path}")
            except Exception as e:
                logger.warning(f"Couldn't delete temp file {path}: {e}")
                st.warning(f"Couldn't clean up temporary file: {os.path.basename(path)}")

# --- Modified Display Function ---
def display_results(result: dict, pdf_path: Optional[str], image_path: Optional[str]):
    """Display processing results with image search"""
    with st.container():
        if pdf_path:
            st.info(f"📄 Processed PDF: {os.path.basename(pdf_path)}")
        if image_path:
            st.info(f"🖼 Processed Image: {os.path.basename(image_path)}")
            
            if st.checkbox("🔍 Find similar images"):
                with st.spinner("Searching similar images..."):
                    try:
                        with open(image_path, "rb") as f:
                            response = requests.post(
                                f"{GPU_API_URL}/search",
                                files={"file": f},
                                headers={"Authorization": f"Bearer {API_TOKEN}"},
                                timeout=10
                            )
                        
                        if response.ok:
                            matches = response.json().get("matches", [])
                            if matches:
                                st.subheader("Top Similar Images")
                                cols = st.columns(3)
                                for i, match in enumerate(matches[:3]):
                                    with cols[i]:
                                    # prefix with GPU_API_URL if it's relative
                                        url = match["image_url"]
                                        if url.startswith("/"):
                                            url = f"{GPU_API_URL}{url}"
                                        img_response = requests.get(url, timeout=10)
                                        img = Image.open(io.BytesIO(img_response.content))
                                        st.image(img, caption=f"Similarity: {match['similarity']:.2f}")
                        else:
                            st.warning("Image search service unavailable")
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
        
        if "final_output" in result:
            st.success(result["final_output"])
        else:
            st.warning("Processing completed but no output was generated")
            
        # Debug info (collapsible)
        with st.expander("Debug Information"):
            st.json({
                "timestamp": datetime.now().isoformat(),
                "pdf_processed": bool(pdf_path),
                "image_processed": bool(image_path),
                "result_keys": list(result.keys())
            })

def main():
    setup_ui()
    
    # Input section
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_input("Ask something...", placeholder="Enter your question here")
    with col2:
        lang = st.selectbox("Response Language", ["en", "de", "hi", "fr"], index=0)
    
    # File upload section
    pdf_path = handle_file_upload("PDF", ["pdf"])
    image_path = handle_file_upload("Image", ["png", "jpg", "jpeg"])

    # Processing
    if st.button("Process", type="primary"):
        if not query and not pdf_path and not image_path:
            st.warning("Please provide at least a question or upload a file")
            return
            
        with st.spinner("Processing your request..."):
            try:
                state = {
                    "input": query,
                    "lang": lang,
                    "pdf_path": pdf_path,
                    "image_path": image_path,
                }
                
                # Initialize and execute graph
                graph = create_graph()
                result = graph.invoke(state)
                
                # Display results
                display_results(result, pdf_path, image_path)
                
            except Exception as e:
                logger.exception("Processing failed")
                st.error(f"❌ Processing error: {str(e)}")
            finally:
                cleanup_files(pdf_path, image_path)

if __name__ == "__main__":
    main()