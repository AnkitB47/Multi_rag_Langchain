# app.py

import streamlit as st
import tempfile
import os
from langgraphagenticai.graph.chatbot_graph import create_graph
from typing import Optional, Tuple
import logging
from datetime import datetime

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

def display_results(result: dict, pdf_path: Optional[str], image_path: Optional[str]):
    """Display processing results"""
    with st.container():
        if pdf_path:
            st.info(f"📄 Processed PDF: {os.path.basename(pdf_path)}")
        if image_path:
            st.info(f"🖼 Processed Image: {os.path.basename(image_path)}")
        
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