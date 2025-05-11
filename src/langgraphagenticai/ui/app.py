# app.py

import streamlit as st
import tempfile
import os
from langgraphagenticai.graph.chatbot_graph import create_graph
from typing import Optional

# Initialize graph
graph = create_graph()

def cleanup_temp_files(*paths):
    """Clean up temporary files"""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                st.warning(f"Couldn't delete temp file {path}: {e}")

def main():
    st.title("🤖 Multi-RAG LangGraph Agent")
    
    # Input widgets
    query = st.text_input("Ask something...")
    lang = st.selectbox("Response Language", ["en", "de", "hi", "fr"])
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    # Temporary file handling
    pdf_path: Optional[str] = None
    image_path: Optional[str] = None

    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_file.getvalue())
            pdf_path = f.name

    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(image_file.getvalue())
            image_path = f.name

    if st.button("Run"):
        try:
            state = {
                "input": query,
                "lang": lang,
                "pdf_path": pdf_path,
                "image_path": image_path,
            }
            
            result = graph.invoke(state)
            
            # Display results
            if pdf_path:
                st.info(f"📄 PDF processed: {os.path.basename(pdf_path)}")
            if image_path:
                st.info(f"🖼 Image processed: {os.path.basename(image_path)}")
            
            st.success(result.get("final_output", "✅ Operation completed successfully."))
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        finally:
            cleanup_temp_files(pdf_path, image_path)

if __name__ == "__main__":
    main()