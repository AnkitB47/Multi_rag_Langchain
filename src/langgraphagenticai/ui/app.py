import streamlit as st
import tempfile
import os
from langgraphagenticai.graph.chatbot_graph import create_graph

graph = create_graph()

st.title("🤖 Multi-RAG LangGraph Agent")

query = st.text_input("Ask something...")
lang = st.selectbox("Response Language", ["en", "de", "hi", "fr"])
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

pdf_path = image_path = None

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
        st.success(result.get("final_output", "✅ Done."))
    except Exception as e:
        st.error(f"❌ {e}")

    st.write("📄 PDF Path:", pdf_path)
    st.write("🖼 Image Path:", image_path)
