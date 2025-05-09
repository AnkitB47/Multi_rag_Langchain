import streamlit as st
import tempfile
from langgraphagenticai.graph.chatbot_graph import create_graph

# Initialize LangGraph
graph = create_graph()

# UI Title
st.title("🤖 Agentic Multi-RAG Chatbot")

# Input fields
query = st.text_input("💬 Ask your question")
lang = st.selectbox("🌍 Response Language", ["en", "de", "hi", "fr"])
pdf_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
image_file = st.file_uploader("🖼️ Upload an Image", type=["png", "jpg", "jpeg"])

# Handle input
if st.button("Ask"):
    pdf_path, img_path = None, None

    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_file.read())
            pdf_path = tmp_pdf.name

    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(image_file.read())
            img_path = tmp_img.name

    state = {
        "input": query,
        "lang": lang,
        "pdf_path": pdf_path,
        "image_path": img_path
    }

    try:
        result = graph.invoke(state)
        st.success(result.get("final_output", "✅ Done, but no output found."))
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
