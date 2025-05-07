import streamlit as st
from langgraphagenticai.graph.chatbot_graph import create_graph

graph = create_graph()

st.title("🧠 Agentic Multi-RAG Chatbot")
query = st.text_input("💬 Ask a question")
lang = st.selectbox("🌍 Language", ["en", "de", "hi", "fr"])
pdf_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
image_file = st.file_uploader("🖼️ Upload an Image", type=["png", "jpg"])

if st.button("Ask"):
    pdf_path, img_path = None, None
    if pdf_file:
        pdf_path = f"/tmp/{pdf_file.name}"
        with open(pdf_path, "wb") as f: f.write(pdf_file.read())

    if image_file:
        img_path = f"/tmp/{image_file.name}"
        with open(img_path, "wb") as f: f.write(image_file.read())

    state = {
        "input": query,
        "lang": lang,
        "pdf_path": pdf_path,
        "image_path": img_path
    }

    output = graph.invoke(state)
    st.success(output['final_output'])
