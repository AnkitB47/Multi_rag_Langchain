import streamlit as st
import tempfile
from langgraphagenticai.graph.chatbot_graph import create_graph
from langgraphagenticai.agentic.agno_team import load_agno_team
from langgraphagenticai.agentic.phi_team import load_phi_team

# Initialize LangGraph
graph = create_graph()

st.set_page_config(page_title="🤖 Multi-Agent Chatbot", layout="centered")
st.title("🤖 Multi-RAG Chatbot (LangGraph + Agno + Phi)")

backend = st.selectbox("🧠 Choose AI Agent Backend", ["LangGraph", "Agno", "Phi"])
query = st.text_input("💬 Ask your question")
lang = st.selectbox("🌍 Response Language", ["en", "de", "hi", "fr"])
pdf_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
image_file = st.file_uploader("🖼️ Upload an Image", type=["png", "jpg", "jpeg"])

pdf_path = image_path = None
if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

if image_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_file.read())
        image_path = tmp.name

if st.button("🚀 Run Query"):
    try:
        if backend == "LangGraph":
            state = {
                "input": query,
                "lang": lang,
                "pdf_path": pdf_path,
                "image_path": image_path
            }
            result = graph.invoke(state)
            st.success(result.get("final_output", "✅ LangGraph response complete."))

        elif backend == "Agno":
            agno = load_agno_team()
            response = agno.run(query)
            st.markdown(response.content)

        elif backend == "Phi":
            phi = load_phi_team()
            response = phi.run(query)
            st.markdown(response.content)

    except Exception as e:
        st.error(f"❌ Error with {backend} backend: {e}")
