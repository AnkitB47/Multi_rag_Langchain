from langgraphagenticai.graph.chatbot_graph import create_graph

class MultiRAGTool:
    name = "MultiRAGLangGraph"
    description = "RAG chain that queries PDF, image, and web in a pipeline."

    def __init__(self):
        self.graph = create_graph()

    def run(self, query, lang="en", pdf_path=None, image_path=None):
        state = {
            "input": query,
            "lang": lang,
            "pdf_path": pdf_path,
            "image_path": image_path
        }
        try:
            result = self.graph.invoke(state)
            return result.get("final_output", "✅ Done but no output.")
        except Exception as e:
            return f"❌ LangGraph tool failed: {e}"
