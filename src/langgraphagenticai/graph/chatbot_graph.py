# src/langgraphagenticai/graph/chatbot_graph.py

import logging
from langgraph.graph import StateGraph
from langgraphagenticai.state.state import GraphState
from langgraphagenticai.nodes.node_runners import (
    run_query_pdf,     # calls your PDF-RAG tool
    run_query_image,   # calls your Image tool
    run_query_search,  # calls your Arxiv/Web search tool
    run_translation    # handles optional translate
)

logger = logging.getLogger(__name__)

def create_pdf_graph() -> StateGraph:
    """
    PDF RAG flow:
      query_pdf
        ├─(success)→ translate
        └─(error)──→ query_search → translate
    """
    g = StateGraph(GraphState)

    # Nodes
    g.add_node("query_pdf",    run_query_pdf)
    g.add_node("query_search", run_query_search)
    g.add_node("translate",    run_translation)

    # Entry point
    g.set_entry_point("query_pdf")

    # From PDF node: on error → search; else → translate
    def next_after_pdf(state):
        return "query_search" if state.get("pdf_error") else "translate"

    g.add_conditional_edges("query_pdf", next_after_pdf)

    # If we fell back to search, then translate
    g.add_edge("query_search", "translate")

    # Finish at translate
    g.set_finish_point("translate")

    return g.compile()


def create_image_graph() -> StateGraph:
    """
    Image‐only search flow:
      query_image
        ├─(success)→ translate
        └─(error)──→ query_search → translate
    """
    g = StateGraph(GraphState)

    # Nodes
    g.add_node("query_image",  run_query_image)
    g.add_node("query_search", run_query_search)
    g.add_node("translate",    run_translation)

    # Entry point
    g.set_entry_point("query_image")

    # From image node: on error → search; else → translate
    def next_after_image(state):
        return "query_search" if state.get("image_error") else "translate"

    g.add_conditional_edges("query_image", next_after_image)

    # Fallback search into translate
    g.add_edge("query_search", "translate")

    # Finish at translate
    g.set_finish_point("translate")

    return g.compile()
