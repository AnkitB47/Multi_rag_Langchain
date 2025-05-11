# src/langgraphagenticai/graph/chatbot_graph.py

from langgraph.graph import StateGraph
from langgraphagenticai.state.state import GraphState
from langgraphagenticai.nodes.node_runners import (
    run_query_pdf, run_query_image, run_query_search, run_translation
)
from typing import Dict, Any

def create_graph() -> StateGraph:
    """Create the LangGraph state machine with enhanced error handling"""
    
    # Define the workflow
    builder = StateGraph(GraphState)
    
    # Add nodes with conditional execution
    builder.add_node("query_pdf", lambda state: run_query_pdf(state) if state.get("pdf_path") else state)
    builder.add_node("query_image", lambda state: run_query_image(state) if state.get("image_path") else state)
    builder.add_node("query_search", run_query_search)
    builder.add_node("translate", run_translation)

    # Set up the workflow
    builder.set_entry_point("query_pdf")
    
    # Conditional edges
    builder.add_conditional_edges(
        "query_pdf",
        lambda state: "query_image" if state.get("image_path") else "query_search"
    )
    builder.add_conditional_edges(
        "query_image",
        lambda state: "query_search"
    )
    builder.add_edge("query_search", "translate")
    
    builder.set_finish_point("translate")

    return builder.compile()

def invoke_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """Safe invocation wrapper for the graph"""
    try:
        graph = create_graph()
        return graph.invoke(state)
    except Exception as e:
        return {"error": str(e), "final_output": f"❌ Processing failed: {str(e)}"}