# src/langgraphagenticai/graph/chatbot_graph.py

from langgraph.graph import StateGraph
from langgraphagenticai.state.state import GraphState
from langgraphagenticai.nodes.node_runners import (
    run_query_pdf, run_query_image, run_query_search, run_translation
)
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def create_graph() -> StateGraph:
    """Create a robust LangGraph state machine with error handling"""
    
    builder = StateGraph(GraphState)
    
    # Add nodes with enhanced error handling
    builder.add_node("query_pdf", run_query_pdf)
    builder.add_node("query_image", run_query_image)
    builder.add_node("query_search", run_query_search)
    builder.add_node("translate", run_translation)

    # Set conditional edges with fallback logic
    def decide_pdf_next(state: Dict[str, Any]) -> str:
        if state.get("pdf_error"):
            logger.warning("PDF processing failed, skipping to search")
            return "query_search"
        return "query_image" if state.get("image_path") else "query_search"
    
    def decide_image_next(state: Dict[str, Any]) -> str:
        if state.get("image_error"):
            logger.warning("Image processing failed, moving to search")
        return "query_search"
    
    # Configure workflow
    builder.set_entry_point("query_pdf")
    builder.add_conditional_edges("query_pdf", decide_pdf_next)
    builder.add_conditional_edges("query_image", decide_image_next)
    builder.add_edge("query_search", "translate")
    builder.set_finish_point("translate")

    return builder.compile()

def invoke_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """Robust graph invocation with comprehensive error handling"""
    try:
        graph = create_graph()
        result = graph.invoke(state)
        
        # Check for any processing errors
        errors = {
            k: v for k, v in result.items() 
            if k.endswith('_error') and v is not None
        }
        
        if errors:
            logger.warning(f"Completed with errors: {errors}")
            if "final_output" not in result:
                result["final_output"] = (
                    "Processing completed with some errors. "
                    f"Primary error: {next(iter(errors.values()))}"
                )
                
        return result
        
    except Exception as e:
        logger.exception("Graph execution failed")
        return {
            "error": str(e),
            "final_output": "❌ System error occurred during processing. Please try again."
        }