# src/langgraphagenticai/state/state.py

from typing import TypedDict, Optional, Literal
from typing_extensions import NotRequired  # For Python < 3.11

class GraphState(TypedDict):
    """
    Enhanced state definition for the LangGraph workflow with:
    - Strict typing
    - Error state tracking
    - Documentation
    """
    # Input fields
    input: str
    lang: Literal["en", "de", "hi", "fr"]
    pdf_path: NotRequired[Optional[str]]
    image_path: NotRequired[Optional[str]]
    
    # Processing results
    pdf_result: NotRequired[Optional[str]]
    image_result: NotRequired[Optional[str]]
    search_result: NotRequired[Optional[str]]
    final_output: NotRequired[Optional[str]]
    
    # Error states (added for enhanced error handling)
    pdf_error: NotRequired[Optional[str]]
    image_error: NotRequired[Optional[str]]
    search_error: NotRequired[Optional[str]]
    translation_error: NotRequired[Optional[str]]
    
    # System metadata (optional)
    processing_time: NotRequired[Optional[float]]
    current_node: NotRequired[Optional[str]]