# src/langgraphagenticai/nodes/node_runners.py

from langgraphagenticai.tools.pdf_tool import query_pdf
from langgraphagenticai.tools.image_tool import query_image
from langgraphagenticai.tools.search_tool import query_search
from langgraphagenticai.tools.translate_tool import translate_text
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def run_query_pdf(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced PDF query runner with better error handling"""
    try:
        if state.get("pdf_path"):
            logger.info(f"Processing PDF at: {state['pdf_path']}")
            response = query_pdf(state["input"], state["pdf_path"])
            
            # Validate response before returning
            if not response or isinstance(response, Exception):
                raise ValueError("PDF processing returned invalid response")
                
            return {**state, "pdf_result": response}
        return state
    except Exception as e:
        logger.error(f"PDF query failed: {str(e)}")
        return {**state, "pdf_error": str(e)}

def run_query_image(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced image query runner"""
    try:
        if state.get("image_path"):
            logger.info(f"Processing image at: {state['image_path']}")
            response = query_image(state["input"], state["image_path"])
            
            if not response or isinstance(response, Exception):
                raise ValueError("Image processing returned invalid response")
                
            return {**state, "image_result": response}
        return state
    except Exception as e:
        logger.error(f"Image query failed: {str(e)}")
        return {**state, "image_error": str(e)}

def run_query_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced web search runner"""
    try:
        logger.info(f"Executing search query: {state['input']}")
        response = query_search(state["input"])
        
        if not response:
            raise ValueError("Search returned no results")
            
        return {**state, "search_result": response}
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return {**state, "search_error": str(e)}

def run_translation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced translation runner with fallback logic"""
    try:
        # Determine base content with fallback logic
        base = (
            state.get("pdf_result") or 
            state.get("image_result") or 
            state.get("search_result") or
            "No content available for translation"
        )
        
        # Only translate if needed and content exists
        if state.get("lang") != "en" and base != "No content available for translation":
            logger.info(f"Translating to {state['lang']}")
            translated = translate_text(base, state["lang"])
            return {**state, "final_output": translated}
            
        return {**state, "final_output": base}
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return {**state, "translation_error": str(e), "final_output": base}