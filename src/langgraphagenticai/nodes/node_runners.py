from langgraphagenticai.tools.pdf_tool import query_pdf
from langgraphagenticai.tools.image_tool import query_image
from langgraphagenticai.tools.search_tool import query_search
from langgraphagenticai.tools.translate_tool import translate_text

def run_query_pdf(state: dict):
    if state.get("pdf_path"):
        response = query_pdf(state["input"], state["pdf_path"])
        if "❌" not in response:
            return {**state, "pdf_result": response}
    return state

def run_query_image(state: dict):
    if state.get("image_path"):
        response = query_image(state["input"], state["image_path"])
        if "❌" not in response:
            return {**state, "image_result": response}
    return state

def run_query_search(state: dict):
    response = query_search(state["input"])
    return {**state, "search_result": response}

def run_translation(state: dict):
    base_response = state.get("pdf_result") or state.get("image_result") or state.get("search_result")
    if not base_response:
        return {**state, "final_output": "❌ No relevant result found from any source."}
    if state.get("lang") != "en":
        translated = translate_text(base_response, state["lang"])
        return {**state, "final_output": translated}
    return {**state, "final_output": base_response}
