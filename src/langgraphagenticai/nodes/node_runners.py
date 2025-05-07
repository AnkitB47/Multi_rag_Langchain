from langgraphagenticai.tools.pdf_tool import query_pdf
from langgraphagenticai.tools.image_tool import query_image
from langgraphagenticai.tools.search_tool import query_search
from langgraphagenticai.tools.translate_tool import translate_text

def run_query_search(state):
    response = query_search(state.input)
    return {**state.dict(), "search_result": response}

def run_query_pdf(state):
    if state.pdf_path:
        pdf_response = query_pdf(state.input, state.pdf_path)
        return {**state.dict(), "pdf_result": pdf_response}
    return state.dict()

def run_query_image(state):
    if state.image_path:
        img_response = query_image(state.input, state.image_path)
        return {**state.dict(), "image_result": img_response}
    return state.dict()

def run_translation(state):
    base_response = state.pdf_result or state.image_result or state.search_result
    if state.lang != "en":
        translated = translate_text(base_response, state.lang)
        return {**state.dict(), "final_output": translated}
    return {**state.dict(), "final_output": base_response}
