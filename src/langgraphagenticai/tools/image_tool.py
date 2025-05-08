from langgraphagenticai.LLMS.load_models import load_gemini_vision
from langgraphagenticai.tools.search_tool import query_search

def query_image(query, image_path):
    try:
        model = load_gemini_vision()
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = model.generate_content([
            {"mime_type": "image/png", "data": image_bytes},
            {"text": query}
        ])
        return response.text
    except Exception:
        return query_search(query)
