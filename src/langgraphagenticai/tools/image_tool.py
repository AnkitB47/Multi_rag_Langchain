from langgraphagenticai.LLMS.load_models import load_gemini_vision
from langgraphagenticai.tools.search_tool import query_search
import logging

logger = logging.getLogger(__name__)

def query_image(query: str, image_path: str):
    try:
        model = load_gemini_vision()
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = model.generate_content([
            {"mime_type": "image/png", "data": image_bytes},
            {"text": query}
        ])
        return response.text
    except Exception as e:
        logger.warning(f"Image query failed. Falling back to web: {e}")
        return query_search(query)
