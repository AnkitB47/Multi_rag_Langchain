from langgraphagenticai.LLMS.load_models import load_gemini_vision
import logging

logger = logging.getLogger(__name__)

def query_image(query: str, image_path: str):
    try:
        model = load_gemini_vision()
        mime_type = "image/png" if image_path.endswith(".png") else "image/jpeg"
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = model.generate_content([
            {"mime_type": mime_type, "data": image_bytes},
            {"text": query}
        ])
        return response.text
    except Exception as e:
        logger.warning(f"Image processing failed: {e}")
        return "❌ Failed to process image. Please try again."
