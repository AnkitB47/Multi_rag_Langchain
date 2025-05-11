# src/langgraphagenticai/tools/image_tool.py

import os
import logging
from langgraphagenticai.LLMS.load_models import load_gemini_vision

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def query_image(query: str, image_path: str) -> str:
    """
    Uses Gemini Vision to analyze the uploaded image and respond to the query.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        logger.info(f"🖼️ Received image: {image_path}")
        mime_type = _get_mime_type(image_path)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        if not image_bytes:
            raise ValueError("Image file appears empty.")

        model = load_gemini_vision()
        response = model.generate_content([
            {"mime_type": mime_type, "data": image_bytes},
            {"text": query}
        ])

        if hasattr(response, "text") and response.text.strip():
            return response.text.strip()

        raise ValueError("Gemini returned an empty or malformed response.")

    except Exception as e:
        logger.exception(f"❌ Failed to process image: {e}")
        return "❌ Failed to process image. Please try again."


def _get_mime_type(image_path: str) -> str:
    """
    Determines the correct MIME type for the image.
    """
    ext = image_path.lower().split(".")[-1]
    if ext in ("png",):
        return "image/png"
    elif ext in ("jpg", "jpeg"):
        return "image/jpeg"
    else:
        logger.warning(f"⚠️ Unknown image extension: .{ext} — defaulting to JPEG")
        return "image/jpeg"
