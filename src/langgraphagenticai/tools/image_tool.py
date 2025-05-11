# src/langgraphagenticai/tools/image_tool.py

import os
import logging
from langgraphagenticai.LLMS.load_models import load_gemini_vision
import base64

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
def query_image(query: str, image_path: str):
    try:
        model = load_gemini_vision()
        mime_type = "image/png" if image_path.endswith(".png") else "image/jpeg"
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        response = model.generate_content([
            {"mime_type": mime_type, "data": image_base64},
            {"text": query}
        ])
        return response.text
    except Exception as e:
        logger.warning(f"Image processing failed: {e}")
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
