# src/langgraphagenticai/tools/image_tool.py

import os
import logging
import base64
from typing import Optional
from langgraphagenticai.LLMS.load_models import load_gemini_vision

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ImageProcessor:
    def __init__(self):
        self.model = load_gemini_vision()
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.webp')

    def _validate_image(self, image_path: str) -> bool:
        """Check if image exists and is in supported format"""
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return False
        
        if not image_path.lower().endswith(self.supported_formats):
            logger.error(f"Unsupported image format: {image_path}")
            return False
            
        return True

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Convert image to base64 with proper MIME type"""
        try:
            mime_type = self._get_mime_type(image_path)
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None

    def _get_mime_type(self, image_path: str) -> str:
        """Get accurate MIME type for the image"""
        ext = os.path.splitext(image_path)[1].lower()
        return {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')  # default to JPEG

    def query_image(self, query: str, image_path: str) -> str:
        """Process image and generate response with error handling"""
        try:
            # Validate input
            if not query.strip():
                return "❌ Please provide a question about the image."
                
            if not self._validate_image(image_path):
                return "❌ Invalid image file. Please upload a PNG, JPG, or JPEG image."

            # Process image
            image_base64 = self._encode_image(image_path)
            if not image_base64:
                return "❌ Failed to process the image data."

            # Generate response
            response = self.model.generate_content([
                {"mime_type": self._get_mime_type(image_path), "data": image_base64},
                {"text": f"Answer concisely about this image: {query}"}
            ])
            
            return self._clean_response(response.text)
            
        except Exception as e:
            logger.exception(f"Image processing error: {e}")
            return "❌ An error occurred while processing the image. Please try again."

    def _clean_response(self, response: str) -> str:
        """Clean up Gemini's response formatting"""
        # Remove common prefixes
        prefixes = [
            "Based on the image",
            "In this image",
            "The image shows",
            "From what I can see"
        ]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].lstrip(",.: ")
        
        # Capitalize first letter
        return response[0].upper() + response[1:] if response else response

# Singleton instance for better performance
image_processor = ImageProcessor()

def query_image(query: str, image_path: str) -> str:
    """Public interface for image queries"""
    return image_processor.query_image(query, image_path)