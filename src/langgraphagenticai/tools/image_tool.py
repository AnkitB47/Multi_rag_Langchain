# src/langgraphagenticai/tools/image_tool.py

import os
import logging
import base64
import time
from typing import Optional
from google.generativeai import GenerativeModel
from google.api_core import retry
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Robust image processing with Gemini Vision"""
    
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.webp')
    MAX_SIZE_MB = 10  # 10MB max image size
    MAX_PIXELS = 20_000_000  # ~20MP resolution limit
    
    def __init__(self):
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize with retry and backoff"""
        try:
            return GenerativeModel("gemini-pro-vision")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise RuntimeError("Vision service unavailable")

    def _validate_image(self, image_path: str) -> bool:
        """Comprehensive image validation"""
        try:
            # Check file existence
            if not os.path.exists(image_path):
                logger.error(f"File not found: {image_path}")
                return False
                
            # Check file size
            if os.path.getsize(image_path) > self.MAX_SIZE_MB * 1024 * 1024:
                logger.error(f"Image too large: {os.path.getsize(image_path)} bytes")
                return False
                
            # Check format
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.SUPPORTED_FORMATS:
                logger.error(f"Unsupported format: {ext}")
                return False
                
            # Check image integrity
            with Image.open(image_path) as img:
                if img.size[0] * img.size[1] > self.MAX_PIXELS:
                    logger.error(f"Resolution too high: {img.size}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False

    def _optimize_image(self, image_path: str) -> Optional[bytes]:
        """Resize and compress if needed"""
        try:
            with Image.open(image_path) as img:
                # Downsample if too large
                if img.size[0] * img.size[1] > 4_000_000:  # >4MP
                    img.thumbnail((2000, 2000))
                
                # Convert to JPEG if not already (smaller size)
                if img.format != 'JPEG':
                    img = img.convert('RGB')
                    
                # Save to buffer
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                return buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            return None

    @retry.Retry(
        initial=1.0,
        maximum=10.0,
        multiplier=2.0,
        deadline=30.0,
        predicate=retry.if_exception_type(Exception)
    )
    def _generate_response(self, image_data: bytes, query: str) -> str:
        """Generate response with retry logic"""
        try:
            response = self.model.generate_content(
                [
                    {"mime_type": "image/jpeg", "data": image_data},
                    {"text": f"{query}\n\nRespond concisely under 100 words."}
                ]
            )
            return self._clean_response(response.text)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _clean_response(self, response: str) -> str:
        """Remove Gemini-specific prefixes"""
        prefixes = [
            "In this image",
            "The image shows",
            "Based on the image",
            "From what I can see"
        ]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].lstrip(" ,")
        return response.capitalize()

    def process_image(self, image_path: str, query: str) -> str:
        """Main processing pipeline"""
        try:
            # Validate first
            if not query.strip():
                return "❌ Please ask a question about the image."
                
            if not self._validate_image(image_path):
                return "❌ Invalid image. Please upload a PNG/JPEG under 10MB."
            
            # Optimize image
            image_data = self._optimize_image(image_path)
            if not image_data:
                return "❌ Failed to process image data."
            
            # Generate response
            return self._generate_response(image_data, query)
            
        except Exception as e:
            logger.exception(f"Image processing error: {e}")
            return "❌ Our vision service is temporarily overloaded. Please try again."

# Singleton instance
image_processor = ImageProcessor()

def query_image(query: str, image_path: str) -> str:
    """Public interface"""
    return image_processor.process_image(image_path, query)