# src/langgraphagenticai/tools/image_tool.py

import os
import logging
import base64
import time
from typing import Optional, List, Union
from google.generativeai import GenerativeModel
from google.api_core import retry
from PIL import Image
import io
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageSearchEngine:
    """CLIP + FAISS based image similarity search engine"""
    
    def __init__(self):
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        self.index = None
        self.image_paths = []
        
    def create_index(self, image_folder: str, index_path: str):
        """Create FAISS index from images in folder"""
        try:
            # Get all image paths
            self.image_paths = self._get_image_paths(image_folder)
            if not self.image_paths:
                raise ValueError("No images found in the specified folder")
                
            # Generate embeddings
            embeddings = []
            for img_path in self.image_paths:
                image = Image.open(img_path)
                embedding = self.clip_model.encode(image)
                embeddings.append(embedding)
                
            # Create FAISS index
            dimension = len(embeddings[0])
            index = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIDMap(index)
            
            vectors = np.array(embeddings).astype(np.float32)
            index.add_with_ids(vectors, np.array(range(len(embeddings))))
            
            # Save index and paths
            faiss.write_index(index, index_path)
            with open(index_path + '.paths', 'w') as f:
                for path in self.image_paths:
                    f.write(path + '\n')
                    
            self.index = index
            return True
            
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return False
            
    def load_index(self, index_path: str):
        """Load existing FAISS index"""
        try:
            self.index = faiss.read_index(index_path)
            with open(index_path + '.paths', 'r') as f:
                self.image_paths = [line.strip() for line in f]
            return True
        except Exception as e:
            logger.error(f"Index loading failed: {e}")
            return False
            
    def search(self, query: Union[str, Image.Image], top_k: int = 3) -> List[str]:
        """Search for similar images"""
        if not self.index:
            raise ValueError("Index not initialized")
            
        if isinstance(query, str) and query.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            query = Image.open(query)
            
        query_features = self.clip_model.encode(query)
        query_features = query_features.astype(np.float32).reshape(1, -1)
        
        distances, indices = self.index.search(query_features, top_k)
        return [self.image_paths[int(idx)] for idx in indices[0]]
        
    def _get_image_paths(self, folder: str) -> List[str]:
        """Get all image paths from folder recursively"""
        return [
            os.path.join(root, file)
            for root, _, files in os.walk(folder)
            for file in files
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]

class ImageProcessor:
    """Robust image processing with Gemini Vision and CLIP+FAISS search"""
    
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.webp')
    MAX_SIZE_MB = 10  # 10MB max image size
    MAX_PIXELS = 20_000_000  # ~20MP resolution limit
    
    def __init__(self):
        self.vision_model = self._initialize_vision_model()
        self.search_engine = ImageSearchEngine()
        
    def _initialize_vision_model(self):
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
    def _generate_vision_response(self, image_data: bytes, query: str) -> str:
        """Generate response with retry logic"""
        try:
            response = self.vision_model.generate_content(
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

    def initialize_search(self, image_folder: str, index_path: str) -> bool:
        """Initialize the image search engine"""
        try:
            if not os.path.exists(index_path):
                return self.search_engine.create_index(image_folder, index_path)
            return self.search_engine.load_index(index_path)
        except Exception as e:
            logger.error(f"Search engine initialization failed: {e}")
            return False

    def search_similar_images(self, query: Union[str, Image.Image], top_k: int = 3) -> List[str]:
        """Search for visually similar images"""
        try:
            return self.search_engine.search(query, top_k)
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []

    def process_image(self, image_path: str, query: str) -> str:
        """Main processing pipeline for image understanding"""
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
            return self._generate_vision_response(image_data, query)
            
        except Exception as e:
            logger.exception(f"Image processing error: {e}")
            return "❌ Our vision service is temporarily overloaded. Please try again."

# Singleton instances
image_processor = ImageProcessor()
image_search_engine = ImageSearchEngine()

def query_image(query: str, image_path: str) -> str:
    """Public interface for image understanding"""
    return image_processor.process_image(image_path, query)

def search_similar_images(query: Union[str, Image.Image], top_k: int = 3) -> List[str]:
    """Public interface for image similarity search"""
    return image_search_engine.search(query, top_k)

def initialize_image_search(image_folder: str, index_path: str) -> bool:
    """Initialize the image search system"""
    return image_processor.initialize_search(image_folder, index_path)