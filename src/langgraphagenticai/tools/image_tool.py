import os
import logging
from typing import List, Union
from google.generativeai import GenerativeModel
from google.api_core import retry as gp_retry
from PIL import Image
from langgraphagenticai.utils.image_utils import (
    get_clip_model,
    validate_image,
    optimize_image,
    create_faiss_index,
    load_faiss_index,
    clean_gemini_response
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────

VISION_MODEL_NAME = "gemini-pro-vision"
DEFAULT_INDEX_PATH = os.getenv("IMAGE_INDEX_PATH", "/data/image.index")
DEFAULT_IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "/data/images")

# ── Spot instances often need backoff ─────────────────

@gp_retry.Retry(
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=30.0,
    predicate=gp_retry.if_exception_type(Exception)
)
def _generate_vision(vision_model: GenerativeModel, image_bytes: bytes, query: str) -> str:
    resp = vision_model.generate_content([
        {"mime_type": "image/jpeg", "data": image_bytes},
        {"text": f"{query}\n\nPlease respond concisely under 100 words."}
    ])
    return clean_gemini_response(resp.text)

# ── Core classes ─────────────────────────────────────

class ImageProcessor:
    """Handle single‐image Q&A via Gemini Vision + FAISS search fallback."""

    def __init__(self):
        # init vision model
        try:
            self.vision = GenerativeModel(VISION_MODEL_NAME)
        except Exception as e:
            logger.error("Vision init failed", exc_info=True)
            raise RuntimeError("Could not initialize vision model")

        # init or load FAISS for similarity search
        idx_path = DEFAULT_INDEX_PATH
        if os.path.exists(idx_path):
            idx, paths = load_faiss_index(idx_path)
        else:
            idx, paths = create_faiss_index(DEFAULT_IMAGE_FOLDER, idx_path)
        self.index = idx
        self.paths = paths

    def describe(self, image_path: str, query: str) -> str:
        """
        Run Gemini Vision Q&A on the image.
        """
        if not query.strip():
            return "❌ Please ask a question about the image."

        if not validate_image(image_path):
            return "❌ Invalid image; must be JPEG/PNG under 10 MB."

        data = optimize_image(image_path)
        if not data:
            return "❌ Failed to preprocess image."

        try:
            return _generate_vision(self.vision, data, query)
        except Exception as e:
            logger.exception("Vision call failed")
            return "❌ Vision service temporarily unavailable."

    def similar(self, image_path: str, top_k: int = 3) -> List[str]:
        """
        Return file‐paths of top_k visually‐similar images.
        """
        if not validate_image(image_path):
            raise ValueError("Invalid image for similarity search")

        data = optimize_image(image_path)
        if not data:
            raise RuntimeError("Could not preprocess image for search")

        model = get_clip_model()
        feat = model.encode(Image.open(image_path)).astype('float32').reshape(1, -1)
        D, I = self.index.search(feat, top_k)
        return [ self.paths[int(i)] for i in I[0] if i >= 0 ]


# ── Public API ───────────────────────────────────────

_processor: ImageProcessor = None

def _get_processor() -> ImageProcessor:
    global _processor
    if _processor is None:
        _processor = ImageProcessor()
    return _processor

def query_image(query: str, image_path: str) -> str:
    """
    For your node_runner: describe what's in the image.
    """
    return _get_processor().describe(image_path, query)

def search_similar_images(image_path: str, top_k: int = 3) -> List[str]:
    """
    For your image‐search node: return similar image paths.
    """
    return _get_processor().similar(image_path, top_k)

def initialize_image_search(image_folder: str, index_path: str) -> bool:
    """
    Manually rebuild or load the FAISS index.
    """
    try:
        _, _ = (create_faiss_index if not os.path.exists(index_path) else load_faiss_index)(
            image_folder, index_path
        )
        return True
    except Exception:
        return False
