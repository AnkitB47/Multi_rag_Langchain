import os
import io
import logging
from typing import List, Union, Tuple
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────

SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.webp')
MAX_SIZE_MB      = 10        # 10 MB
MAX_PIXELS       = 20_000_000  # ~20 MP
CLIP_MODEL       = 'clip-ViT-B-32'

# ── Paths & embedding model singletons ───────────────

_clip = None
def get_clip_model() -> SentenceTransformer:
    global _clip
    if _clip is None:
        _clip = SentenceTransformer(CLIP_MODEL)
    return _clip

# ── Image I/O & validation ────────────────────────────

def get_image_paths(folder: str) -> List[str]:
    """Recursively find supported images under `folder`."""
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(SUPPORTED_FORMATS):
                paths.append(os.path.join(root, f))
    return paths

def validate_image(path: str) -> bool:
    """Check existence, size, format, resolution."""
    try:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return False
        size = os.path.getsize(path)
        if size > MAX_SIZE_MB * 1024**2:
            logger.error(f"Image too big ({size} bytes)")
            return False
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: {ext}")
            return False
        with Image.open(path) as img:
            w, h = img.size
            if w * h > MAX_PIXELS:
                logger.error(f"Resolution too high: {w}×{h}")
                return False
        return True
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

def optimize_image(path: str) -> Union[bytes, None]:
    """
    Downsample if >4 MP, convert to JPEG, return raw bytes.
    """
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w * h > 4_000_000:
                img.thumbnail((2000, 2000))
            if img.format != 'JPEG':
                img = img.convert('RGB')
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            return buf.getvalue()
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return None

# ── FAISS index helpers ───────────────────────────────

def _make_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    flat = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(flat)
    return index

def create_faiss_index(image_folder: str, index_path: str) -> Tuple[faiss.Index, List[str]]:
    """
    Walk `image_folder`, embed each image, build & save a FAISS index to `index_path`.
    Returns *(index, image_paths)*.
    """
    paths = get_image_paths(image_folder)
    if not paths:
        raise ValueError("No images found in " + image_folder)

    model = get_clip_model()
    embs = []
    for p in paths:
        img = Image.open(p)
        embs.append(model.encode(img).astype('float32'))
    embs = np.stack(embs, axis=0)
    idx = _make_index(embs)
    ids = np.arange(len(paths), dtype='int64')
    idx.add_with_ids(embs, ids)

    faiss.write_index(idx, index_path)
    with open(index_path + '.paths', 'w') as f:
        for p in paths:
            f.write(p + "\n")
    return idx, paths

def load_faiss_index(index_path: str) -> Tuple[faiss.Index, List[str]]:
    """
    Load a saved FAISS index from `index_path`, and its `.paths` file.
    """
    idx = faiss.read_index(index_path)
    with open(index_path + '.paths') as f:
        paths = [line.strip() for line in f]
    return idx, paths

# ── Response cleaning ────────────────────────────────

def clean_gemini_response(text: str) -> str:
    """
    Strip common Gemini prefixes from generated text.
    """
    prefixes = [
        "In this image", "The image shows", "Based on the image",
        "From what I can see"
    ]
    for pre in prefixes:
        if text.startswith(pre):
            text = text[len(pre):].lstrip(" ,")
    return text.capitalize()
