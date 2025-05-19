from fastapi import FastAPI, UploadFile, HTTPException, Header, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import io
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────
API_TOKEN           = os.getenv("API_AUTH_TOKEN")
if not API_TOKEN:
    logger.error("Missing API_AUTH_TOKEN")
    raise RuntimeError("API_AUTH_TOKEN must be set")

INDEX_PATH          = os.getenv("FAISS_INDEX_PATH", "/data/vector.index")
IMAGE_STORAGE_PATH  = os.getenv("IMAGE_STORAGE_PATH", "/data/images")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "clip-ViT-B-32")

# ─── Ensure storage dirs ───────────────────────────────────
Path(IMAGE_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
os.chmod(IMAGE_STORAGE_PATH, 0o755)

# ─── FastAPI setup ────────────────────────────────────────
app = FastAPI(
    title="Image Similarity Search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    description="Find visually similar images via CLIP + FAISS"
)
app.mount("/images", StaticFiles(directory=IMAGE_STORAGE_PATH), name="images")

# ─── Load model + index ────────────────────────────────────
logger.info("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

if os.path.exists(INDEX_PATH):
    logger.info(f"Loading FAISS index from {INDEX_PATH}…")
    index = faiss.read_index(INDEX_PATH)
else:
    logger.info("Creating new FAISS index (empty)…")
    index = faiss.IndexIDMap(faiss.IndexFlatIP(512))
    faiss.write_index(index, INDEX_PATH)

logger.info(f"Index contains {index.ntotal} vectors")

# ─── Root & health endpoints ──────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"service": "image-search", "status": "ok"})

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "index_size": index.ntotal,
        "images_dir": IMAGE_STORAGE_PATH
    }

# ─── Search endpoint ───────────────────────────────────────
@app.post("/search", response_model=List[Dict[str, Any]])
async def search_images(
    file: UploadFile,
    top_k: int = 3,
    authorization: str = Header(..., alias="Authorization")
):
    # Auth
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    if not (1 <= top_k <= 100):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="top_k must be 1–100")

    # Read & embed
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    emb = model.encode(img, convert_to_tensor=False).astype(np.float32).reshape(1, -1)

    # FAISS search
    dists, idxs = index.search(emb, top_k)

    results = []
    for dist, idx in zip(dists[0], idxs[0]):
        if idx < 0:  # no match
            continue
        img_path = os.path.join(IMAGE_STORAGE_PATH, f"{idx}.jpg")
        results.append({
            "image_id":    int(idx),
            "similarity":  float(dist),
            "image_url":   f"/images/{idx}.jpg",
            "file_exists": os.path.exists(img_path)
        })

    return results
