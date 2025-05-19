from fastapi import FastAPI, UploadFile, HTTPException, Header
from fastapi.staticfiles import StaticFiles
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import io
from PIL import Image
from pathlib import Path

# Configuration
API_TOKEN = os.getenv("API_AUTH_TOKEN")  # From environment variable
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/data/vector.index")
IMAGE_STORAGE_PATH = os.getenv("IMAGE_STORAGE_PATH", "/data/images")
EMBEDDING_MODEL = "clip-ViT-B-32"

# Initialize FastAPI
app = FastAPI()

# Ensure image directory exists at startup
Path(IMAGE_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGE_STORAGE_PATH), name="images")

# Initialize model and index
model = SentenceTransformer(EMBEDDING_MODEL)

# Load or create FAISS index
try:
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        logging.info(f"Loaded existing index from {INDEX_PATH}")
    else:
        index = faiss.IndexFlatIP(512)  # CLIP default dimension
        faiss.write_index(index, INDEX_PATH)
        logging.info(f"Created new index at {INDEX_PATH}")
except Exception as e:
    logging.error(f"Index initialization failed: {e}")
    raise RuntimeError(f"Could not initialize FAISS index: {e}")

@app.post("/search")
async def search_images(
    file: UploadFile,
    top_k: int = 3,
    authorization: str = Header(None)
):
    """Search for similar images using CLIP embeddings"""
    # Authentication
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        # Process image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        embedding = model.encode(img, convert_to_tensor=False)
        
        # Search index
        distances, indices = index.search(np.array([embedding]), top_k)
        
        # Format results
        return {
            "matches": [
                {
                    "image_id": int(idx),
                    "similarity": float(dist),
                    "image_url": f"/images/{idx}.jpg",
                    "file_exists": os.path.exists(f"{IMAGE_STORAGE_PATH}/{idx}.jpg")
                }
                for idx, dist in zip(indices[0], distances[0])
                if idx >= 0  # Skip invalid indices
            ]
        }
    except Exception as e:
        logging.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image processing error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "index_size": index.ntotal,
        "images_dir_exists": os.path.exists(IMAGE_STORAGE_PATH)
    }