from fastapi import FastAPI, UploadFile, HTTPException, Header
from fastapi.staticfiles import StaticFiles
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import io
from PIL import Image

# Configuration
API_TOKEN = os.getenv("API_AUTH_TOKEN")  # From environment variable
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/data/vector.index")
EMBEDDING_MODEL = "clip-ViT-B-32"

app = FastAPI()
app.mount("/images", StaticFiles(directory="/data/images"), name="images")
model = SentenceTransformer(EMBEDDING_MODEL)

# Load FAISS index
try:
    index = faiss.read_index(INDEX_PATH)
except Exception as e:
    logging.error(f"Index load failed: {e}")
    index = faiss.IndexFlatIP(512)  # CLIP default dimension
    faiss.write_index(index, INDEX_PATH)

@app.post("/search")
async def search_images(
    file: UploadFile,
    top_k: int = 3,
    authorization: str = Header(None)
):
    # Authentication
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        # Process image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        embedding = model.encode(img, convert_to_tensor=False)
        
        # Search
        distances, indices = index.search(np.array([embedding]), top_k)
        
        return {
            "matches": [
                {
                    "image_id": int(idx),
                    "similarity": float(dist),
                    "image_url": f"/images/{idx}.jpg"  # Your storage path
                }
                for idx, dist in zip(indices[0], distances[0])
            ]
        }
    except Exception as e:
        logging.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Processing error")