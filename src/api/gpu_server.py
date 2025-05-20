# src/api/gpu_server.py

import os
import io
import logging
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Dict
from PIL import Image
from langgraphagenticai.tools.image_tool import query_image, search_similar_images

# ── Logging ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gpu_server")

# ── Configuration ───────────────────────────────────────
API_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_TOKEN:
    logger.error("API_AUTH_TOKEN is not set")
    raise RuntimeError("API_AUTH_TOKEN environment variable must be set")

# ── FastAPI setup ───────────────────────────────────────
app = FastAPI(
    title="GPU Image Service",
    version="1.0",
    description="Image Q&A (Gemini Vision) and FAISS similarity search",
    docs_url="/docs",
    redoc_url=None
)

@app.middleware("http")
async def check_auth(request, call_next):
    # All endpoints require Bearer token
    auth = request.headers.get("authorization")
    if auth != f"Bearer {API_TOKEN}":
        return JSONResponse(
            {"detail": "Invalid or missing authorization token"},
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    return await call_next(request)

# ── /describe endpoint ───────────────────────────────────
@app.post("/describe", summary="Ask a question about an image")
async def describe_image(
    file: UploadFile = File(..., description="Your image file"),
    query: str      = Form(..., description="Your free-form question about the image")
) -> Dict[str, str]:
    """
    Returns a concise answer about the contents of the image,
    powered by Gemini Vision (with retry/backoff).
    """
    # save to temp
    tmp_path = f"/tmp/{file.filename}"
    data = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(data)

    try:
        answer = query_image(query, tmp_path)
        return {"description": answer}
    except Exception as e:
        logger.exception("describe_image failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Image description failed")

# ── /search endpoint ─────────────────────────────────────
@app.post("/search", summary="Find visually similar images")
async def find_similar(
    file: UploadFile  = File(..., description="Your image file"),
    top_k: int       = Form(3, ge=1, le=20, description="How many matches to return (1–20)")
) -> Dict[str, List[str]]:
    """
    Returns a list of file-paths (or URLs) of the top_k images
    in your FAISS index most similar to the uploaded file.
    """
    tmp_path = f"/tmp/{file.filename}"
    data = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(data)

    try:
        matches = search_similar_images(tmp_path, top_k)
        return {"matches": matches}
    except ValueError as e:
        # invalid image format / validation failure
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("find_similar failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Similarity search failed")

# ── Health & Root ───────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"service": "gpu-image", "status": "ok"}

@app.get("/health", summary="Service health check")
async def health():
    return {"status": "healthy"}
