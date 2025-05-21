# src/api/main_pdf.py
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

# ─── CONFIG ──────────────────────────────────────────────────────────────────
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_AUTH_TOKEN:
    raise RuntimeError("API_AUTH_TOKEN must be set in the environment")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_rag_service")

# ─── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and run Retrieval-Augmented queries over it",
)

# Allow CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

# Redirect root → docs
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.get("/health", summary="Health check")
async def health() -> Dict[str,str]:
    return {"status":"ok"}

@app.post(
    "/process",
    summary="Ingest uploaded PDF & run a RAG query",
    response_model=Dict[str,Any]
)
async def process_pdf(
    query: str = Form(..., description="Your question about the PDF"),
    file: UploadFile = File(..., description="PDF file to ingest"),
):
    # save file
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        data = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(data)
    except Exception as e:
        logger.error("Failed to write upload: %s", e, exc_info=True)
        raise HTTPException(500, "Could not save uploaded PDF")

    # lazy import so startup is fast
    try:
        from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf
    except ImportError as e:
        logger.error("Import error: %s", e, exc_info=True)
        raise HTTPException(500, "Internal server error")

    # ingest + query
    try:
        ingest_pdf(tmp_path)
        answer = query_pdf(query)
    except Exception as e:
        logger.error("RAG pipeline error: %s", e, exc_info=True)
        raise HTTPException(500, "Error processing PDF")

    # cleanup
    try:
        os.remove(tmp_path)
    except OSError:
        logger.warning("Could not delete temp file %s", tmp_path)

    return {"output": answer}

# Custom 404 → point at docs
@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return JSONResponse(404, {"detail":"Not Found – see /docs for API."})
