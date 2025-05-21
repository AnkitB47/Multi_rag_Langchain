# src/api/main_pdf.py

import os
import uuid
import logging
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

# ─── CONFIG ────────────────────────────────────────────────────────────────────
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_AUTH_TOKEN:
    raise RuntimeError("API_AUTH_TOKEN must be set in the environment")

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_rag_service")

# ─── APP SETUP ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and run Retrieval-Augmented queries over it",
)

# allow any origin for debugging; lock down in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# redirect root to docs
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

# ─── PROCESS ENDPOINT ───────────────────────────────────────────────────────────
@app.post(
    "/process",
    summary="Ingest uploaded PDF & run a RAG query",
    response_model=Dict[str, Any],
)
async def process_pdf(
    query: str = Form(..., description="Your question about the PDF"),
    file: UploadFile = File(..., description="PDF file to ingest"),
):

    # Write upload to disk
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        data = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(data)
        logger.info("Saved upload to %s", tmp_path)
    except Exception:
        logger.exception("Failed to write uploaded PDF")
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to save uploaded PDF"},
        )

    # Lazy-import your RAG functions
    try:
        from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf
    except Exception:
        logger.exception("Failed to import RAG pipeline")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal import error"},
        )

    # Ingest
    try:
        logger.info("Starting ingestion")
        ingest_pdf(tmp_path)
        logger.info("Ingestion complete")
    except Exception:
        logger.exception("Error during ingestion")
        return JSONResponse(
            status_code=500,
            content={"detail": "Error during PDF ingestion"},
        )

    # (5) Query
    try:
        logger.info("Running query: %s", query)
        answer = query_pdf(query)
        logger.info("Query result: %s", answer)
    except Exception:
        logger.exception("Error during query")
        return JSONResponse(
            status_code=500,
            content={"detail": "Error during PDF query"},
        )
    finally:
        # (6) Cleanup
        try:
            os.remove(tmp_path)
            logger.info("Cleaned up %s", tmp_path)
        except Exception:
            logger.warning("Could not delete temp file %s", tmp_path)

    return {"output": answer}


# ─── CUSTOM 404 HANDLER ─────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return JSONResponse(404, {"detail": "Not Found – see /docs for API."})
