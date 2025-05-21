# src/api/main_pdf.py

import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

# ─── CONFIG ────────────────────────────────────────────────────────────────────
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_AUTH_TOKEN:
    raise RuntimeError("API_AUTH_TOKEN must be set")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_rag_service")

# ─── APP & CORS ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and run Retrieval-Augmented queries over it",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─── ROOT REDIRECT → DOCS ──────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

# ─── PDF INGEST & QUERY ENDPOINT ───────────────────────────────────────────────
@app.post(
    "/process",
    summary="Ingest uploaded PDF & run a RAG query",
    response_model=Dict[str, Any],
)
async def process_pdf(
    query: str = Form(..., description="Your question about the PDF"),
    file: UploadFile = File(..., description="PDF file to ingest"),
):
    # 1) Save the upload
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        data = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(data)
    except Exception as e:
        logger.exception("Failed to write uploaded PDF")
        raise HTTPException(status_code=500, detail=f"I/O error: {e}")

    # 2) Lazy-import your RAG helpers (fast startup)
    try:
        from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf
    except ImportError as e:
        logger.exception("Import error in pdf_tool")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Import error: {e}"}
        )

    # 3) Ingest into Pinecone
    try:
        result = ingest_pdf(tmp_path)
        logger.info("Ingested %s chunks", result.get("ingested_chunks"))
    except Exception as e:
        logger.exception("Error during ingest_pdf")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Ingest error: {e}"}
        )

    # 4) Run the RAG query
    try:
        answer = query_pdf(query)
        logger.info("RAG query succeeded")
    except Exception as e:
        logger.exception("Error during query_pdf")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Query error: {e}"}
        )
    finally:
        # 5) Cleanup temp file
        try:
            os.remove(tmp_path)
        except OSError:
            logger.warning("Could not delete temp file %s", tmp_path)

    # 6) Return the answer
    return {"output": answer}

# ─── FALLBACK 404 → DOCS ───────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return RedirectResponse(url="/docs")
