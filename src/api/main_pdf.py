# src/api/main_pdf.py

import os
import uuid
import logging
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from langgraphagenticai.tools import pdf_tool  # for type checking only; actual import done lazily

# ─── CONFIG & LOGGER ──────────────────────────────────────────────────────────
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")
if not API_AUTH_TOKEN:
    logging.warning("No API_AUTH_TOKEN set; skipping auth checks.")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_rag_service")

# ─── APP & CORS ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and run Retrieval-Augmented queries over it",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in prod!
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ─── ROOT → REDIRECT TO SWAGGER ─────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url=app.docs_url)

# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

# ─── CUSTOM ERROR HANDLERS ────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 404:
        # send every 404 back to docs
        return RedirectResponse(url=app.docs_url)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# ─── PDF INGEST & QUERY ENDPOINT ───────────────────────────────────────────────
@app.post(
    "/process",
    summary="Ingest PDF & run a RAG query",
    response_model=Dict[str, Any],
)
async def process_pdf(
    query: str = Form(..., description="Your question about the PDF"),
    file: UploadFile = File(..., description="The PDF file to ingest"),
):
    # 1) Save the upload to a temp file
    tmp_dir = "/tmp"
    file_id = uuid.uuid4().hex
    tmp_path = os.path.join(tmp_dir, f"{file_id}_{file.filename}")
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
    except Exception:
        logger.exception("Failed to save uploaded PDF")
        raise HTTPException(status_code=500, detail="Could not save uploaded PDF")

    # 2) Lazy-import the RAG helpers
    try:
        from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf
    except Exception:
        logger.exception("Failed to import PDF tool")
        cleanup(tmp_path)
        raise HTTPException(status_code=500, detail="Internal import error")

    # 3) Ingest into Pinecone
    try:
        ingest_result = ingest_pdf(tmp_path)
        logger.info("Ingested %d chunks", ingest_result.get("ingested_chunks", 0))
    except Exception:
        logger.exception("Ingest error")
        cleanup(tmp_path)
        raise HTTPException(status_code=500, detail="Error ingesting PDF")

    # 4) Run the RAG query
    try:
        answer = query_pdf(query)
        logger.info("Query succeeded")
    except Exception:
        logger.exception("Query error")
        cleanup(tmp_path)
        raise HTTPException(status_code=500, detail="Error running query")

    # 5) Clean up and respond
    cleanup(tmp_path)
    return {"output": answer}

def cleanup(path: str):
    try:
        os.remove(path)
    except Exception:
        logger.warning("Could not delete temp file %s", path)
