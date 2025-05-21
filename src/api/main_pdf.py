# src/api/main_pdf.py
import os
import uuid
import logging
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

# ─── CONFIG & LOGGER ──────────────────────────────────────────────────────────
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_AUTH_TOKEN:
    logging.warning("API_AUTH_TOKEN not set — skipping header check.")

logging.basicConfig(level=logging.DEBUG)  # <-- DEBUG level
logger = logging.getLogger("pdf_rag_service")

# ─── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and run RAG over it, with full debug logging.",
    debug=True,  # <-- enable debug mode so tracebacks show in responses
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def catch_all_exceptions(request: Request, exc: Exception):
    # Log full traceback
    tb = traceback.format_exc()
    logger.error("Unhandled exception:\n%s", tb)
    # Return the exception text back in JSON so you can see it in the client
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}", "trace": tb}
    )

@app.get("/", include_in_schema=False)
def root() -> Dict[str, str]:
    return {"message": "See /docs for the OpenAPI UI."}

@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post(
    "/process",
    summary="Ingest a PDF & run a RAG query",
    response_model=Dict[str, Any],
)
async def process_pdf(
    query: str = Form(..., description="Your question"),
    file: UploadFile = File(..., description="PDF file to ingest"),
):
    # 1) Save upload
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
        logger.debug("Wrote upload to %s", tmp_path)
    except Exception as e:
        logger.exception("Failed to write PDF")
        raise HTTPException(500, f"I/O error: {e}")

    # 2) Lazy-import RAG tools (so import-time errors bubble here)
    try:
        from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf
    except Exception as e:
        logger.exception("Import error in pdf_tool")
        raise HTTPException(500, f"Import error: {e}")

    # 3) Ingest
    try:
        stats = ingest_pdf(tmp_path)
        logger.debug("ingest_pdf returned: %s", stats)
    except Exception as e:
        logger.exception("Error in ingest_pdf")
        raise HTTPException(500, f"Ingest error: {e}")

    # 4) Query
    try:
        answer = query_pdf(query)
        logger.debug("query_pdf returned: %s", answer)
    except Exception as e:
        logger.exception("Error in query_pdf")
        raise HTTPException(500, f"Query error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            logger.warning("Could not delete %s", tmp_path)

    return {"output": answer}
