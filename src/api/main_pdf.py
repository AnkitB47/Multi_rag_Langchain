# src/api/main_pdf.py

import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG & LOGGER
# ─────────────────────────────────────────────────────────────────────────────
API_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_AUTH_TOKEN must be set in the environment")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("pdf-rag-service")

# ─────────────────────────────────────────────────────────────────────────────
#  APP INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and run Retrieval-Augmented Generation queries over it.",
)

# Allow browser‐based UIs on other hosts (if you expand later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten in prod
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def redirect_to_docs():
    """
    Redirect root to the interactive OpenAPI docs.
    """
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Service healthcheck")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/process",
    summary="Ingest uploaded PDF & run a RAG query over it",
    response_model=Dict[str, Any],
)
async def process_pdf(
    request: Request,
    query: str,
    file: UploadFile = File(...),
    authorization: str = Header(..., description="Bearer <token>"),
) -> Dict[str, Any]:
    # ----- Auth check -----
    if authorization != f"Bearer {API_TOKEN}":
        logger.warning("Unauthorized attempt from %s", request.client.host)
        raise HTTPException(status_code=401, detail="Invalid API token")

    # ----- Save upload locally -----
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as out_fp:
            out_fp.write(contents)
    except Exception as e:
        logger.error("Failed to save upload: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not save uploaded file")

    # ----- Lazy import heavy dependencies -----
    try:
        from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf
    except ImportError as e:
        logger.error("PDF tool not found: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    # ----- Ingest & query -----
    try:
        ingest_pdf(tmp_path)
        answer = query_pdf(query)
    except Exception as e:
        logger.error("RAG pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing PDF")

    # ----- Cleanup -----
    try:
        os.remove(tmp_path)
    except OSError:
        logger.warning("Could not remove temp file %s", tmp_path)

    return {"output": answer}


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL ERROR HANDLER (optional)
# ─────────────────────────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "Route not found. See /docs for available endpoints."},
    )
