# src/api/main_pdf.py
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

# ─── CONFIG & LOGGER ──────────────────────────────────────────────────────────
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_AUTH_TOKEN:
    logging.warning("API_AUTH_TOKEN not set — requests will still be gated but not checked.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_rag_service")

# ─── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and run Retrieval-Augmented queries over it, no external Replicate.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root() -> Dict[str, str]:
    return {"message": "Welcome! Point your client at /docs for the OpenAPI UI."}

@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post(
    "/process",
    summary="Ingest a PDF & run a RAG query in one go",
    response_model=Dict[str, Any],
)
async def process_pdf(
    query: str = Form(..., description="Your question about the PDF"),
    file: UploadFile = File(..., description="The PDF to ingest"),
):
    # 1) Save the upload to a temp file
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.exception("Failed to write uploaded PDF")
        raise HTTPException(500, f"I/O error: {e}")

    # 2) Lazy-import your RAG tool
    try:
        from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf
    except Exception as e:
        logger.exception("Import error in pdf_tool")
        return JSONResponse(500, {"detail": f"Import error: {e}"})

    # 3) Ingest into Pinecone
    try:
        stats = ingest_pdf(tmp_path)
        logger.info("Ingested %d chunks", stats.get("ingested_chunks", 0))
    except Exception as e:
        logger.exception("Error during ingest_pdf")
        return JSONResponse(500, {"detail": f"Ingest error: {e}"})

    # 4) Run the RAG query
    try:
        answer = query_pdf(query)
        logger.info("RAG query succeeded")
    except Exception as e:
        logger.exception("Error during query_pdf")
        return JSONResponse(500, {"detail": f"Query error: {e}"})
    finally:
        # 5) Cleanup temp file
        try:
            os.remove(tmp_path)
        except OSError:
            logger.warning("Could not delete temp file %s", tmp_path)

    # 6) Return the answer
    return {"output": answer}
