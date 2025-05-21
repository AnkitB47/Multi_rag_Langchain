# src/api/main_pdf.py
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# ─── CONFIG & LOGGER ──────────────────────────────────────────────────────────
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_AUTH_TOKEN:
    # we’ll still require it under the hood, but no header on the API call
    logging.warning("API_AUTH_TOKEN not set — requests will still be gated but not checked.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_rag_service")

# ─── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and run Retrieval-Augmented queries over it, no extra headers needed.",
)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Point your client at /docs for the OpenAPI UI"}


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.post(
    "/process",
    summary="Ingest a PDF & run a RAG query in one go",
    response_model=dict,
)
async def process_pdf(
    query: str = Form(..., description="Your question about the PDF"),
    file: UploadFile = File(..., description="The PDF to ingest"),
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

    # 2) Lazy-import your PDF-RAG tool so uvicorn can bind instantly
    try:
        from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf
    except Exception as e:
        logger.exception("Import error in pdf_tool")
        # include the real exception in the detail
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
