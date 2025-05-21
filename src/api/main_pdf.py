# src/api/main_pdf.py
import os, uuid, traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from typing import Dict, Any

# Only one secret is read at import-time; defer the rest to inside the endpoint.
API_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_AUTH_TOKEN must be set in env")

app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and run RAG queries over it.",
)

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/process")
async def process_pdf(
    query: str,
    file: UploadFile = File(...),
    authorization: str = Header(...)
) -> Dict[str, Any]:
    # — auth header check —
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "Invalid token")

    # — verify all the other secrets are set —
    missing = [
        k for k in ("PINECONE_API_KEY","PINECONE_INDEX_NAME","OPENAI_API_KEY")
        if not os.getenv(k)
    ]
    if missing:
        raise HTTPException(
            500,
            detail=f"Missing environment variables: {', '.join(missing)}"
        )

    # — lazy import so /health stays fast, and so we can catch errors —
    try:
        from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(500, detail=f"Import/pdf_tool error:\n{tb}")

    # — write upload to disk —
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(tmp_path, "wb") as out:
        out.write(await file.read())

    # — ingest & run RAG —
    try:
        ingest_pdf(tmp_path)
        answer = query_pdf(query)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(500, detail=f"Processing error:\n{tb}")
    finally:
        # cleanup even on error
        try: os.remove(tmp_path)
        except OSError: pass

    return {"output": answer}
