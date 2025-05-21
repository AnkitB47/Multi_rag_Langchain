# src/api/main_pdf.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from typing import Dict, Any

from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf

API_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_AUTH_TOKEN must be set")

app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    description="Upload a PDF and ask questions over it."
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process")
async def process_pdf(
    query: str,
    file: UploadFile = File(...),
    authorization: str = Header(...)
) -> Dict[str, Any]:
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(tmp_path, "wb") as out:
        out.write(await file.read())

    # ingest + query
    ingest_pdf(tmp_path)
    answer = query_pdf(query)

    # cleanup
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return {"output": answer}
