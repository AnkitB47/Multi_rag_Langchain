# src/api/main_pdf.py

import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from typing import Dict, Any

# ← import your PDF-tools from the package root
from langgraphagenticai.tools.pdf_tool import ingest_pdf, query_pdf

API_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_AUTH_TOKEN must be set")

app = FastAPI(
    title="PDF-RAG Service",
    description="Upload & query PDFs via RAG",
    version="1.0.0",
    openapi_prefix=""  # root
)

@app.post("/process")
async def process_pdf(
    query: str,
    lang: str = "en",
    file: UploadFile = File(...),
    authorization: str = Header(None)
) -> Dict[str, Any]:
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "Invalid token")

    # write upload to /tmp with a unique name
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(tmp_path, "wb") as out:
        out.write(await file.read())

    # ingest into Pinecone (namespace default)
    ingest_pdf(tmp_path)

    # run RAG query
    answer = query_pdf(query)

    # cleanup
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return {"output": answer}
