# src/api/main_pdf.py

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status

from langgraphagenticai.graph.chatbot_graph import create_pdf_graph
from langgraphagenticai.state.state import GraphState

# ─── Config & sanity checks ─────────────────────────────
API_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_AUTH_TOKEN environment variable must be set")

# make sure our temp dir exists
Path("/tmp").mkdir(parents=True, exist_ok=True)

# ─── FastAPI app ────────────────────────────────────────
app = FastAPI(
    title="PDF-RAG Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    description="Upload a PDF and ask free-form questions over it via RetrievalQA"
)

# compile the PDF graph once at startup
_pdf_graph = create_pdf_graph()


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Simple liveness probe."""
    return {"status": "ok", "service": "pdf-rag"}


@app.post(
    "/process",
    response_model=Dict[str, Any],
    summary="Process a PDF + question",
    description="Saves your PDF to disk, runs it through the RAG graph, and returns the answer."
)
async def process_pdf(
    query: str = Form(..., description="Your natural-language question"),
    lang: str = Form("en", description="Two-letter target language code"),
    file: UploadFile = File(..., description="Your PDF file"),
    authorization: str = Header(
        ..., alias="Authorization",
        description="Must be `Bearer <API_AUTH_TOKEN>`"
    )
) -> Dict[str, Any]:
    # ─── Auth ─────────────────────────────────────────────
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Authorization header"
        )

    # ─── Save upload ──────────────────────────────────────
    tmp_path = f"/tmp/{file.filename}"
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as out:
            out.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save PDF: {e}"
        )

    # ─── Invoke the graph ─────────────────────────────────
    state: GraphState = {
        "input": query,
        "lang": lang,
        "pdf_path": tmp_path
    }
    result = _pdf_graph.invoke(state)

    # ─── Build response ──────────────────────────────────
    return {
        "output": result.get("final_output")
    }
