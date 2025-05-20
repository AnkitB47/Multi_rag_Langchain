# src/api/main_pdf.py

import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from typing import Dict, Any
from langgraphagenticai.graph.chatbot_graph import create_pdf_graph
from langgraphagenticai.state.state import GraphState

API_TOKEN = os.getenv("API_AUTH_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Missing API_AUTH_TOKEN")

# build your PDF‐only graph
pdf_graph = create_pdf_graph()

app = FastAPI(
    title="PDF-RAG Service",
    description="Upload a PDF and ask questions",
    version="1.0.0"
)

@app.post("/process")
async def process_pdf(
    query: str,
    lang: str = "en",
    file: UploadFile = File(...),
    authorization: str = Header(None, alias="Authorization")
) -> Dict[str, Any]:
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "Invalid token")

    # write temp file
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as out:
        out.write(await file.read())

    # invoke LangGraph
    state: GraphState = {
        "input": query,
        "lang": lang,
        "pdf_path": tmp_path
    }
    result = pdf_graph.invoke(state)
    return {"output": result.get("final_output", "")}
