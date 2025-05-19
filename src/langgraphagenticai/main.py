# src/langgraphagenticai/main.py

import uvicorn
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from typing import Optional, Dict, Any
from langgraphagenticai.graph.chatbot_graph import create_pdf_graph, create_image_graph
from langgraphagenticai.state.state import GraphState

API_TOKEN = os.getenv("API_AUTH_TOKEN")
PDF_PORT   = int(os.getenv("PDF_SERVICE_PORT", 8001))
IMG_PORT   = int(os.getenv("GPU_SERVICE_PORT", 8000))

def make_pdf_app() -> FastAPI:
    app = FastAPI(title="PDF-RAG Service", openapi_prefix="/pdf")
    pdf_graph = create_pdf_graph()

    @app.post("/process")
    async def process_pdf(
        query: str,
        lang: str = "en",
        file: UploadFile = File(...),
        authorization: str = Header(None)
    ) -> Dict[str, Any]:
        if authorization != f"Bearer {API_TOKEN}":
            raise HTTPException(401, "Invalid token")

        path = f"/tmp/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())

        state: GraphState = {
            "input": query,
            "lang": lang,
            "pdf_path": path
        }
        result = pdf_graph.invoke(state)
        return {"output": result["final_output"]}

    return app

def make_image_app() -> FastAPI:
    app = FastAPI(title="Image-Search Service", openapi_prefix="/image")
    img_graph = create_image_graph()

    @app.post("/search")
    async def search_image(
        top_k: int = 3,
        file: UploadFile = File(...),
        authorization: str = Header(None)
    ) -> Dict[str, Any]:
        if authorization != f"Bearer {API_TOKEN}":
            raise HTTPException(401, "Invalid token")

        path = f"/tmp/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())

        state: GraphState = {
            "input": "",          # no free-form query for pure similarity
            "lang": "en",         # no translation needed here
            "image_path": path,
            "top_k": top_k
        }
        result = img_graph.invoke(state)
        return {"matches": result["image_result"]}

    return app

if __name__ == "__main__":
    # Launch two Uvicorn servers in parallel:
    # PDF on 8001, Image on 8000
    import multiprocessing

    def serve_pdf():
        uvicorn.run(make_pdf_app(), host="0.0.0.0", port=PDF_PORT)
    def serve_img():
        uvicorn.run(make_image_app(), host="0.0.0.0", port=IMG_PORT)

    for target in (serve_pdf, serve_img):
        p = multiprocessing.Process(target=target)
        p.start()

    # keep the main process alive
    multiprocessing.Event().wait()
