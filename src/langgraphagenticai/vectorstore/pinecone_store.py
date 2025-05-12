# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
import math
import time
import threading
from typing import List, Optional
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multiagentrag"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 1536
MAX_RETRIES = 3
INITIAL_DELAY = 1  # seconds

class PineconeService:
    def __init__(self):
        """Initialize with automatic retry and fallback handling."""
        self._init_services()
        
    def _init_services(self):
        """Initialize all services with retry logic."""
        self.pc = Pinecone(
            api_key=PINECONE_API_KEY,
            additional_headers={"X-Pinecone-Request-Source": "langgraph-agent"}
        )
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSION
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )

    @retry(stop=stop_after_attempt(MAX_RETRIES), 
           wait=wait_exponential(multiplier=1, min=INITIAL_DELAY, max=10))
    def _verify_index(self):
        """Ensure index is available with retry logic."""
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            raise ValueError(f"Index {PINECONE_INDEX_NAME} not found")
        if not self.pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            raise ConnectionError("Index not ready")

    def _process_pdf(self, pdf_path: str) -> List:
        """Robust PDF processing with validation."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        if os.path.getsize(pdf_path) > 10 * 1024 * 1024:
            raise ValueError("PDF exceeds 10MB size limit")

        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load_and_split(self.text_splitter)
        
        if not docs:
            raise ValueError("No readable content in PDF")
            
        for doc in docs:
            doc.metadata.update({
                "source": os.path.basename(pdf_path),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        return docs

    @retry(stop=stop_after_attempt(MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=INITIAL_DELAY, max=10))
    def get_vectorstore(self, pdf_path: str) -> PineconeVectorStore:
        """Main processing with automatic retries."""
        try:
            docs = self._process_pdf(pdf_path)
            avg_size = sum(len(str(d.page_content)) for d in docs) / len(docs)
            batch_size = max(1, min(100, math.floor(1.8 * 1024 * 1024 / (avg_size * 1.2))))
            
            return PineconeVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings,
                index_name=PINECONE_INDEX_NAME,
                batch_size=batch_size,  
                namespace="default"
            )
        except Exception as e:
            logger.error(f"Attempt failed: {str(e)}")
            raise

# Thread-safe singleton
_service_instance = None
_service_lock = threading.Lock()

def get_vectordb(pdf_path: str) -> PineconeVectorStore:
    """Public interface with proper error handling."""
    global _service_instance
    try:
        with _service_lock:
            if _service_instance is None:
                _service_instance = PineconeService()
                _service_instance._verify_index()
        
        return _service_instance.get_vectorstore(pdf_path)
    except Exception as e:
        logger.error(f"Service error: {str(e)}")
        raise RuntimeError(
            "We're experiencing high demand. "
            "Please try again in a moment."
        )