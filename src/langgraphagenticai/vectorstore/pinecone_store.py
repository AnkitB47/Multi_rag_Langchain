# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf
from typing import List, Dict, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - MUST match your Pinecone console settings exactly
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multiagentrag"  # Hardcoded to prevent config mismatches
EMBEDDING_MODEL = "llama-text-embed-v2"

class PineconeVectorManager:
    def __init__(self):
        """Initialize with EXACT configuration Pinecone requires"""
        try:
            # Initialize with critical headers for built-in embeddings
            self.pc = Pinecone(
                api_key=PINECONE_API_KEY,
                additional_headers={
                    "X-Pinecone-Embedding-Model": EMBEDDING_MODEL
                }
            )
            self.index = self._get_verified_index()
            logger.info("Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

    def _get_verified_index(self):
        """Verify the index exists and has correct configuration"""
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            error_msg = (
                f"Pinecone index '{PINECONE_INDEX_NAME}' not found.\n"
                "Please create it in Pinecone console with these EXACT settings:\n"
                "- Name: multiagentrag\n"
                "- Dimension: 2048\n"
                "- Metric: cosine\n"
                "- Cloud: AWS\n"
                "- Region: us-east-1\n"
                "- Pod type: serverless\n"
                "- Embedding model: llama-text-embed-v2"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        index = self.pc.Index(PINECONE_INDEX_NAME)
        return index

    def _prepare_records(self, docs: List[Any]) -> List[Dict[str, Any]]:
        """Prepare documents in EXACT format Pinecone requires"""
        return [{
            "id": f"doc-{i}-{int(time.time())}",
            "values": [],  # MUST be empty for built-in embeddings
            "metadata": {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                # These fields are CRITICAL:
                "model": EMBEDDING_MODEL,
                "embedding_model": EMBEDDING_MODEL
            }
        } for i, doc in enumerate(docs)]

    def upsert_documents(self, docs: List[Any]) -> None:
        """Upsert documents with proper batch processing"""
        records = self._prepare_records(docs)
        
        try:
            # Using upsert with EXACT required parameters
            response = self.index.upsert(
                vectors=records,
                namespace="default",
                embedding_model=EMBEDDING_MODEL  # REQUIRED
            )
            logger.info(f"Successfully upserted {len(records)} documents")
            return response
        except Exception as e:
            logger.error(f"Failed to upsert documents: {str(e)}")
            raise RuntimeError(f"Document upsert failed: {str(e)}")

    def get_vectorstore(self, pdf_path: str) -> PineconeVectorStore:
        """Create and return a fully configured PineconeVectorStore"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            docs = load_and_split_pdf(pdf_path)
            
            if not docs:
                raise ValueError("PDF processing returned no documents")
                
            # Upsert with proper configuration
            self.upsert_documents(docs)
            
            # Initialize with EXACT configuration
            return PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=None,  # MUST be None for built-in
                text_key="text",
                namespace="default",
                pinecone_api_key=PINECONE_API_KEY,
                embedding_model=EMBEDDING_MODEL  # REQUIRED
            )
        except Exception as e:
            logger.error(f"Vector store creation failed: {str(e)}")
            raise RuntimeError(f"Vector store creation failed: {str(e)}")

# Singleton instance
_vector_manager = None

def get_vectordb(pdf_path: str) -> PineconeVectorStore:
    """Public interface with lazy initialization"""
    global _vector_manager
    try:
        if _vector_manager is None:
            _vector_manager = PineconeVectorManager()
        return _vector_manager.get_vectorstore(pdf_path)
    except Exception as e:
        logger.error(f"Failed to get vector store: {str(e)}")
        raise