# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf
from typing import List, Dict, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - MUST match your Pinecone index exactly
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multiagentrag"  # Hardcoded to prevent env var issues
EMBEDDING_MODEL = "llama-text-embed-v2"
PINECONE_REGION = "us-east-1"  # Hardcoded to match your index

class PineconeVectorManager:
    def __init__(self):
        """Initialize with EXACT configuration Pinecone requires for built-in embeddings"""
        try:
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
        
        # Verify index is ready
        desc = self.pc.describe_index(PINECONE_INDEX_NAME)
        if not desc.status['ready']:
            raise ValueError(f"Index {PINECONE_INDEX_NAME} is not ready")
            
        return index

    def _prepare_document_batch(self, docs: List[Any]) -> List[Dict[str, Any]]:
        """Prepare documents with ALL REQUIRED Pinecone metadata fields"""
        return [{
            "id": f"doc-{i}-{time.time_ns()}",
            "values": [],  # MUST be empty array for built-in embeddings
            "metadata": {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                # These fields are ABSOLUTELY REQUIRED:
                "model": EMBEDDING_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                # Recommended additional metadata:
                "doc_type": "pdf",
                "chunk_index": i,
                "processing_time": time.time()
            }
        } for i, doc in enumerate(docs)]

    def upsert_documents(self, docs: List[Any]) -> None:
        """Upsert documents with proper batch processing and error handling"""
        records = self._prepare_document_batch(docs)
        
        # Optimal batch size for serverless
        batch_size = 50  
        success_count = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                response = self.index.upsert(
                    vectors=batch,
                    namespace="default",
                    # These parameters are CRITICAL:
                    embedding_model=EMBEDDING_MODEL,
                    batch_size=batch_size
                )
                success_count += len(batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}/{len(records)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to upsert batch starting at doc {i}: {str(e)}")
                raise RuntimeError(f"Failed to upsert documents: {str(e)}")
                
        logger.info(f"Successfully upserted {success_count}/{len(records)} documents")

    def get_vectorstore(self, pdf_path: str) -> PineconeVectorStore:
        """
        Create and return a fully configured PineconeVectorStore
        with guaranteed compatibility for built-in embeddings
        """
        try:
            logger.info(f"Loading and processing PDF: {pdf_path}")
            docs = load_and_split_pdf(pdf_path)
            
            if not docs:
                raise ValueError("PDF processing returned no documents")
                
            logger.info(f"Upserting {len(docs)} document chunks")
            self.upsert_documents(docs)
            
            # Initialize with EXACT configuration required
            return PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=None,  # MUST be None for built-in
                text_key="text",
                namespace="default",
                pinecone_api_key=PINECONE_API_KEY,
                # These are ABSOLUTELY REQUIRED:
                embedding_model=EMBEDDING_MODEL,
                index=self.index
            )
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise RuntimeError(f"Vector store creation failed: {str(e)}")

# Singleton instance - ensures single connection pool
_vector_manager: Optional[PineconeVectorManager] = None

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