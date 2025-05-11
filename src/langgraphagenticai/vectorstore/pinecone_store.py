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

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_REGION") or "us-east-1"
EMBEDDING_MODEL = "llama-text-embed-v2"

class PineconeVectorManager:
    def __init__(self):
        # Initialize Pinecone client with proper configuration
        self.pc = Pinecone(
            api_key=PINECONE_API_KEY,
            additional_headers={
                "X-Pinecone-Embedding-Model": EMBEDDING_MODEL  # Critical header
            }
        )
        self.index = self._verify_index()

    def _verify_index(self):
        """Verify index exists and is properly configured"""
        try:
            if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
                raise ValueError(
                    f"Index {PINECONE_INDEX_NAME} not found. "
                    "Please create it in Pinecone console with:\n"
                    "- Dimension: 2048\n"
                    "- Metric: cosine\n"
                    "- Pod type: serverless\n"
                    "- Cloud: AWS\n"
                    "- Region: us-east-1"
                )
            
            index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Verify index status
            desc = self.pc.describe_index(PINECONE_INDEX_NAME)
            if not desc.status['ready']:
                raise ValueError("Pinecone index is not ready")
                
            logger.info(f"Verified index: {PINECONE_INDEX_NAME}")
            return index
            
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            raise

    def prepare_documents(self, docs: List[Any]) -> List[Dict[str, Any]]:
        """Prepare documents for Pinecone with built-in embeddings"""
        return [{
            "id": f"doc-{i}",
            "values": [],  # MUST be empty for built-in embeddings
            "metadata": {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                # These fields are CRITICAL for built-in embeddings:
                "model": EMBEDDING_MODEL,
                "embedding_model": EMBEDDING_MODEL
            }
        } for i, doc in enumerate(docs)]

    def upsert_documents(self, docs: List[Any]) -> None:
        """Upsert documents with proper batch processing"""
        records = self.prepare_documents(docs)
        
        # Batch upsert with error handling
        batch_size = 50  # Optimal for serverless
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                response = self.index.upsert(
                    vectors=batch,
                    namespace="default",
                    # These parameters are REQUIRED:
                    embedding_model=EMBEDDING_MODEL,
                    batch_size=batch_size
                )
                logger.debug(f"Upserted batch {i//batch_size + 1}: {response}")
            except Exception as e:
                logger.error(f"Failed to upsert batch {i//batch_size + 1}: {e}")
                raise

    def get_vectorstore(self, pdf_path: str):
        """Create and return a configured Pinecone vector store"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            docs = load_and_split_pdf(pdf_path)
            
            if not docs:
                raise ValueError("No content extracted from PDF")
                
            # Upsert documents with proper embedding configuration
            self.upsert_documents(docs)
            
            # Initialize vector store with EXACT configuration
            return PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=None,  # MUST be None for built-in
                text_key="text",
                namespace="default",
                pinecone_api_key=PINECONE_API_KEY,
                # These additional configs are REQUIRED:
                embedding_model=EMBEDDING_MODEL,
                index=self.index
            )
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise

# Global instance
vector_manager = PineconeVectorManager()
get_vectordb = vector_manager.get_vectorstore