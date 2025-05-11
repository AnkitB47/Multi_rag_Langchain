# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf
from typing import List, Dict, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")
EMBEDDING_MODEL = "llama-text-embed-v2"

class PineconeManager:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self._initialize_index()

    def _initialize_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
                raise ValueError(f"Index {PINECONE_INDEX_NAME} does not exist. Please create it first in Pinecone console with dimension 2048.")
            
            index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Verify index description
            desc = self.pc.describe_index(PINECONE_INDEX_NAME)
            if not desc.status['ready']:
                raise ValueError("Index is not ready")
                
            logger.info(f"Using existing index: {PINECONE_INDEX_NAME}")
            return index
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise

    def prepare_documents(self, docs: List[Any]) -> List[Dict[str, Any]]:
        """Prepare documents for Pinecone upsert with built-in embeddings"""
        return [{
            "id": f"doc-{i}",
            "values": [],  # Empty array for built-in embeddings
            "metadata": {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "model": EMBEDDING_MODEL  # Critical for built-in embeddings
            }
        } for i, doc in enumerate(docs)]

    def get_vectordb(self, pdf_path: str):
        """Create and return a Pinecone vector store"""
        try:
            logger.info(f"📄 Loading and splitting PDF: {pdf_path}")
            docs = load_and_split_pdf(pdf_path)
            
            if not docs:
                raise ValueError("No documents extracted from PDF")
                
            records = self.prepare_documents(docs)
            
            logger.info(f"📤 Upserting {len(records)} records to Pinecone")
            
            # Upsert with batch processing
            batch_size = 100  # Adjust based on your needs
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace="default"
                )
            
            # Initialize vector store with proper configuration
            return PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                pinecone_api_key=PINECONE_API_KEY,
                embedding=None,  # Using built-in embeddings
                namespace="default",
                text_key="text",
                index=self.index
            )
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

# Singleton instance
pinecone_manager = PineconeManager()
get_vectordb = pinecone_manager.get_vectordb