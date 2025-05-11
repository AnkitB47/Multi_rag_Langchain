# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - MUST match your Pinecone index
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multiagentrag"  # Must match exactly
EMBEDDING_MODEL = "text-embedding-3-large"  # Updated to match your index
EMBEDDING_DIMENSION = 1536  # Must match index dimension

class PineconeVectorManager:
    def __init__(self):
        """Initialize with proper embedding model and Pinecone connection"""
        try:
            # Initialize embeddings with correct dimension
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSION  # Critical for dimension matching
            )
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self._verify_index()
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize: {str(e)}")

    def _verify_index(self):
        """Verify index exists and has correct configuration"""
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            raise ValueError(
                f"Index '{PINECONE_INDEX_NAME}' not found. "
                "Please create it in Pinecone console with:\n"
                f"- Dimension: {EMBEDDING_DIMENSION}\n"
                "- Metric: cosine\n"
                "- Serverless (AWS us-east-1)"
            )

    def get_vectorstore(self, pdf_path: str) -> PineconeVectorStore:
        """Create and return a properly configured vector store"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Load and split PDF
            docs = load_and_split_pdf(pdf_path)
            if not docs:
                raise ValueError("No content extracted from PDF")
            
            # Create vector store with EXACT configuration
            return PineconeVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings,
                index_name=PINECONE_INDEX_NAME,
                namespace="default",
                text_key="text"  # Must match metadata field
            )
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

# Singleton instance for efficient reuse
vector_manager = None

def get_vectordb(pdf_path: str) -> PineconeVectorStore:
    """Public interface with lazy initialization"""
    global vector_manager
    try:
        if vector_manager is None:
            vector_manager = PineconeVectorManager()
        return vector_manager.get_vectorstore(pdf_path)
    except Exception as e:
        logger.error(f"Failed to get vector store: {str(e)}")
        raise