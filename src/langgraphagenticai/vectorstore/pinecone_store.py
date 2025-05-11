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

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multiagentrag"  # Must match your new index name
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI's standard model

class PineconeManager:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self._verify_index()

    def _verify_index(self):
        """Ensure index exists with correct configuration"""
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            raise ValueError(
                f"Index '{PINECONE_INDEX_NAME}' not found. "
                "Please create it in Pinecone console with:\n"
                "- Dimension: 1536\n"
                "- Metric: cosine\n"
                "- Serverless (AWS us-east-1)"
            )

    def get_vectorstore(self, pdf_path: str) -> PineconeVectorStore:
        """Create and return a ready-to-use vector store"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            docs = load_and_split_pdf(pdf_path)
            
            return PineconeVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings,
                index_name=PINECONE_INDEX_NAME,
                namespace="default"
            )
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

# Singleton instance
vector_manager = PineconeManager()
get_vectordb = vector_manager.get_vectorstore