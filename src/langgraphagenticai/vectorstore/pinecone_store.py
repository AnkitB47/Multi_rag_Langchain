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
DIMENSION = 2048  # Must match your index dimension

class PineconeManager:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self._initialize_index()

    def _initialize_index(self):
        """Initialize or connect to Pinecone index with proper configuration"""
        try:
            if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
                logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
                
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_REGION
                    ),
                    metadata_config={"indexed": ["text"]}
                )
                
                # Wait for index to be ready
                self._wait_for_index_ready()
                logger.info("Index created successfully")
            else:
                logger.info(f"Using existing index: {PINECONE_INDEX_NAME}")
                
            return self.pc.Index(PINECONE_INDEX_NAME)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise

    def _wait_for_index_ready(self, timeout: int = 300):
        """Wait for index to be ready"""
        start_time = time.time()
        while True:
            try:
                desc = self.pc.describe_index(PINECONE_INDEX_NAME)
                if desc.status['ready']:
                    return
                if time.time() - start_time > timeout:
                    raise TimeoutError("Index creation timed out")
                time.sleep(5)
            except Exception as e:
                logger.warning(f"Waiting for index: {e}")
                time.sleep(5)

    def prepare_documents(self, docs: List[Any]) -> List[Dict[str, Any]]:
        """Prepare documents for Pinecone upsert with built-in embeddings"""
        return [{
            "id": f"doc-{i}",
            "values": [],  # Empty for built-in embeddings
            "metadata": {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "model": EMBEDDING_MODEL  # Specify the embedding model
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
            response = self.index.upsert(
                vectors=records,
                namespace="default"
            )
            
            logger.debug(f"Pinecone upsert response: {response}")
            
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