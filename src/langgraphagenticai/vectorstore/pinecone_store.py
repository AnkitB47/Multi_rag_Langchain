# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def initialize_pinecone_index():
    """Initialize or connect to Pinecone index with proper configuration"""
    try:
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
            
            # Updated index creation with proper spec
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=2048,  # Required dimension for llama-text-embed-v2
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_REGION
                )
            )
            
            # Wait for index to be ready
            import time
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            
            logger.info("Index created successfully")
        else:
            logger.info(f"Using existing index: {PINECONE_INDEX_NAME}")
            
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone index: {e}")
        raise

index = initialize_pinecone_index()

def prepare_documents(docs: List[Any]) -> List[Dict[str, Any]]:
    """Prepare documents for Pinecone upsert"""
    return [{
        "id": f"doc-{i}",
        "values": [],  # Empty when using built-in embeddings
        "metadata": {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown")
        }
    } for i, doc in enumerate(docs)]

def get_vectordb(pdf_path: str):
    try:
        logger.info(f"📄 Loading and splitting PDF: {pdf_path}")
        docs = load_and_split_pdf(pdf_path)
        
        if not docs:
            raise ValueError("No documents extracted from PDF")
            
        records = prepare_documents(docs)
        
        logger.info(f"📤 Upserting {len(records)} records to Pinecone")
        response = index.upsert(
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
            index=index
        )
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise