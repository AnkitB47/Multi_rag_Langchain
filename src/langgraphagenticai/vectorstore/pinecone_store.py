# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index with built-in embedding if not exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' with built-in embeddings...")
    pc.create_index_for_model(
        name=PINECONE_INDEX_NAME,
        cloud="aws",
        region=PINECONE_REGION,
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "text"}
        }
    )
else:
    logger.info(f"Using existing index: {PINECONE_INDEX_NAME}")

# Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)

def get_vectordb(pdf_path: str):
    logger.info(f"📄 Loading and splitting PDF: {pdf_path}")
    docs = load_and_split_pdf(pdf_path)

    records = [{"id": f"doc-{i}", "text": doc.page_content} for i, doc in enumerate(docs)]

    logger.info(f"📤 Upserting {len(records)} records to Pinecone.")
    index.upsert(records=records, namespace="default")

    return PineconeVectorStore(
        index=index,
        embedding=None,  # using built-in embeddings
        namespace="default",
        text_key="text"
    )
