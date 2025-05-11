# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load secrets from env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' with integrated model.")
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
    logger.info(f"Using existing Pinecone index '{PINECONE_INDEX_NAME}'.")

# Get reference to index
index = pc.Index(PINECONE_INDEX_NAME)

def get_vectordb(pdf_path: str):
    """
    Loads a PDF, extracts text, and inserts into Pinecone using built-in embedding model.
    """
    try:
        docs = load_and_split_pdf(pdf_path)
        logger.info(f"Embedding and upserting {len(docs)} PDF chunks.")

        # Prepare documents without manual embedding
        records = [
            {
                "id": f"doc-{i}",
                "metadata": {"text": doc.page_content}
            }
            for i, doc in enumerate(docs)
        ]

        index.upsert(records=records, namespace="default")
        logger.info("Upsert successful.")

        return PineconeVectorStore(
            index=index,
            namespace="default",
            text_key="text"
        )
    except Exception as e:
        logger.error(f"Vector DB generation failed: {e}")
        raise e
