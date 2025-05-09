# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load secrets
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load OpenAI embedding model (1536-dim)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure Pinecone index exists
existing_indexes = pc.list_indexes().names()
if PINECONE_INDEX_NAME not in existing_indexes:
    logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' with 1536 dims.")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )
else:
    logger.info(f"Using existing Pinecone index '{PINECONE_INDEX_NAME}'.")

# Get index reference
index = pc.Index(PINECONE_INDEX_NAME)

def get_vectordb(pdf_path: str):
    """
    Load PDF, embed with OpenAI embeddings (1536-dim), and upsert to Pinecone.
    """
    logger.info(f"Loading and splitting PDF: {pdf_path}")
    docs = load_and_split_pdf(pdf_path)

    logger.info(f"Generating {len(docs)} embeddings...")
    vectors = [
        {
            "id": f"doc-{i}",
            "values": embedding_model.embed_query(doc.page_content),
            "metadata": {"text": doc.page_content}
        }
        for i, doc in enumerate(docs)
    ]

    try:
        logger.info(f"Upserting {len(vectors)} vectors to Pinecone...")
        index.upsert(vectors=vectors, namespace="default")
        logger.info("Upsert successful.")
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        raise e

    return PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        namespace="default",
        text_key="text"
    )
