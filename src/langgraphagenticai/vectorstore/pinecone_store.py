# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
import hashlib
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load secrets
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI Embedding Model (1536-dim)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure Pinecone index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )
else:
    logger.info(f"Using existing index '{PINECONE_INDEX_NAME}'")

# Index object
index = pc.Index(PINECONE_INDEX_NAME)

def hash_file_name(filename: str) -> str:
    """Hash filename to create namespace"""
    return hashlib.sha256(filename.encode()).hexdigest()

def get_vectordb(pdf_path: str):
    """
    Load and cache embedded vectors per PDF using hashed namespace.
    Avoids re-embedding same document.
    """
    namespace = f"pdf-{hash_file_name(os.path.basename(pdf_path))}"
    logger.info(f"Using namespace: {namespace}")

    # Check if namespace already has vectors
    stats = index.describe_index_stats()
    if namespace in stats.get("namespaces", {}):
        logger.info(f"Vectors already exist for {namespace}, skipping upsert.")
    else:
        logger.info(f"Embedding and uploading new PDF: {pdf_path}")
        docs = load_and_split_pdf(pdf_path)
        vectors = [
            {
                "id": f"doc-{i}",
                "values": embedding_model.embed_query(doc.page_content),
                "metadata": {"text": doc.page_content}
            }
            for i, doc in enumerate(docs)
        ]
        try:
            index.upsert(vectors=vectors, namespace=namespace)
            logger.info("PDF vectors successfully upserted.")
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise

    return PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        namespace=namespace,
        text_key="text"
    )
