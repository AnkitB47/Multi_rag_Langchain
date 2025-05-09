# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.utils.pdf_utils import load_and_split_pdf

# Load credentials from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")

# Define embedding model (2048 dimensions)
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-bert-2048"
)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists with correct 2048 dimensions
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=2048,  # Important: match embedding output dim
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

# Get reference to index
index = pc.Index(PINECONE_INDEX_NAME)

def get_vectordb(pdf_path: str):
    """
    Load PDF, embed using nomic-bert-2048 (2048-dim), and upsert to Pinecone.
    """
    docs = load_and_split_pdf(pdf_path)
    
    vectors = [
        {
            "id": f"doc-{i}",
            "values": embedding_model.embed_query(doc.page_content),
            "metadata": {"text": doc.page_content}
        }
        for i, doc in enumerate(docs)
    ]

    # Upsert to Pinecone under 'default' namespace
    index.upsert(vectors=vectors, namespace="default")

    # Return vector store object
    return PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        namespace="default",
        text_key="text"
    )
