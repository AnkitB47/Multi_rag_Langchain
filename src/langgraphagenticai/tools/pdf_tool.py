import os
from typing import List, Dict

# Updated imports to use community packages
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

from langgraphagenticai.utils.pdf_utils import load_and_split_pdf

# ─────────────────────────────────────────────────────────────────────────────
#   CONFIGURATION FROM ENV
# ─────────────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX_NAME"]
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
#   SETUP PINECONE & VECTORSTORE (Fixed initialization)
# ─────────────────────────────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX)  # Renamed to avoid confusion

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)

# Initialize vectorstore with correct parameters
vectordb = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,  # Pass the embeddings object directly
    text_key="text"
)

# ─────────────────────────────────────────────────────────────────────────────
#   PDF INGEST & QUERY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: str, namespace: str = "default") -> Dict[str, int]:
    """
    Load PDF, split into chunks, embed, and upsert into Pinecone.
    Returns dictionary with count of ingested chunks.
    """
    docs = load_and_split_pdf(pdf_path)
    vectors = []
    
    for i, doc in enumerate(docs):
        embedding = embeddings.embed_documents([doc.page_content])[0]
                
        vectors.append({
            "id": f"doc-{i}",
            "values": embedding,
            "metadata": {
                "text": doc.page_content,
                "source": os.path.basename(pdf_path)
            }
        })
    
    pinecone_index.upsert(vectors=vectors, namespace=namespace)
    return {"ingested_chunks": len(vectors)}

def query_pdf(query: str, namespace: str = "default") -> str:
    """
    Run a RetrievalQA chain over Pinecone index and return the answer.
    """
    retriever = vectordb.as_retriever(namespace=namespace)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa.run(query)