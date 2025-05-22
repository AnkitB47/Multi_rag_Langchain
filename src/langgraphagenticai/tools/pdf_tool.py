import os
from typing import Dict

import pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langgraphagenticai.utils.pdf_utils import load_and_split_pdf

# ─────────────────────────────────────────────────────────────────────────────
#   CONFIGURATION FROM ENV
# ─────────────────────────────────────────────────────────────────────────────
PINECONE_API_KEY       = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT   = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
PINECONE_INDEX_NAME    = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL_ID     = "sentence-transformers/all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
#   INITIALIZE PINECONE CLIENT & INDEX
# ─────────────────────────────────────────────────────────────────────────────
pinecone.init(
    api_key     = PINECONE_API_KEY,
    environment = PINECONE_ENVIRONMENT,
)
pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)

# ─────────────────────────────────────────────────────────────────────────────
#   EMBEDDING MODEL & VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────
embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)

vectordb = PineconeVectorStore(
    client     = pinecone_index,       # ← exactly the pinecone.Index instance
    embedding  = embed_model,
    text_key   = "text",
    index_name = PINECONE_INDEX_NAME,
)

# ─────────────────────────────────────────────────────────────────────────────
#   PDF INGEST & QUERY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: str, namespace: str = "default") -> Dict[str, int]:
    """
    Load PDF, split into chunks, embed each chunk, and upsert to Pinecone.
    """
    docs = load_and_split_pdf(pdf_path)
    vectors = []
    for i, doc in enumerate(docs):
        emb = embed_model.embed_documents([doc.page_content])[0]
        vectors.append({
            "id":     f"{os.path.basename(pdf_path)}-{i}",
            "values": emb,
            "metadata": {
                "text":   doc.page_content,
                "source": os.path.basename(pdf_path),
            },
        })
    # upsert via Pinecone’s Python client
    pinecone_index.upsert(vectors=vectors, namespace=namespace)
    return {"ingested_chunks": len(vectors)}

def query_pdf(query: str, namespace: str = "default") -> str:
    """
    Run a RetrievalQA chain over the Pinecone index and return the answer.
    """
    retriever = vectordb.as_retriever(namespace=namespace)
    qa_chain  = RetrievalQA.from_chain_type(
        llm                    = ChatOpenAI(),
        chain_type             = "stuff",
        retriever              = retriever,
        return_source_documents= False,
    )
    return qa_chain.run(query)
