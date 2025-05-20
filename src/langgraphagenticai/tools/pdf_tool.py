import os
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone

from langgraphagenticai.utils.pdf_utils import load_and_split_pdf

# ─────────────────────────────────────────────────────────────────────────────
#   CONFIGURATION FROM ENV
# ─────────────────────────────────────────────────────────────────────────────
PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX     = os.environ["PINECONE_INDEX_NAME"]
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
#   SETUP PINECONE & VECTORSTORE
# ─────────────────────────────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
vectordb   = PineconeVectorStore(
    index=index,
    embedding=embeddings.embed_query,
    text_key="text"
)

# ─────────────────────────────────────────────────────────────────────────────
#   PDF INGEST & QUERY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: str, namespace: str = "default") -> None:
    """
    Load a PDF, split it into chunks, embed them, and upsert into Pinecone.
    """
    docs = load_and_split_pdf(pdf_path)
    vectordb.add_documents(docs, namespace=namespace)

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
