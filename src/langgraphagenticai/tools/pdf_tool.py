import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
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
    Load PDF, split into chunks, embed, and upsert into Pinecone.
    Uses direct upsert for better control over vector data.
    """
    docs = load_and_split_pdf(pdf_path)
    vectors = []
    
    for i, doc in enumerate(docs):
        embedding = embeddings.embed_documents([doc.page_content])[0]
        vectors.append({
            "id": f"doc-{i}-{abs(hash(doc.page_content))}",   # Positive hash
            "values": embedding,
            "metadata": {"text": doc.page_content}
        })
    
    index.upsert(vectors=vectors, namespace=namespace)

def query_pdf(query: str, namespace: str = "default") -> str:
    """
    Run a RetrievalQA chain over Pinecone index and return the answer.
    """
    retriever = vectordb.as_retriever(namespace=namespace)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa.run(query)
