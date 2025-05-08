import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraphagenticai.tools.pdf_tool import load_and_split_pdf

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX_NAME)

def get_vectordb(pdf_path: str):
    docs = load_and_split_pdf(pdf_path)
    vectors = [
        {
            "id": f"doc-{i}",
            "values": embedding_model.embed_query(doc.page_content),
            "metadata": {"text": doc.page_content}
        }
        for i, doc in enumerate(docs)
    ]
    index.upsert(vectors=vectors, namespace="default")

    return PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        namespace="default",
        text_key="text"
    )
