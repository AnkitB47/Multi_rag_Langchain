from langchain.chains import RetrievalQA
from langgraphagenticai.vectorstore.pinecone_store import get_vectordb
from langgraphagenticai.LLMS.load_models import load_openai
import logging

logger = logging.getLogger(__name__)

def query_pdf(query, pdf_path: str):
    try:
        vectordb = get_vectordb(pdf_path)
        llm = load_openai()
        retriever = vectordb.as_retriever()
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return chain.run(query)
    except Exception as e:
        logger.exception(f"PDF query failed: {e}")
        return "❌ Failed to process PDF. Please try again."
