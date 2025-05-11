# src/langgraphagenticai/tools/pdf_tool.py

import logging
from langchain.chains import RetrievalQA
from langgraphagenticai.vectorstore.pinecone_store import get_vectordb
from langgraphagenticai.LLMS.load_models import load_openai

logger = logging.getLogger(__name__)

def query_pdf(query: str, pdf_path: str) -> str:
    try:
        logger.info(f"📄 PDF saved to: {pdf_path}")
        vectordb = get_vectordb(pdf_path)
        llm = load_openai()

        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        return qa_chain.run(query)
    except Exception as e:
        logger.exception(f"❌ Failed to process PDF: {e}")
        return "❌ Failed to process PDF. Please try again."
