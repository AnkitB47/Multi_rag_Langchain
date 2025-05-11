# src/langgraphagenticai/tools/pdf_tool.py

import logging
from langchain.chains import RetrievalQA
from langgraphagenticai.vectorstore.pinecone_store import get_vectordb
from langgraphagenticai.LLMS.load_models import load_openai

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def query_pdf(query: str, pdf_path: str) -> str:
    """
    Uses Pinecone's built-in embedding + OpenAI LLM to perform QA over a PDF.
    """
    try:
        logger.info(f"📄 PDF Path: {pdf_path}")

        vectordb = get_vectordb(pdf_path)
        llm = load_openai()

        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        logger.info("🔍 Running retrieval-based QA on embedded PDF...")
        result = qa_chain.run(query)

        if not result:
            raise ValueError("❗ Empty response from QA chain.")
        return result

    except Exception as e:
        logger.exception(f"❌ PDF processing failed: {e}")
        return "❌ Failed to process PDF. Please try again."
