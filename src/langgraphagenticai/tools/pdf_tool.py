# src/langgraphagenticai/tools/pdf_tool.py

import logging
from langchain.chains import RetrievalQA
from langgraphagenticai.vectorstore.pinecone_store import get_vectordb
from langgraphagenticai.LLMS.load_models import load_openai

logger = logging.getLogger(__name__)

def query_pdf(query: str, pdf_path: str) -> str:
    """
    Processes a PDF file, embeds its content using Pinecone's built-in model,
    and performs retrieval-based question answering using OpenAI LLM.
    """
    try:
        # Load PDF into Pinecone vectorstore
        vectordb = get_vectordb(pdf_path)

        # Load OpenAI LLM
        llm = load_openai()

        # Create retriever and QA chain
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Run query and return result
        return qa_chain.run(query)

    except Exception as e:
        logger.exception(f"PDF processing failed: {e}")
        return "❌ Failed to process PDF. Please try again."
