# src/langgraphagenticai/tools/pdf_tool.py

import logging
from langchain.chains import RetrievalQA
from langgraphagenticai.vectorstore.pinecone_store import get_vectordb
from langgraphagenticai.LLMS.load_models import load_openai
from typing import Optional

logger = logging.getLogger(__name__)

def query_pdf(query: str, pdf_path: str) -> Optional[str]:
    try:
        logger.info(f"📄 Processing PDF: {pdf_path}")
        
        # Initialize vector store
        vectordb = get_vectordb(pdf_path)
        if not vectordb:
            raise ValueError("Failed to initialize vector store")
            
        # Load LLM
        llm = load_openai()
        if not llm:
            raise ValueError("Failed to load LLM")
            
        # Configure retriever
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Execute query
        result = qa_chain({"query": query})
        return result.get("result", "No answer found")
        
    except Exception as e:
        logger.exception(f"❌ PDF processing failed: {e}")
        return f"❌ Error processing PDF: {str(e)}"