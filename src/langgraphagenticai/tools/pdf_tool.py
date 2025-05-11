# src/langgraphagenticai/tools/pdf_tool.py

import logging
from langchain.chains import RetrievalQA
from langgraphagenticai.vectorstore.pinecone_store import get_vectordb
from langgraphagenticai.LLMS.load_models import load_openai
from typing import Optional

logger = logging.getLogger(__name__)

class PDFQueryProcessor:
    def __init__(self):
        self.llm = load_openai()
        self.retriever_kwargs = {
            "search_type": "similarity",
            "search_kwargs": {
                "k": 4,
                "filter": {"model": "llama-text-embed-v2"}
            }
        }

    def process_query(self, query: str, pdf_path: str) -> Optional[str]:
        """Process PDF query with enhanced error handling"""
        try:
            logger.info(f"📄 Processing PDF: {pdf_path}")
            
            # Initialize vector store
            vectordb = get_vectordb(pdf_path)
            if not vectordb:
                raise ValueError("Failed to initialize vector store")
                
            # Configure retriever
            retriever = vectordb.as_retriever(**self.retriever_kwargs)
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
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

# Singleton instance for better performance
pdf_processor = PDFQueryProcessor()

def query_pdf(query: str, pdf_path: str) -> str:
    """Public interface for PDF querying"""
    return pdf_processor.process_query(query, pdf_path)