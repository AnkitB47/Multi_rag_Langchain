# src/langgraphagenticai/tools/pdf_tool.py

import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langgraphagenticai.vectorstore.pinecone_store import get_vectordb
from langgraphagenticai.LLMS.load_models import load_openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom prompt template for better results
QA_PROMPT = PromptTemplate(
    template="""Use the following context to answer the question. 
    If you don't know the answer, say you don't know.
    
    Context: {context}
    Question: {question}
    
    Answer:""",
    input_variables=["context", "question"]
)

def query_pdf(query: str, pdf_path: str) -> str:
    """Query PDF with proper error handling and configuration"""
    try:
        logger.info(f"Initializing PDF query for: {pdf_path}")
        
        # Initialize vector store
        vectordb = get_vectordb(pdf_path)
        
        # Configure retriever with proper settings
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
                "filter": {
                    "embedding_model": "llama-text-embed-v2",
                    "model": "llama-text-embed-v2"
                }
            }
        )
        
        # Create QA chain with proper configuration
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_openai(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        
        # Execute query
        result = qa_chain({"query": query})
        return result.get("result", "No answer found")
        
    except Exception as e:
        logger.exception(f"PDF query failed: {e}")
        return f"Error processing PDF: {str(e)}"