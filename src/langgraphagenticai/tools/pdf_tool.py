# src/langgraphagenticai/tools/pdf_tool.py
import os
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
    try:
        # First confirm PDF is processed
        if not os.path.exists(pdf_path):
            return "Error: PDF not found"
            
        # Explicit loading message
        print(f"Processing PDF: {pdf_path}")  # Log for debugging
        
        vectordb = get_vectordb(pdf_path)
        
        # Test retrieval
        test_docs = vectordb.similarity_search("attention", k=1)
        if not test_docs:
            return "Error: PDF content not loaded properly"
            
        # Proceed with actual query
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_openai(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True  # Critical for verification
        )
        
        result = qa_chain({"query": query})
        
        # Add source verification
        if not result.get('source_documents'):
            return "I couldn't find relevant content in the PDF. Try asking about specific sections."
            
        return f"From the PDF: {result['result']}\n\nSources: {[d.metadata for d in result['source_documents']]}"
        
    except Exception as e:
        return f"Processing error: {str(e)}"