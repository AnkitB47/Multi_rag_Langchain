# src/langgraphagenticai/utils/pdf_utils.py

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import logging

logger = logging.getLogger(__name__)

def validate_pdf(path: str) -> bool:
    """Basic PDF validation"""
    try:
        with open(path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False

def load_and_split_pdf(path: str) -> List:
    """Load and split PDF with enhanced error handling"""
    try:
        if not validate_pdf(path):
            raise ValueError("Invalid PDF file")
            
        logger.info(f"Loading PDF: {path}")
        loader = PyMuPDFLoader(path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content extracted from PDF")
            
        logger.info(f"Loaded {len(documents)} pages")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        
        splits = splitter.split_documents(documents)
        logger.info(f"Split into {len(splits)} chunks")
        return splits
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise RuntimeError(f"Failed to process PDF: {str(e)}")