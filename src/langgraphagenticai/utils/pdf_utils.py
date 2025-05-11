# src/langgraphagenticai/utils/pdf_utils.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging
import fitz  # PyMuPDF
import os

logger = logging.getLogger(__name__)

def validate_pdf(path: str) -> bool:
    """Enhanced PDF validation with proper file checking"""
    try:
        # Check file exists and is readable
        if not os.path.isfile(path) or not os.access(path, os.R_OK):
            return False
            
        # Check PDF header and try to open
        with fitz.open(path) as doc:
            return len(doc) > 0  # Verify it has pages
    except Exception as e:
        logger.debug(f"PDF validation failed: {e}")
        return False

def load_and_split_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Load and split PDF with enhanced metadata handling
    Returns list of documents with proper metadata for Pinecone
    """
    try:
        logger.info(f"Processing PDF: {path}")
        
        if not validate_pdf(path):
            raise ValueError("Invalid or corrupted PDF file")
            
        # Load with enhanced metadata
        loader = PyMuPDFLoader(path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("PDF contained no extractable text")
            
        logger.info(f"Loaded {len(documents)} pages")
        
        # Configure splitter for optimal RAG performance
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Slightly smaller for better embedding quality
            chunk_overlap=150,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]  # Better paragraph handling
        )
        
        splits = splitter.split_documents(documents)
        
        # Enhance metadata for Pinecone compatibility
        for doc in splits:
            doc.metadata.update({
                "source": os.path.basename(path),
                "document_type": "pdf",
                "processing_method": "langchain-pymupdf"
            })
        
        logger.info(f"Split into {len(splits)} chunks")
        return splits
        
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        raise RuntimeError(f"Failed to process PDF: {str(e)}")