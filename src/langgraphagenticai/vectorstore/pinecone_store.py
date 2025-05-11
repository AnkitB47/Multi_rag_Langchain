# src/langgraphagenticai/vectorstore/pinecone_store.py

import os
import logging
import math
from typing import List, Optional
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - MUST match your Pinecone index
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multiagentrag"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 1536
MAX_REQUEST_SIZE = 1.8 * 1024 * 1024  # 1.8MB safety margin

class PineconeVectorManager:
    def __init__(self):
        """Initialize with proper configuration and safety checks"""
        try:
            # Validate environment
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            # Initialize embeddings with strict dimension
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSION
            )
            
            # Pinecone connection with retry config
            self.pc = Pinecone(
                api_key=PINECONE_API_KEY,
                additional_headers={
                    "X-Pinecone-Request-Source": "langgraph-agent"
                }
            )
            
            # PDF processing setup
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks prevent size issues
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            self._verify_index()
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise RuntimeError(f"System initialization error: {str(e)}")

    def _verify_index(self):
        """Ensure index exists with correct configuration"""
        try:
            active_indexes = self.pc.list_indexes().names()
            if PINECONE_INDEX_NAME not in active_indexes:
                error_msg = (
                    f"Index '{PINECONE_INDEX_NAME}' not found.\n"
                    "Required configuration:\n"
                    f"- Dimension: {EMBEDDING_DIMENSION}\n"
                    "- Metric: cosine\n"
                    "- Cloud: AWS\n"
                    "- Region: us-east-1\n"
                    "- Pod type: Serverless"
                )
                raise ValueError(error_msg)
                
            # Verify index status
            desc = self.pc.describe_index(PINECONE_INDEX_NAME)
            if not desc.status['ready']:
                raise ValueError(f"Index {PINECONE_INDEX_NAME} is not ready")
                
        except Exception as e:
            logger.error(f"Index verification failed: {e}")
            raise

    def _calculate_batch_size(self, docs: List) -> int:
        """Dynamically determine safe batch size based on content"""
        try:
            avg_doc_size = sum(len(str(doc.page_content)) for doc in docs) / len(docs)
            batch_size = max(1, math.floor(MAX_REQUEST_SIZE / (avg_doc_size * 1.2)))  # 20% buffer
            logger.debug(f"Calculated batch size: {batch_size}")
            return min(batch_size, 100)  # Never exceed 100 docs/batch
        except Exception as e:
            logger.warning(f"Batch calculation failed, using safe default: {e}")
            return 10  # Fallback value

    def _process_document(self, pdf_path: str) -> List:
        """Robust PDF loading and splitting"""
        try:
            # Validate file first
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
                
            if os.path.getsize(pdf_path) > 10 * 1024 * 1024:  # 10MB max
                raise ValueError("PDF exceeds maximum size (10MB)")

            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load_and_split(self.text_splitter)
            
            if not docs:
                raise ValueError("No readable content extracted from PDF")
                
            # Add metadata for tracking
            for doc in docs:
                doc.metadata.update({
                    "source_file": os.path.basename(pdf_path),
                    "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
            return docs
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise RuntimeError(f"Could not process PDF: {str(e)}")

    def get_vectorstore(self, pdf_path: str) -> PineconeVectorStore:
        """Create and return a fully configured vector store"""
        try:
            logger.info(f"Starting PDF processing: {pdf_path}")
            
            # Step 1: Load and split document
            docs = self._process_document(pdf_path)
            
            # Step 2: Calculate safe batch size
            batch_size = self._calculate_batch_size(docs)
            
            # Step 3: Create vector store with batch control
            return PineconeVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings,
                index_name=PINECONE_INDEX_NAME,
                namespace="default",
                text_key="text",
                batch_size=batch_size,  # Critical for large PDFs
                embedding_chunk_size=512  # Optimal for text-embedding-3
            )
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise RuntimeError(f"Failed to create vector store: {str(e)}")

# Singleton with thread-safe initialization
import threading
vector_manager = None
vector_manager_lock = threading.Lock()

def get_vectordb(pdf_path: str) -> PineconeVectorStore:
    """Thread-safe public interface with lazy initialization"""
    global vector_manager
    try:
        with vector_manager_lock:
            if vector_manager is None:
                vector_manager = PineconeVectorManager()
        return vector_manager.get_vectorstore(pdf_path)
    except Exception as e:
        logger.error(f"Service unavailable: {str(e)}")
        raise RuntimeError(f"Document processing service is currently unavailable. Please try again later.")