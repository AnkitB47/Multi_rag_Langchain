# src/langgraphagenticai/utils/pdf_utils.py

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def load_and_split_pdf(path: str):
    """
    Loads a PDF file and splits content into chunks for vector indexing.
    """
    try:
        loader = PyMuPDFLoader(path)
        documents = loader.load()

        if not documents:
            raise ValueError("PDF appears to be empty or unreadable.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("PDF text split returned no chunks.")

        logger.info(f"✅ Loaded and split PDF: {len(chunks)} chunks created.")
        return chunks

    except Exception as e:
        logger.exception(f"❌ Error loading/splitting PDF: {e}")
        raise
