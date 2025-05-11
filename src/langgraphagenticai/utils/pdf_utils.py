# src/langgraphagenticai/utils/pdf_utils.py

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(path: str):
    try:
        loader = PyMuPDFLoader(path)
        documents = loader.load()
        if not documents:
            raise ValueError("No content extracted from PDF.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(documents)
    except Exception as e:
        raise RuntimeError(f"PDF loading failed: {e}")
