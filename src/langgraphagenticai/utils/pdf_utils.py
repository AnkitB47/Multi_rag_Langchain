import fitz  # PyMuPDF
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def is_valid_pdf(path: str) -> bool:
    try:
        with fitz.open(path) as doc:
            return doc.page_count > 0
    except Exception:
        return False

def load_and_split_pdf(path: str):
    if not is_valid_pdf(path):
        raise ValueError("Invalid or corrupted PDF file.")

    loader = PyMuPDFLoader(path)
    documents = loader.load()

    if not documents:
        raise ValueError("No documents found in PDF.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks found after splitting PDF.")
    return chunks
