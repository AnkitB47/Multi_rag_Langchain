import fitz  # PyMuPDF
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_split_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Read a PDF from disk, extract all text, split into overlapping chunks,
    and wrap each chunk in a LangChain Document.
    """
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(full_text)
    return [Document(page_content=chunk) for chunk in chunks]
