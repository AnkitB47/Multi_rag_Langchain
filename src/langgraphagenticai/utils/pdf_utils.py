# src/langgraphagenticai/utils/pdf_utils.py

import fitz
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]
