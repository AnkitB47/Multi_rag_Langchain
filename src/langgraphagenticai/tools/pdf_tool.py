import fitz
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langgraphagenticai.vectorstore.pinecone_store import get_vectordb
from langgraphagenticai.LLMS.load_models import load_openai

def load_and_split_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

def query_pdf(query, pdf_path):
    vectordb = get_vectordb(pdf_path)
    llm = load_openai()
    retriever = vectordb.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain.run(query)
