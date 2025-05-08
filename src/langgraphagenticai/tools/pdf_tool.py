from langchain.chains import RetrievalQA
from langgraphagenticai.vectorstore.pinecone_store import get_vectordb
from langgraphagenticai.LLMS.load_models import load_openai

def query_pdf(query, pdf_path):
    vectordb = get_vectordb(pdf_path)
    llm = load_openai()
    retriever = vectordb.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain.run(query)
