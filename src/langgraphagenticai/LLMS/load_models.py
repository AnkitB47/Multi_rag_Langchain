import os
from langchain_openai import ChatOpenAI
from google.generativeai import GenerativeModel

def load_openai():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )

def load_gemini():
    return GenerativeModel("gemini-pro")

def load_gemini_vision():
    return GenerativeModel("gemini-pro-vision")
