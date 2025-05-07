import os
from langchain_community.chat_models import ChatOpenAI
from google.generativeai import GenerativeModel, configure

configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_openai():
    return ChatOpenAI(temperature=0.5, model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

def load_gemini_vision():
    return GenerativeModel(model_name="models/gemini-pro-vision")

def load_gemini():
    return GenerativeModel(model_name="models/gemini-pro")
