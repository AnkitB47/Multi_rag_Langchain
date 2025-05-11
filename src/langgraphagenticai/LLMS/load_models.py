import os
from langchain_openai import ChatOpenAI
from google.generativeai import GenerativeModel
from google.generativeai import configure

# Load API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure Gemini globally
if GOOGLE_API_KEY:
    configure(api_key=GOOGLE_API_KEY)

def load_openai():
    return ChatOpenAI(
        model="gpt-4o",  # OpenAI multimodal
        temperature=0.3,
        api_key=OPENAI_API_KEY
    )

def load_gemini():
    return GenerativeModel(
        model_name="gemini-pro",  # Text-only model
        generation_config={
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 2048
        }
    )

def load_gemini_vision():
    return GenerativeModel(
        model_name="gemini-pro-vision",  # Vision-compatible model
        generation_config={
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 2048
        }
    )
