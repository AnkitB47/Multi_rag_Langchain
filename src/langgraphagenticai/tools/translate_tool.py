import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
gemini_pro = genai.GenerativeModel("gemini-pro")

def translate_text(text: str, target_lang: str) -> str:
    try:
        model = load_gemini()
        prompt = f"Translate this to {target_lang}: {text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Translation failed: {e}"
