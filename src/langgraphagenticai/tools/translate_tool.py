import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
gemini_pro = genai.GenerativeModel("gemini-pro")

def translate_text(text: str, target_lang: str) -> str:
    try:
        if target_lang != "en":
            prompt = f"Translate this to {target_lang}: {text}"
            response = gemini_pro.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"Translation failed: {e}"
