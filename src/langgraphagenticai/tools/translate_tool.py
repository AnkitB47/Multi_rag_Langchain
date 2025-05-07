from langgraphagenticai.LLMS.load_models import load_gemini

def translate_text(text, lang):
    model = load_gemini()
    prompt = f"Translate this to {lang}:\n{text}"
    response = model.generate_content(prompt)
    return response.text
