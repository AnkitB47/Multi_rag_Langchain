from langgraphagenticai.LLMS.load_models import load_gemini
import logging

logger = logging.getLogger(__name__)

def translate_text(text: str, target_lang: str):
    try:
        model = load_gemini()
        prompt = f"Translate this to {target_lang}: {text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.exception(f"Translation failed: {e}")
        return f"❌ Translation failed. Response in original language:\n{text}"
