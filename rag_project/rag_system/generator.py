import os
from groq import Groq
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        # API Anahtarlarını Kontrol Et
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def generate(self, docs, query):
        # Sidebar'dan (app.py) gelen seçimi oku
        selected_model = os.getenv("CURRENT_MODEL", "Llama-3.3-70b")
        
        # Bağlamı (context) birleştir
        context = "\n".join([
          doc["text"] if isinstance(doc, dict) and "text" in doc
          else doc.page_content if hasattr(doc, "page_content")
          else str(doc)
          for doc in docs])
        
        system_prompt = "Sen profesyonel bir veri analistisin. Sadece verilen bağlama dayanarak yanıt ver."
        user_prompt = f"Bağlam:\n{context}\n\nSoru: {query}"

        # --- 1. GROQ MODELLERİ ---
        if "Llama" in selected_model:
            client = Groq(api_key=self.groq_key)
            model_id = "llama-3.3-70b-versatile" if "70b" in selected_model else "llama-3.1-8b-instant"
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            return response.choices[0].message.content

        # --- 2. GEMINI MODELLERİ ---
        elif "Gemini" in selected_model:
            genai.configure(api_key=self.gemini_key)
            model_id = "models/gemini-2.5-pro" if "Pro" in selected_model else "models/gemini-2.5-flash"
            model = genai.GenerativeModel(model_id)
            response = model.generate_content(user_prompt)
            return response.text

        # --- 3. OPENAI MODELLERİ ---
        elif "GPT" in selected_model:
            client = OpenAI(api_key=self.openai_key)
            if "GPT-5" in selected_model:
                model_id = "gpt-5.4" 
            elif "4o-mini" in selected_model:
                model_id = "gpt-4o-mini"
            else:
                model_id = "gpt-4o"
            
            resp = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": user_prompt}])
            return resp.choices[0].message.content

        return "HATA: Model eşleşmedi."

generator = Generator()