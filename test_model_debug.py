import sys
sys.path.insert(0, 'd:/RAG')

import os
from dotenv import load_dotenv
load_dotenv('d:/RAG/.env')

import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
print(f"API key present: {bool(api_key)}")

genai.configure(api_key=api_key)

# Try to find available models
try:
    models = list(genai.list_models())
    gen_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    print(f"Available models for generateContent:")
    for m in gen_models:
        print(f"  - {m}")
except Exception as e:
    print(f"list_models error: {e}")

# Test direct classify for 'who is Elon Musk'
for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.0-pro"]:
    try:
        model = genai.GenerativeModel(model_name)
        prompt = """You are a strict query classifier. Classify this query:
"who is Elon Musk"

Is this about: (A) a real person -> IRRELEVANT, (B) a concept -> EDUCATIONAL, (C) research -> RESEARCH_CLAIM?
Reply with ONLY: IRRELEVANT"""
        resp = model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=10))
        print(f"\n{model_name} raw response: '{resp.text.strip()}'")
        break
    except Exception as e:
        print(f"{model_name} failed: {e}")
