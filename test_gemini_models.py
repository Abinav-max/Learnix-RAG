import sys, os
sys.path.insert(0, 'd:/RAG')
from dotenv import load_dotenv
load_dotenv('d:/RAG/.env')
import google.generativeai as genai
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

for m in ['gemini-2.5-flash-lite', 'gemini-flash-latest', 'gemini-2.5-pro', 'gemini-flash-lite-latest']:
    try:
        mod = genai.GenerativeModel(m)
        r = mod.generate_content('hello')
        print(f'SUCCESS with {m}: {r.text.strip()}')
    except Exception as e:
        print(f'FAILED with {m}: {e}')
