import sys, os
sys.path.insert(0, 'd:/RAG')
from dotenv import load_dotenv
load_dotenv('d:/RAG/.env')

from backend.gemini_agent import run_gemini_devils_advocate

print("=== Running Multi-Source Search Test Across All 11+ APIs ===")
res = run_gemini_devils_advocate("transformers for forecasting", source_filter="All")
results = res.get("results", [])

sources = set(r.get("source") for r in results)
print(f"Total results returned: {len(results)}")
print(f"Unique sources present in live results: {sources}")
for idx, r in enumerate(results[:10], 1):
    print(f"{idx}. [{r.get('source')}] {r.get('title')} ({r.get('year')})")
