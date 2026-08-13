import sys, os
sys.path.insert(0, 'd:/RAG')
from dotenv import load_dotenv
load_dotenv('d:/RAG/.env')
import google.generativeai as genai
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
mod = genai.GenerativeModel('gemini-flash-latest')

prompt_template = """You are a strict query classifier for an academic peer-review research system.
Classify the following query into EXACTLY ONE of these four categories:

EDUCATIONAL - The query asks for an explanation, definition, overview, or introduction to any CONCEPT, TECHNOLOGY, DRUG, PLANT, PROGRAMMING LANGUAGE, SCIENTIFIC TOPIC, or SUBJECT.
Examples: "what is Python", "what is RAG", "what is paracetamol", "explain neural networks", "what is hibiscus", "what is deep learning"

RESEARCH_CLAIM - The query is about a specific research claim, model comparison, benchmark, methodological flaw, limitation, tradeoff, or academic hypothesis.
Examples: "limitations of GPT-4", "do Transformers beat GNNs for forecasting", "benchmark contamination in LLMs"

IRRELEVANT - The query is about: (1) a real person, (2) current events, (3) geography or place facts, (4) personal identity or face checks, (5) chit-chat, OR (6) a simple yes/no question about an established physical or scientific fact that requires no peer review.
Examples: "who is the CM of Tamil Nadu", "who is Elon Musk", "face check", "Is the sky blue?", "Is water wet?"

AMBIGUOUS_ACRONYM - A bare acronym or abbreviation without enough context.

CRITICAL RULE 1: If the query asks about a PERSON or a POLITICAL ROLE, it is ALWAYS IRRELEVANT.
CRITICAL RULE 2: If the query is a simple yes/no question about a well-known physical or scientific fact (e.g. "Is the sky blue?"), it is ALWAYS IRRELEVANT.

Reply with ONLY the category name. No explanation. No punctuation.

Query: {query}"""

test_queries = [
    'Is the sky blue?',
    'what is python',
    'who is the CM of Tamil Nadu',
    'limitations of GPT-4',
    'Is water wet?',
    'what is paracetamol'
]

for q in test_queries:
    r = mod.generate_content(prompt_template.format(query=q))
    print(f"'{q}' --> {repr(r.text.strip())}")
