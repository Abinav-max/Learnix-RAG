import sys
sys.path.insert(0, 'd:/RAG')
from backend.gemini_agent import classify_query_gemini

print('=== Classifier Tests ===')
tests = [
    ('what is Python', 'EDUCATIONAL'),
    ('what is RAG', 'EDUCATIONAL'),
    ('what is paracetamol', 'EDUCATIONAL'),
    ('who is CM of Tamil Nadu', 'IRRELEVANT'),
    ('who is Elon Musk', 'IRRELEVANT'),
    ('face check', 'IRRELEVANT'),
    ('limitations of GPT-4', 'RESEARCH_CLAIM'),
    ('do Transformers beat GNNs for forecasting', 'RESEARCH_CLAIM'),
    ('benchmark contamination in LLMs', 'RESEARCH_CLAIM'),
]

passed = 0
failed = 0
for query, expected in tests:
    label, reason = classify_query_gemini(query)
    status = 'PASS' if label == expected else 'FAIL'
    if label == expected:
        passed += 1
    else:
        failed += 1
    print(f'{status} [expected={expected}] [got={label}] - "{query}"')

print(f'\n=== {passed}/{len(tests)} passed, {failed} failed ===')
