import sys
sys.path.insert(0, ".")

import json

# ============================================================
# TEST 1: extract_explicit_ids
# ============================================================
print("=" * 70)
print("TEST 1: extract_explicit_ids — ID Parsing")
print("=" * 70)

from backend.live_agent import extract_explicit_ids, has_explicit_ids

# DOI extraction
q1 = "Fetch source ID #10.52202/079017-2123 and analyze"
ids1 = extract_explicit_ids(q1)
assert ids1["dois"] == ["10.52202/079017-2123"], f"DOI extraction failed: {ids1}"
print(f"  [PASS] DOI extracted: {ids1['dois']}")

# OpenReview extraction
q2 = "Fetch OpenReview forum #w6nlcS8Kkn and cite it"
ids2 = extract_explicit_ids(q2)
assert "w6nlcS8Kkn" in ids2["openreview_ids"], f"OpenReview extraction failed: {ids2}"
print(f"  [PASS] OpenReview ID extracted: {ids2['openreview_ids']}")

# ArXiv extraction
q3 = "Get paper arxiv:2308.10783 abstract"
ids3 = extract_explicit_ids(q3)
assert "2308.10783" in ids3["arxiv_ids"], f"ArXiv extraction failed: {ids3}"
print(f"  [PASS] ArXiv ID extracted: {ids3['arxiv_ids']}")

# Mixed extraction
q4 = "Compare #10.52202/079017-2123 with OpenReview #w6nlcS8Kkn and arxiv 2308.10783"
ids4 = extract_explicit_ids(q4)
assert len(ids4["dois"]) >= 1, f"Mixed DOI failed: {ids4}"
assert len(ids4["openreview_ids"]) >= 1, f"Mixed OpenReview failed: {ids4}"
assert len(ids4["arxiv_ids"]) >= 1, f"Mixed ArXiv failed: {ids4}"
print(f"  [PASS] Mixed extraction: DOI={ids4['dois']}, OR={ids4['openreview_ids']}, ArXiv={ids4['arxiv_ids']}")

# No IDs
q5 = "Does Chain-of-Thought reasoning scale with model size?"
assert not has_explicit_ids(q5), "False positive ID detection"
print(f"  [PASS] No IDs detected in semantic query")

print()

# ============================================================
# TEST 2: keyword_overlap_filter
# ============================================================
print("=" * 70)
print("TEST 2: keyword_overlap_filter — Tag-Blindness Defense")
print("=" * 70)

from backend.live_agent import keyword_overlap_filter, extract_query_keywords

query = "Does Chain-of-Thought reasoning scale with model size?"
keywords = extract_query_keywords(query)
print(f"  Query keywords: {keywords}")

# Simulate the exact garbage papers the user complained about
mock_papers = [
    {"title": "Monoaural Speech Separation Using Gaussian Processes", "text": "We propose a novel approach to speech source separation...", "raw_text": "We propose a novel approach to speech source separation..."},
    {"title": "Humanity's Last Exam: Benchmark Saturation in LLMs", "text": "Large language models achieve near-human benchmark scores...", "raw_text": "Large language models achieve near-human benchmark scores..."},
    {"title": "Self-training for Source-Free Domain Adaptation", "text": "Label propagation improves SFDA accuracy...", "raw_text": "Label propagation improves SFDA accuracy..."},
    {"title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "text": "Chain-of-thought reasoning emerges with model scale, showing improved performance with larger model size...", "raw_text": "Chain-of-thought reasoning emerges with model scale, showing improved performance with larger model size..."},
    {"title": "Does Reasoning Emerge with Scale? Evidence from Chain-of-Thought", "text": "We study whether chain-of-thought reasoning capability scales with model size and compute...", "raw_text": "We study whether chain-of-thought reasoning capability scales with model size and compute..."},
]

filtered = keyword_overlap_filter(query, mock_papers, min_overlap=2)
filtered_titles = [p["title"] for p in filtered]

# Speech separation should be DROPPED
assert "Monoaural Speech Separation Using Gaussian Processes" not in filtered_titles, f"Speech paper not filtered: {filtered_titles}"
print(f"  [PASS] Speech separation paper correctly DROPPED")

# Self-training/SFDA should be DROPPED
assert "Self-training for Source-Free Domain Adaptation" not in filtered_titles, f"SFDA paper not filtered: {filtered_titles}"
print(f"  [PASS] SFDA paper correctly DROPPED")

# CoT papers should SURVIVE
assert any("Chain-of-Thought" in t for t in filtered_titles), f"CoT paper incorrectly filtered: {filtered_titles}"
print(f"  [PASS] Chain-of-Thought paper correctly RETAINED")

# Reasoning+scale paper should SURVIVE
assert any("Reasoning" in t and "Scale" in t for t in filtered_titles), f"Reasoning paper filtered: {filtered_titles}"
print(f"  [PASS] Reasoning+Scale paper correctly RETAINED")

print(f"  Survived: {len(filtered)}/{len(mock_papers)} papers")
print(f"  Titles: {filtered_titles}")

# Exact-ID fetch papers should never be filtered
mock_exact = [{"title": "Unrelated But Exact-Fetched", "text": "...", "raw_text": "...", "exact_id_fetch": True}]
exact_filtered = keyword_overlap_filter(query, mock_exact, min_overlap=2)
assert len(exact_filtered) == 1, "Exact-ID paper was incorrectly filtered!"
print(f"  [PASS] exact_id_fetch papers bypass keyword filter")

print()

# ============================================================
# TEST 3: fetch_arxiv_by_id (live API)
# ============================================================
print("=" * 70)
print("TEST 3: fetch_arxiv_by_id — Direct ArXiv Fetch")
print("=" * 70)

from backend.live_agent import fetch_arxiv_by_id

result = fetch_arxiv_by_id("2308.10783")
if result.get("status") == "UNAVAILABLE":
    print(f"  [SKIP] ArXiv API unavailable: {result.get('error')}")
else:
    assert "Bangla" in result["title"] or "Sentiment" in result["title"], f"Wrong paper: {result['title']}"
    assert result.get("exact_id_fetch") == True
    assert result["source"] == "ArXiv"
    print(f"  [PASS] Fetched: {result['title'][:70]}...")
    print(f"  Year: {result['year']}, Authors: {result['authors']}")

print()

# ============================================================
# TEST 4: fetch_by_doi (live API)
# ============================================================
print("=" * 70)
print("TEST 4: fetch_by_doi — Direct DOI Fetch")
print("=" * 70)

from backend.live_agent import fetch_by_doi

# Use a well-known, stable DOI
result = fetch_by_doi("10.1038/s41586-021-03819-2")
if result.get("status") == "UNAVAILABLE":
    print(f"  [SKIP] CrossRef API unavailable: {result.get('error')}")
else:
    assert result.get("exact_id_fetch") == True
    assert result["source"] == "CrossRef/DOI"
    print(f"  [PASS] Fetched: {result['title'][:70]}...")
    print(f"  Year: {result['year']}, Publisher: {result['publisher']}")

print()

# ============================================================
# TEST 5: Preview Search Endpoint (via FastAPI test client)
# ============================================================
print("=" * 70)
print("TEST 5: /api/preview-search Endpoint")
print("=" * 70)

try:
    import urllib.request
    import urllib.parse
    
    # Test with semantic query via local server endpoint
    req_data = json.dumps({
        "query": "Does Chain-of-Thought reasoning scale with model size?",
        "source": "All",
        "attack_vector": "All"
    }).encode('utf-8')
    
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/preview-search",
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert "candidates" in data, "No candidates key in response"
        assert "domain" in data, "No domain key in response"
        assert "query_keywords" in data, "No query_keywords key in response"
        assert data["has_explicit_ids"] == False
        print(f"  [PASS] Preview endpoint returned {data['total_candidates']} candidates")
        print(f"  Domain: {data['domain']}, Keywords: {data['query_keywords']}")
    
    # Test with exact-ID query
    req_data2 = json.dumps({
        "query": "Fetch arxiv 2308.10783",
        "source": "All"
    }).encode('utf-8')
    
    req2 = urllib.request.Request(
        "http://127.0.0.1:8000/api/preview-search",
        data=req_data2,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req2, timeout=5) as resp2:
        assert resp2.status == 200
        data2 = json.loads(resp2.read().decode('utf-8'))
        assert data2["has_explicit_ids"] == True
        print(f"  [PASS] Exact-ID preview: has_explicit_ids={data2['has_explicit_ids']}")
        for c in data2["candidates"]:
            if c.get("exact_id_fetch"):
                print(f"    EXACT: {c['title'][:60]}... (score={c['relevance_score']})")

except Exception as e:
    print(f"  [SKIP/FAIL] /api/preview-search call: {e}")

# ============================================================
# TEST 6: Full pipeline integration test — exact ID bypass
# ============================================================
print("=" * 70)
print("TEST 6: Full Pipeline — Exact ID Bypass (no fuzzy search)")
print("=" * 70)

from backend.adversarial_rag import run_adversarial_search

# Query with explicit ArXiv ID
result = run_adversarial_search("Fetch arxiv 2308.10783 and analyze")
papers = result.get("results", [])
if papers:
    for p in papers:
        if p.get("exact_id_fetch"):
            print(f"  [PASS] Exact fetch: {p['title'][:60]}...")
            assert p["relevance_score"] == 0.99
        elif "UNAVAILABLE" in p.get("title", ""):
            print(f"  [INFO] Source unavailable: {p['title']}")
        else:
            print(f"  [FAIL] Non-exact paper slipped through: {p['title'][:60]}")
else:
    print(f"  [INFO] No results (API may be rate-limited)")

print()
print("=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
