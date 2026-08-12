"""
Realtime Agentic Database & Search Dispatcher:
Connects to live ArXiv API (domain filtered), OpenReview API, and PubMed API.
Uses Adversarial Sentiment Gate & Cross-Encoder Re-Ranker to select Top 3-4 Actually Relevant Critiques.
"""

import re
from typing import List, Dict, Any, Optional
from backend.live_agent import (
    fetch_arxiv_realtime, 
    fetch_openreview_realtime, 
    fetch_pubpeer_realtime,
    detect_fine_grained_attack_vector,
    rerank_results_cross_encoder,
    is_supportive_marketing_fluff,
    detect_query_domain,
    get_paper_domain,
    filter_by_domain,
    build_adversarial_query,
    get_domain_categories,
    keyword_overlap_filter,
    has_explicit_ids,
    fetch_all_exact_ids
)

from backend.db import save_critique_chunk_db, get_all_critiques_db

# Runtime wrapper for backwards compatibility
class CritiqueDatabaseList(list):
    def append(self, item):
        save_critique_chunk_db(item)
        super().append(item)

    def __iter__(self):
        db_items = get_all_critiques_db()
        return iter(db_items if db_items else super().__iter__())

    def __len__(self):
        db_items = get_all_critiques_db()
        return len(db_items) if db_items else super().__len__()

CRITIQUE_DATABASE = CritiqueDatabaseList()

def search_critiques_db(query: str, attack_vector: Optional[str] = None, source: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Realtime Agentic Search & Adversarial Sentiment Pipeline:
    0. If query contains explicit IDs (DOI, OpenReview forum, ArXiv), fetch those directly — no fuzzy search.
    1. Detects query domain (e.g. AI/ML/COMPUTER_SCIENCE).
    2. Resolves ArXiv category codes for that domain.
    3. Fetches papers ONLY from matching categories (prevents physics/astrophysics spillover at the API level).
    4. Applies post-fetch domain filter as a safety net.
    4b. Applies keyword overlap filter (≥2 query keywords must appear in title+abstract).
    5. Passes papers through Adversarial Sentiment Gate (filters out pro-paper marketing fluff).
    6. Runs TF-IDF Cosine Similarity Re-Ranker against user query.
    7. Applies hard relevance threshold (≥0.12).
    8. Retains Top 3-4 Genuine Adversarial Critiques with precise flaw tags.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    # 0. Exact-ID bypass — if query contains explicit IDs, fetch those directly
    if has_explicit_ids(clean_query):
        exact_results, unavailable_ids = fetch_all_exact_ids(clean_query)
        processed = []
        for item in exact_results:
            raw_text = item.get("raw_text", "")
            title = item.get("title", "")
            vector, severity, adversarial_tag, skep_score = detect_fine_grained_attack_vector(title, raw_text)
            
            raw_authors = item.get("authors", ["Academic Researchers"])
            authors_str = ", ".join(raw_authors[:3]) if isinstance(raw_authors, list) else str(raw_authors)
            publisher_str = item.get("publisher") or f"{item.get('source', 'Academic')} Forum"
            
            critique_chunk = {
                "id": item.get("id", "exact-fetch"),
                "title": title,
                "authors": authors_str,
                "publisher": publisher_str,
                "year": item.get("year", 2024),
                "source": item.get("source", "Direct Fetch"),
                "source_id": item.get("source_id", ""),
                "url": item.get("url", ""),
                "section": item.get("section", "Direct ID Fetch"),
                "attack_vector": vector,
                "target": f"{title[:40]}...",
                "risk_level": severity,
                "skepticism_score": skep_score,
                "replication_prob": round(100.0 - skep_score, 1),
                "paragraph_type": "Limitation/Critique",
                "adversarial_tag": adversarial_tag,
                "text": raw_text[:380] + ("..." if len(raw_text) > 380 else ""),
                "raw_text": raw_text,
                "query_keywords": [w.lower() for w in clean_query.split() if len(w) > 2],
                "severity": severity,
                "relevance_score": 0.99,
                "exact_id_fetch": True,
                "mitigation_suggestion": "Verify source content directly."
            }
            processed.append(critique_chunk)
            save_critique_chunk_db(critique_chunk)
        
        # Add unavailable markers
        for uid in unavailable_ids:
            processed.append({
                "id": f"unavailable-{uid}",
                "title": f"UNAVAILABLE: {uid}",
                "authors": "N/A",
                "publisher": "N/A",
                "year": 0,
                "source": "UNAVAILABLE",
                "source_id": uid,
                "url": "",
                "section": "Fetch Failed",
                "attack_vector": "N/A",
                "target": uid,
                "risk_level": "N/A",
                "skepticism_score": 0,
                "replication_prob": 0,
                "paragraph_type": "Error",
                "adversarial_tag": f"UNAVAILABLE: {uid}",
                "text": f"Source {uid} is inaccessible. No data returned.",
                "raw_text": f"UNAVAILABLE: {uid}",
                "severity": "N/A",
                "relevance_score": 0,
                "query_keywords": [],
                "mitigation_suggestion": "Try alternative source or verify the ID."
            })
        return processed

    # 1. Detect query domain & resolve ArXiv categories
    target_domain = detect_query_domain(clean_query)
    target_categories = get_domain_categories(target_domain)
    search_query = build_adversarial_query(clean_query)

    # 2. Fetch live candidates WITH category enforcement
    arxiv_hits = []
    openreview_hits = []
    pubpeer_hits = []
    
    if source is None or source.lower() in ["all", "arxiv"]:
        arxiv_hits = fetch_arxiv_realtime(search_query, max_results=10, categories=target_categories)
        
    if source is None or source.lower() in ["all", "openreview"]:
        openreview_hits = fetch_openreview_realtime(clean_query, max_results=5)

    if source is None or source.lower() in ["all", "pubpeer"]:
        pubpeer_hits = fetch_pubpeer_realtime(clean_query, max_results=5)

    if source is None or source.lower() == "all":
        candidates = []
        max_len = max(len(arxiv_hits), len(openreview_hits), len(pubpeer_hits))
        for i in range(max_len):
            if i < len(openreview_hits):
                candidates.append(openreview_hits[i])
            if i < len(arxiv_hits):
                candidates.append(arxiv_hits[i])
            if i < len(pubpeer_hits):
                candidates.append(pubpeer_hits[i])
    else:
        candidates = arxiv_hits + openreview_hits + pubpeer_hits

    if source and source.lower() != "all":
        candidates = [c for c in candidates if c.get("source", "").lower() == source.lower()]

    # 3. Post-fetch domain safety net: double-check every paper's domain
    domain_filtered = filter_by_domain(candidates, target_domain)
    if not domain_filtered and candidates:
        # Fallback: at minimum remove anything clearly from wrong domain
        domain_filtered = [c for c in candidates if get_paper_domain(c["title"], c["raw_text"]) != "PHYSICS/ASTROPHYSICS"]

    # 3b. Keyword overlap filter — require ≥2 query keyword matches
    keyword_filtered = keyword_overlap_filter(clean_query, domain_filtered, min_overlap=2)
    if not keyword_filtered:
        keyword_filtered = domain_filtered  # fallback

    # 4. Sentiment Filter: discard supportive pro-paper fluff / surveys praising capabilities
    filtered_candidates = [c for c in keyword_filtered if not is_supportive_marketing_fluff(c["title"], c["raw_text"])]

    # If all candidates were filtered out, re-query with explicit adversarial terms ONLY for ArXiv/All
    if not filtered_candidates and (source is None or source.lower() in ["all", "arxiv"]):
        adv_hits = fetch_arxiv_realtime(f"{clean_query} limitation critique flaw", max_results=10, categories=target_categories)
        adv_domain_filtered = filter_by_domain(adv_hits, target_domain)
        keyword_filtered = keyword_overlap_filter(clean_query, adv_domain_filtered, min_overlap=2)
        filtered_candidates = [c for c in keyword_filtered if not is_supportive_marketing_fluff(c["title"], c["raw_text"])]

    # 5. Re-rank using TF-IDF Cosine Similarity Re-Ranker
    top_ranked = rerank_results_cross_encoder(clean_query, filtered_candidates, top_k=6)
    
    # 5b. Strict relevance threshold limitation: drop anything below 0.20
    top_ranked = [r for r in top_ranked if r.get("relevance_score", 0) >= 0.20]
    
    # 4. Process top ranked critiques
    processed_critiques = []
    
    for item in top_ranked:
        raw_text = item["raw_text"]
        title = item["title"]
        
        # Double-check sentiment gate
        if is_supportive_marketing_fluff(title, raw_text):
            continue

        vector, severity, adversarial_tag, skep_score = detect_fine_grained_attack_vector(title, raw_text)
        
        # Check attack vector filter if provided
        if attack_vector and attack_vector.lower() != "all" and attack_vector.lower() not in vector.lower():
            continue
            
        raw_authors = item.get("authors", ["Academic Researchers"])
        if isinstance(raw_authors, list):
            authors_str = ", ".join(raw_authors[:3])
        else:
            authors_str = str(raw_authors)

        publisher_str = item.get("publisher") or f"{item.get('source', 'Academic')} Forum"

        critique_chunk = {
            "id": item["id"],
            "title": title,
            "authors": authors_str,
            "publisher": publisher_str,
            "year": item["year"],
            "source": item["source"],
            "source_id": item["source_id"],
            "url": item["url"],
            "section": item["section"],
            "attack_vector": vector,
            "target": f"{title[:40]}...",
            "risk_level": "Fatal" if severity == "Fatal" else ("Major" if severity == "Major" else "Moderate"),
            "skepticism_score": skep_score,
            "replication_prob": round(100.0 - skep_score, 1),
            "paragraph_type": "Limitation/Critique",
            "adversarial_tag": adversarial_tag,
            "text": raw_text[:380] + ("..." if len(raw_text) > 380 else ""),
            "query_keywords": [w.lower() for w in clean_query.split() if len(w) > 2],
            "severity": severity,
            "relevance_score": item.get("relevance_score", 0.8),
            "mitigation_suggestion": f"Subject methodology to strict benchmark de-contamination, non-public human test suites, and scaffold splits."
        }
        processed_critiques.append(critique_chunk)
        save_critique_chunk_db(critique_chunk)

    # Include custom ingested items from Supabase ONLY if they match query keywords
    if source is None or source.lower() == "all":
        all_stored = get_all_critiques_db()
        query_words = set(w.lower() for w in re.findall(r'\w+', clean_query) if len(w) > 3)
        for existing in all_stored:
            if existing.get("source") == "User Ingest" or "custom" in str(existing.get("id", "")):
                text_to_check = (existing.get("title", "") + " " + existing.get("text", "")).lower()
                if query_words and any(w in text_to_check for w in query_words):
                    if not any(p["id"] == existing["id"] for p in processed_critiques):
                        processed_critiques.append(existing)

    return processed_critiques

