"""
Adversarial Self-Audit Diagnostic Script
Runs three precision audits against the RAG pipeline.
"""
import sys, json
sys.path.insert(0, ".")

from backend.database import search_critiques_db, CRITIQUE_DATABASE
from backend.adversarial_rag import generate_academic_risk_report, run_adversarial_search

# ============================================================
# AUDIT 1: The Self-Audit
# Check if ANY cited sources contain 'zero-shot', 'fine-tuning overlap', or 'test condition leakage'
# ============================================================
print("=" * 80)
print("AUDIT 1: SELF-AUDIT — Exact Trigram Search in Cited Sources")
print("=" * 80)

query = "Does the claimed generalization to novel conditions actually involve zero-shot inference, or does the fine-tuning data overlap with the test conditions?"
search_results = run_adversarial_search(query)
critiques = search_results.get("results", [])

target_phrases = ["zero-shot", "fine-tuning overlap", "test condition leakage"]
audit_results = []

for c in critiques:
    combined = (c.get("title", "") + " " + c.get("raw_text", c.get("text", ""))).lower()
    matches = {phrase: phrase in combined for phrase in target_phrases}
    any_match = any(matches.values())
    audit_results.append({
        "id": c.get("id", "unknown"),
        "title": c.get("title", "unknown")[:80],
        "source": c.get("source", "unknown"),
        "relevance_score": c.get("relevance_score", 0),
        "matches": matches,
        "has_any_match": any_match
    })

total_with_match = sum(1 for r in audit_results if r["has_any_match"])

for r in audit_results:
    match_str = ", ".join(f"'{k}': {'YES' if v else 'NO'}" for k, v in r["matches"].items())
    print(f"  [{r['source']}] {r['title']}")
    print(f"    Relevance: {r['relevance_score']} | Trigram Matches: {match_str}")
    print(f"    Verdict: {'MATCH' if r['has_any_match'] else 'NO MATCH'}")
    print()

if total_with_match == 0:
    print(">>> AUDIT 1 RESULT: No evidence found in the corpus to answer this query.")
else:
    print(f">>> AUDIT 1 RESULT: {total_with_match} of {len(audit_results)} sources contain target phrases.")

# ============================================================
# AUDIT 2: The Source Disqualifier
# Exclude all sources tagged solely as 'Methodological Flaw' without specific data leakage terms
# ============================================================
print()
print("=" * 80)
print("AUDIT 2: SOURCE DISQUALIFIER — Filter for Data Leakage / Training Split Terms")
print("=" * 80)

disqualifier_phrases = [
    "training data overlaps with test",
    "seen during fine-tuning",
    "data leakage",
    "training splits",
    "few-shot evaluation",
    "training data overlap"
]

passed_count = 0
for c in critiques:
    combined = (c.get("title", "") + " " + c.get("raw_text", c.get("text", ""))).lower()
    attack_vec = c.get("attack_vector", "").lower()
    
    # Check if source is tagged SOLELY as generic 'Methodological Flaw'
    is_generic_tag = "methodological flaw" in attack_vec and "leakage" not in attack_vec and "contamination" not in attack_vec
    
    # Check for specific disqualifier phrases
    has_specific = any(dp in combined for dp in disqualifier_phrases)
    
    status = "EXCLUDED (generic tag, no specific terms)" if (is_generic_tag and not has_specific) else ("PASSED" if has_specific else "EXCLUDED (no specific terms)")
    if has_specific:
        passed_count += 1
    print(f"  [{c.get('source')}] {c.get('title', '')[:70]}")
    print(f"    Attack Vector: {c.get('attack_vector', 'N/A')}")
    print(f"    Has specific data leakage terms: {has_specific}")
    print(f"    Status: {status}")
    print()

print(f">>> AUDIT 2 RESULT: {passed_count} document(s) passed the disqualifier filter.")

# ============================================================
# AUDIT 3: The Forced Comparison — ArXiv #2308.10783 Abstract Analysis
# ============================================================
print()
print("=" * 80)
print("AUDIT 3: FORCED COMPARISON — ArXiv #2308.10783 (Bangla Sentiment)")
print("=" * 80)

# The actual abstract fetched from ArXiv API
abstract = """The rapid expansion of the digital world has propelled sentiment analysis into a critical tool across diverse sectors such as marketing, politics, customer service, and healthcare. While there have been significant advancements in sentiment analysis for widely spoken languages, low-resource languages, such as Bangla, remain largely under-researched due to resource constraints. Furthermore, the recent unprecedented performance of Large Language Models (LLMs) in various applications highlights the need to evaluate them in the context of low-resource languages. In this study, we present a sizeable manually annotated dataset encompassing 33,606 Bangla news tweets and Facebook comments. We also investigate zero- and few-shot in-context learning with several language models, including Flan-T5, GPT-4, and Bloomz, offering a comparative analysis against fine-tuned models. Our findings suggest that monolingual transformer-based models consistently outperform other models, even in zero and few-shot scenarios. To foster continued exploration, we intend to make this dataset and our research tools publicly available to the broader research community."""

abstract_lower = abstract.lower()

check_terms = {
    "test set was unseen during fine-tuning": "test set was unseen during fine-tuning" in abstract_lower,
    "unseen": "unseen" in abstract_lower,
    "held-out": "held-out" in abstract_lower,
    "held out": "held out" in abstract_lower,
    "test split": "test split" in abstract_lower,
    "training data overlap": "training data overlap" in abstract_lower,
    "data leakage": "data leakage" in abstract_lower,
    "fine-tuning overlap": "fine-tuning overlap" in abstract_lower,
    "test condition": "test condition" in abstract_lower,
    "zero-shot": "zero-shot" in abstract_lower or "zero- and few-shot" in abstract_lower,
    "fine-tuned": "fine-tuned" in abstract_lower or "fine-tuning" in abstract_lower,
}

print(f"\nFull Abstract Text:")
print(f"  \"{abstract[:300]}...\"")
print()
print("Term Presence Audit:")
for term, found in check_terms.items():
    print(f"  '{term}': {'FOUND' if found else 'NOT FOUND'}")

print()
print(">>> AUDIT 3 ANSWER (strictly from abstract text):")
print("  The abstract does NOT explicitly state that the test set was unseen during fine-tuning.")
print("  The abstract does NOT mention 'held-out split', 'test split', 'unseen', or 'data leakage'.")
print("  The abstract DOES mention 'zero- and few-shot in-context learning' and 'fine-tuned models',")
print("  but only in the context of comparing prompting strategies vs fine-tuning accuracy.")
print("  It presents a 33,606-sample dataset and compares LLM zero/few-shot performance against")
print("  fine-tuned transformer models. The abstract is SILENT on whether the fine-tuning and")
print("  evaluation used the same dataset split or independent test conditions.")
print("  CONCLUSION: Cannot determine from the abstract alone whether test/train overlap exists.")

print()
print("=" * 80)
print("FULL AUDIT COMPLETE")
print("=" * 80)
