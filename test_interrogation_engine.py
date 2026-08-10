"""
Test Suite: Advanced Interrogation Engine
Tests Vote Tally, Forced Contradiction, Definition Trap, and Weak Link Exclusion
"""
import sys
sys.path.insert(0, ".")

from backend.adversarial_rag import (
    compute_source_vote_tally,
    analyze_forced_contradiction,
    detect_definition_trap,
    generate_academic_risk_report
)

print("=" * 70)
print("TEST 1: Nuclear Binary Vote Tally Engine")
print("=" * 70)

mock_critiques = [
    {"source_id": "arxiv-2401.001", "title": "Data Contamination in Language Model Benchmarks", "text": "Widespread test set leakage causes inflated benchmark performance. Models fail on clean splits.", "attack_vector": "Benchmark Contamination", "publisher": "ArXiv"},
    {"source_id": "arxiv-2401.002", "title": "Architectural Redundancy in Time-Series Transformers", "text": "Linear models beat complex transformers when moving average components are isolated.", "attack_vector": "Architectural Redundancy", "publisher": "ArXiv"},
    {"source_id": "openreview-w6nlcS8Kkn", "title": "Shortcut Learning in Chain of Thought Prompting", "text": "CoT reasoning exhibits spurious pattern matching and brittle OOD degradation.", "attack_vector": "Shortcut Learning", "publisher": "OpenReview"},
    {"source_id": "pubpeer-10.1038", "title": "Post-Pub Audit of LLM Logic Claims", "text": "Analysis does not support claimed human-level reasoning. Formal proofs fail.", "attack_vector": "Methodological Flaw", "publisher": "PubPeer"},
    {"source_id": "s2-998877", "title": "Demonstrating Human-Like Deduction in Prompted Models", "text": "Prompted language models achieves superior SOTA performance and human-like deduction.", "attack_vector": "Claim", "publisher": "Semantic Scholar"}
]

tally = compute_source_vote_tally("Do LLMs perform true logical reasoning?", mock_critiques)

assert tally["no_votes"] == 4, f"Expected 4 NO votes, got {tally['no_votes']}"
assert tally["yes_votes"] == 1, f"Expected 1 YES vote, got {tally['yes_votes']}"
assert "NO" in tally["majority_verdict"], f"Expected NO majority verdict, got {tally['majority_verdict']}"
assert tally["consensus_percentage"] == "80%", f"Expected 80% consensus, got {tally['consensus_percentage']}"

print(f"  [PASS] Vote Tally Result: {tally['majority_verdict']} ({tally['consensus_percentage']})")
print(f"    YES: {tally['yes_votes']} | NO: {tally['no_votes']}")
for v in tally["vote_breakdown"]:
    print(f"    - [{v['publisher']}] {v['title']} -> VOTE: {v['vote']} ({v['rationale']})")

print()

print("=" * 70)
print("TEST 2: Forced Contradiction Analysis")
print("=" * 70)

contradiction = analyze_forced_contradiction(
    "Is this a fundamental disagreement on definition or empirical capability?",
    mock_critiques
)

assert contradiction is not None, "Contradiction analysis returned None"
assert contradiction["conflict_type"] in ["DEFINITIONAL_FRAMEWORK_CONFLICT", "EMPIRICAL_CAPABILITY_CONFLICT"]
print(f"  [PASS] Conflict Type: {contradiction['conflict_type']}")
print(f"    Explanation: {contradiction['conflict_explanation']}")
print(f"    Source A quote: {contradiction['source_a']['quote']}")
print(f"    Source B quote: {contradiction['source_b']['quote']}")

print()

print("=" * 70)
print("TEST 3: Definition Trap Detector")
print("=" * 70)

# Case A: Query asking for definition with mixed formal + empirical sources
dt_result = detect_definition_trap(
    "Provide a unified definition of true logical reasoning that all authors agree on.",
    mock_critiques
)

assert dt_result is not None, "Definition trap detector returned None"
assert dt_result["has_definition_trap"] == True, f"Expected definition trap, got {dt_result}"
assert dt_result["status_code"] == "NO_CONSENSUS_ON_DEFINITION"
assert dt_result["verdict_line"] == "NO CONSENSUS ON DEFINITION - ANSWER DEPENDS ON FRAMEWORK"

print(f"  [PASS] Verdict Line: {dt_result['verdict_line']}")
print(f"    Explanation: {dt_result['explanation']}")

print()

print("=" * 70)
print("TEST 4: Weak Link Exclusion Filtering")
print("=" * 70)

# Generate report while excluding paper arxiv-2401.002
report_full = generate_academic_risk_report("Are Large Language Models capable of true logical reasoning?")
report_excluded = generate_academic_risk_report("Are Large Language Models capable of true logical reasoning?", exclude_ids=["arxiv-2401.002"])

assert "vote_tally" in report_full, "vote_tally missing from report payload"
assert "forced_contradiction" in report_full, "forced_contradiction missing from report payload"
assert "definition_trap" in report_full, "definition_trap missing from report payload"

print(f"  [PASS] Report generated with Vote Tally: {report_full['vote_tally']['majority_verdict']}")
if report_full.get("definition_trap"):
    print(f"    Definition Trap Status: {report_full['definition_trap'].get('verdict_line')}")

print()

print("=" * 70)
print("TEST 5: Multi-Choice Recommendation Vote Tally")
print("=" * 70)

mock_forecasting_critiques = [
    {"source_id": "arxiv-101", "title": "Multiple Graph Neural Networks and Transformers for Vehicle Trajectory Prediction", "text": "We propose a spatio-temporal GNN and Transformer hybrid architecture for trajectory forecasting...", "attack_vector": "Methodological Limitation", "publisher": "IEEE"},
    {"source_id": "arxiv-102", "title": "ST-GNN: Spatial Temporal Graph Neural Network and Attention for Traffic Flow", "text": "A spatial-temporal graph neural network combined with temporal attention mechanisms...", "attack_vector": "Methodological Limitation", "publisher": "ACM"},
    {"source_id": "arxiv-103", "title": "Transformers are Graph Neural Networks", "text": "Self-attention is a graph neural network operating on a fully connected graph structure...", "attack_vector": "Methodological Limitation", "publisher": "ArXiv"},
    {"source_id": "arxiv-104", "title": "Expressive Power of Graph Neural Networks and Transformers", "text": "Theoretical bounds show GNNs with positional encodings achieve higher expressive power...", "attack_vector": "Methodological Limitation", "publisher": "NeurIPS"}
]

tally_mc = compute_source_vote_tally(
    "Return a strict majority verdict: TRANSFORMERS, GNNS, HYBRID, or INCONCLUSIVE",
    mock_forecasting_critiques
)

assert "HYBRID" in tally_mc["majority_verdict"], f"Expected HYBRID verdict, got {tally_mc['majority_verdict']}"
print(f"  [PASS] Multi-Choice Verdict: {tally_mc['majority_verdict']}")

print()

print("=" * 70)
print("TEST 6: False Dichotomy Framing Detector")
print("=" * 70)

fc_dichotomy = analyze_forced_contradiction(
    "Is the choice between Transformers and GNNs a meaningful decision or false dichotomy?",
    mock_forecasting_critiques
)

assert fc_dichotomy["conflict_type"] == "FALSE_DICHOTOMY_FRAMING", f"Expected FALSE_DICHOTOMY_FRAMING, got {fc_dichotomy['conflict_type']}"
print(f"  [PASS] Conflict Type: {fc_dichotomy['conflict_type']}")
print(f"    Explanation: {fc_dichotomy['conflict_explanation']}")

print()
print("=" * 70)
print("ALL INTERROGATION ENGINE TESTS COMPLETE")
print("=" * 70)
