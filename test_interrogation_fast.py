import sys
sys.path.insert(0, ".")

from backend.adversarial_rag import (
    compute_source_vote_tally,
    analyze_forced_contradiction,
    detect_definition_trap
)

print("Running fast unit tests for Interrogation Engine...")

mock_logic_critiques = [
    {"source_id": "arxiv-1", "title": "Data Contamination in Language Model Benchmarks", "text": "Models fail on clean splits due to leakage.", "attack_vector": "Benchmark Contamination", "publisher": "ArXiv"},
    {"source_id": "arxiv-2", "title": "Architectural Redundancy in Time-Series Transformers", "text": "Linear models beat complex transformers.", "attack_vector": "Architectural Redundancy", "publisher": "ArXiv"},
    {"source_id": "openreview-3", "title": "Shortcut Learning in Chain of Thought Prompting", "text": "CoT reasoning exhibits spurious pattern matching.", "attack_vector": "Shortcut Learning", "publisher": "OpenReview"},
    {"source_id": "pubpeer-4", "title": "Post-Pub Audit of LLM Logic Claims", "text": "Formal proofs fail under adversarial shifts.", "attack_vector": "Methodological Flaw", "publisher": "PubPeer"},
    {"source_id": "s2-5", "title": "Demonstrating Human-Like Deduction in Prompted Models", "text": "Achieves superior SOTA performance and human-like deduction.", "attack_vector": "Claim", "publisher": "Semantic Scholar"}
]

# 1. Binary Vote Tally
t1 = compute_source_vote_tally("Do LLMs perform true logical reasoning?", mock_logic_critiques)
assert t1["no_votes"] == 4 and t1["yes_votes"] == 1
assert "NO" in t1["majority_verdict"]
print("  [PASS] Test 1: Nuclear Binary Vote Tally")

# 2. Forced Contradiction (Definitional)
t2 = analyze_forced_contradiction("Is this a fundamental disagreement on definition?", mock_logic_critiques)
assert t2["conflict_type"] == "DEFINITIONAL_FRAMEWORK_CONFLICT"
print("  [PASS] Test 2: Forced Contradiction (Definitional)")

# 3. Definition Trap
t3 = detect_definition_trap("Provide a unified definition of true logical reasoning", mock_logic_critiques)
assert t3["has_definition_trap"] == True
assert t3["verdict_line"] == "NO CONSENSUS ON DEFINITION - ANSWER DEPENDS ON FRAMEWORK"
print("  [PASS] Test 3: Definition Trap Detector")

# 4. Multi-Choice Vote Tally
mock_forecasting_critiques = [
    {"source_id": "arxiv-101", "title": "Multiple Graph Neural Networks and Transformers for Vehicle Trajectory Prediction", "text": "We propose a spatio-temporal GNN and Transformer hybrid architecture for trajectory forecasting...", "attack_vector": "Methodological Limitation", "publisher": "IEEE"},
    {"source_id": "arxiv-102", "title": "ST-GNN: Spatial Temporal Graph Neural Network and Attention for Traffic Flow", "text": "A spatial-temporal graph neural network combined with temporal attention mechanisms...", "attack_vector": "Methodological Limitation", "publisher": "ACM"},
    {"source_id": "arxiv-103", "title": "Transformers are Graph Neural Networks", "text": "Self-attention is a graph neural network operating on a fully connected graph structure...", "attack_vector": "Methodological Limitation", "publisher": "ArXiv"},
    {"source_id": "arxiv-104", "title": "Expressive Power of Graph Neural Networks and Transformers", "text": "Theoretical bounds show GNNs with positional encodings achieve higher expressive power...", "attack_vector": "Methodological Limitation", "publisher": "NeurIPS"}
]

t4 = compute_source_vote_tally("Return a strict majority verdict: TRANSFORMERS, GNNS, HYBRID, or INCONCLUSIVE", mock_forecasting_critiques)
assert "HYBRID" in t4["majority_verdict"]
print("  [PASS] Test 4: Multi-Choice Recommendation Vote Tally (HYBRID Majority)")

# 5. False Dichotomy Detector
t5 = analyze_forced_contradiction("Is the choice between Transformers and GNNs a meaningful decision or false dichotomy?", mock_forecasting_critiques)
assert t5["conflict_type"] == "FALSE_DICHOTOMY_FRAMING"
print("  [PASS] Test 5: False Dichotomy Framing Detector")

# 6. Theoretical vs Empirical Gap Detector
t6 = analyze_forced_contradiction("Theoretical expressive power vs empirical gains comparison", mock_forecasting_critiques)
assert t6["conflict_type"] == "THEORETICAL_VS_EMPIRICAL_GAP"
print("  [PASS] Test 6: Theoretical vs Empirical Gap Detector")

print("\nALL FAST INTERROGATION TESTS PASSED CLEANLY!")
