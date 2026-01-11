#!/usr/bin/env python3
"""
Debug script to test BDH pipeline
"""

import google.generativeai as genai
import os
from reasoner import run_bdh_pipeline, analyze_chunk, split_into_sentences

# Initialize
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("❌ Set GEMINI_API_KEY environment variable")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Test data
narrative = "John helped his friend. This shows loyalty. He prioritized friendship."
backstory = "John is a loyal friend."

print("=" * 60)
print("DEBUG TEST - BDH PIPELINE")
print("=" * 60)

print(f"\n📝 Narrative: {narrative}")
print(f"🎭 Backstory: {backstory}")

# Step 1: Test sentence splitting
print(f"\n🔍 Step 1: Sentence Splitting")
chunks = split_into_sentences(narrative)
print(f"   Found {len(chunks)} chunks:")
for i, chunk in enumerate(chunks, 1):
    words = len(chunk.split())
    print(f"   {i}. '{chunk}' ({words} words)")

# Step 2: Test analyze_chunk directly
print(f"\n🔍 Step 2: Testing analyze_chunk()")
test_chunk = "John helped his friend."
print(f"   Testing chunk: '{test_chunk}'")
print(f"   Against backstory: '{backstory}'")

claim, score = analyze_chunk(model, test_chunk, backstory)
print(f"   Result: score={score}, claim='{claim}'")

# Step 3: Test the full pipeline
print(f"\n🔍 Step 3: Testing full pipeline")
print(f"   Parameters: min_chunk_length=1, signal_threshold=0.01")

prediction, state, metadata = run_bdh_pipeline(
    model=model,
    narrative=narrative,
    backstory=backstory,
    min_chunk_length=1,
    signal_threshold=0.01,
    decision_threshold=0.05,
    use_caching=False  # Disable cache for debugging
)

print(f"\n📊 Results:")
print(f"   Prediction: {prediction}")
print(f"   Total chunks: {metadata['total_chunks']}")
print(f"   Processed chunks: {metadata['processed_chunks']}")
print(f"   Belief nodes: {metadata['belief_nodes']}")
print(f"   Normalized score: {metadata['normalized_score']}")

if state.trajectory:
    print(f"\n📈 Trajectory:")
    for t in state.trajectory:
        print(f"   Step {t['step']}: signal={t['signal']}, claim='{t['claim'][:50]}...'")
else:
    print(f"\n📈 No trajectory entries")

if state.nodes:
    print(f"\n🧩 Belief Nodes:")
    for claim, node in state.nodes.items():
        print(f"   '{claim[:50]}...': score={node.score}, support={node.support}, conflict={node.conflict}")
else:
    print(f"\n🧩 No belief nodes")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
