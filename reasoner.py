import json
import re
from bdh_core import BDHState


# =====================================================
# Analyze a single narrative chunk
# =====================================================
def analyze_chunk(model, narrative_chunk: str, backstory: str):
    """
    Analyze a narrative chunk and return (claim, signal).
    Robust to malformed or partial LLM outputs.
    """

    prompt = f"""
You are evaluating narrative causality.

Backstory:
{backstory}

Narrative evidence:
{narrative_chunk}

Respond ONLY with valid JSON in this exact format:
{{
  "score": 1 | 0.5 | -0.5 | -1,
  "claim": "short causal claim"
}}
"""

    try:
        response = model.generate_content(prompt)

        # -------- SAFE RAW TEXT EXTRACTION --------
        if hasattr(response, "text") and response.text:
            raw = response.text
        elif hasattr(response, "candidates") and response.candidates:
            raw = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("Empty LLM response")

        print("Raw LLM response:\n", raw)

        # -------- JSON EXTRACTION (ROBUST) --------
        match = re.search(r"\{[\s\S]*?\}", raw)
        if not match:
            raise ValueError("No JSON block found")

        data = json.loads(match.group())

        score = float(data.get("score", 0.0))
        claim = str(data.get("claim", "unknown_claim"))

        return claim, score

    except Exception as e:
        # Weak negative signal ensures belief trace still exists
        print("Chunk parsing error:", e)
        return "unparseable_response", -0.1


# =====================================================
# BDH Continuous Reasoning Pipeline
# =====================================================
def run_bdh_pipeline(model, narrative: str, backstory: str):
    """
    Full BDH-style continuous reasoning pipeline.
    GUARANTEED to always return (prediction, state).
    """

    state = BDHState()
    prediction = 0  # safe default

    try:
        # -------- Narrative chunking --------
        chunks = [c.strip() for c in narrative.split("\n\n") if c.strip()]
        chunks = chunks[:8]  # safety cap for demo stability

        update_count = 0

        for chunk in chunks:
            claim, signal = analyze_chunk(model, chunk, backstory)

            # -------- Sparse BDH Update --------
            if abs(signal) >= 0.1:
                state.sparse_update(claim, signal)
                update_count += 1

        final_score = state.global_score()

        # -------- Final consistency decision --------
        if update_count > 0:
            prediction = 1 if final_score >= 0 else 0
        else:
            prediction = 0  # explicit neutral state

        print("BDH updates applied:", update_count)
        print("Final global belief score:", final_score)

    except Exception as e:
        # Pipeline-level failure should never crash UI
        print("BDH pipeline error:", e)

    # 🔒 ABSOLUTE GUARANTEE
    return prediction, state
