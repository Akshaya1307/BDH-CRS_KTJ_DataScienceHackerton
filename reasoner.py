import json
import google.generativeai as genai
from bdh_core import BDHState


def analyze_chunk(model, narrative_chunk: str, backstory: str):
    """
    Analyze a single narrative chunk and return (claim, signal).
    This function is HARDENED against Gemini format drift.
    """

    prompt = f"""
You are evaluating narrative causality.

Backstory:
{backstory}

Narrative evidence:
{narrative_chunk}

Respond in JSON ONLY using this exact schema:
{{
  "score": 1 | 0.5 | -0.5 | -1,
  "claim": "short causal claim"
}}
"""

    try:
        response = model.generate_content(prompt)

        # ---- SAFE TEXT EXTRACTION ----
        text = None
        if hasattr(response, "text") and response.text:
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            text = response.candidates[0].content.parts[0].text

        if not text:
            raise ValueError("Empty response from Gemini")

        # ---- JSON PARSING ----
        data = json.loads(text)

        score = float(data.get("score", 0.0))
        claim = str(data.get("claim", "unknown_claim"))

        return claim, score

    except Exception:
        # NEVER crash Track B pipeline
        return "unparseable_response", 0.0


def run_bdh_pipeline(model, narrative: str, backstory: str):
    """
    Full BDH-inspired continuous reasoning pipeline.
    """

    state = BDHState()

    # Simulate long-context reasoning
    chunks = [c.strip() for c in narrative.split("\n\n") if c.strip()]
    chunks = chunks[:8]  # cap for stability

    for chunk in chunks:
        claim, signal = analyze_chunk(model, chunk, backstory)

        # Sparse update (BDH-inspired)
        if abs(signal) >= 0.5:
            state.sparse_update(claim, signal)

    final_score = state.global_score()
    prediction = 1 if final_score >= 0 else 0

    return prediction, state
