import json
import re
from bdh_core import BDHState


def analyze_chunk(model, narrative_chunk: str, backstory: str):
    """
    Analyze a narrative chunk and return (claim, signal).
    Robust against Gemini format drift.
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

        # ---- SAFE RAW TEXT EXTRACTION ----
        if hasattr(response, "text") and response.text:
            raw = response.text
        elif hasattr(response, "candidates") and response.candidates:
            raw = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("Empty model response")

        # ---- EXTRACT JSON BLOCK EVEN WITH EXTRA TEXT ----
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")

        data = json.loads(match.group(0))

        score = float(data["score"])
        claim = str(data["claim"]).strip()

        return claim, score

    except Exception:
        # IMPORTANT: parse failure is informative, not neutral
        return "model_output_parse_failure", -0.5


def run_bdh_pipeline(model, narrative: str, backstory: str):
    """
    Full BDH-style continuous reasoning pipeline.
    Guarantees belief evolution visibility.
    """

    state = BDHState()

    # Chunk narrative (simulate long-context reasoning)
    chunks = [c.strip() for c in narrative.split("\n\n") if c.strip()]
    chunks = chunks[:8]  # safety cap

    for chunk in chunks:
        claim, signal = analyze_chunk(model, chunk, backstory)

        # 🔑 ALWAYS update if claim exists
        if claim:
            state.sparse_update(claim, signal)

    # 🔒 Ensure at least one belief node exists
    if not getattr(state, "beliefs", None):
        state.sparse_update("default_consistency_assumption", 0.1)

    final_score = state.global_score()
    prediction = 1 if final_score >= 0 else 0

    return prediction, state
