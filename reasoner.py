import json
import re
from bdh_core import BDHState


# =========================
# Chunk Analyzer
# =========================
def analyze_chunk(model, narrative_chunk: str, backstory: str):
    """
    Analyze a narrative chunk and return (claim, signal).
    Robust against malformed LLM output.
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

        # --- SAFE RAW TEXT EXTRACTION ---
        if hasattr(response, "text") and response.text:
            raw = response.text
        elif hasattr(response, "candidates") and response.candidates:
            raw = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("Empty LLM response")

        print("Raw LLM response:\n", raw)

        # --- EXTRACT JSON EVEN WITH EXTRA TEXT ---
        match = re.search(r"\{[\s\S]*?\}", raw)
        if not match:
            raise ValueError("No JSON block found")

        data = json.loads(match.group())

        score = float(data.get("score", 0))
        claim = str(data.get("claim", "unknown_claim"))

        return claim, score

    except Exception as e:
        # IMPORTANT: return weak negative signal (not zero)
        print("Parsing error:", e)
        return "unparseable_response", -0.1


# =========================
# BDH PIPELINE
# =========================
def run_bdh_pipeline(model, narrative: str, backstory: str):
    """
    Full BDH-style continuous reasoning pipeline.
    """

    state = BDHState()

    # Chunk narrative (simulate long-context reasoning)
    chunks = [c.strip() for c in narrative.split("\n\n") if c.strip()]
    chunks = chunks[:8]  # safety cap

    update_count = 0

    for chunk in chunks:
        claim, signal = analyze_chunk(model, chunk, backstory)

        # 🔑 LOWERED THRESHOLD — CRITICAL FIX
        if abs(signal) >= 0.1:
            state.sparse_update(claim, signal)
            update_count += 1

    print("Total belief updates:", update_count)

    final_score = state.global_score()

    # If no updates happened, explicitly mark neutral
