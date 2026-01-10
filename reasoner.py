import json
import re
from bdh_core import BDHState

def analyze_chunk(model, narrative_chunk: str, backstory: str):
    """
    Analyze a narrative chunk and return (claim, signal).
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
            raise ValueError("Empty Gemini response")

        # ---- DEBUG LOGGING ----
        # Print raw response for debugging (can be logged to a file if needed)
        print(f"Raw LLM response:\n{raw}")

        # ---- EXTRACT JSON BLOCK EVEN WITH EXTRA TEXT ----
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response")

        json_text = match.group(0)
        data = json.loads(json_text)

        score = float(data["score"])
        claim = str(data["claim"])

        return claim, score

    except Exception as e:
        # Log exception details for debugging
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response was: {raw if 'raw' in locals() else 'N/A'}")

        # Use a neutral fallback claim and zero signal to avoid polluting belief nodes negatively
        return "unparseable_response", 0.0


def run_bdh_pipeline(model, narrative: str, backstory: str):
    """
    Full BDH-style continuous reasoning pipeline.
    """

    state = BDHState()

    # Chunk narrative (simulate long-context reasoning)
    chunks = [c.strip() for c in narrative.split("\n\n") if c.strip()]
    chunks = chunks[:8]  # safety cap

    for chunk in chunks:
        claim, signal = analyze_chunk(model, chunk, backstory)

        # 🔑 SPARSE UPDATE (LOWERED FOR VISIBILITY)
        if abs(signal) >= 0.3:
            state.sparse_update(claim, signal)

    final_score = state.global_score()
    prediction = 1 if final_score >= 0 else 0

    return prediction, state
