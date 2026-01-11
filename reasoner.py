import json
import re
from bdh_core import BDHState


def analyze_chunk(model, narrative_chunk: str, backstory: str):
    """
    Analyze a narrative chunk and return (claim, signal).
    Prompt explicitly evaluates consistency vs contradiction.
    """

    prompt = f"""
You are evaluating whether the narrative evidence is CONSISTENT or CONTRADICTORY
with the given backstory.

Backstory:
{backstory}

Narrative evidence:
{narrative_chunk}

Scoring rules:
- Strong alignment with backstory → score = 1
- Mild alignment → score = 0.5
- Mild contradiction → score = -0.5
- Strong contradiction → score = -1

Respond ONLY with valid JSON in this exact format:
{{
  "score": 1 | 0.5 | -0.5 | -1,
  "claim": "short causal claim"
}}
"""

    try:
        response = model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            raw = response.text
        elif hasattr(response, "candidates") and response.candidates:
            raw = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("Empty LLM response")

        match = re.search(r"\{[\s\S]*?\}", raw)
        if not match:
            raise ValueError("No JSON found")

        data = json.loads(match.group())

        score = float(data.get("score", 0.0))
        claim = str(data.get("claim", "unknown_claim"))

        return claim, score

    except Exception as e:
        # Weak negative signal ensures traceability without domination
        print("Chunk parsing error:", e)
        return "unparseable_response", -0.1


def run_bdh_pipeline(model, narrative: str, backstory: str):
    """
    Full BDH-style continuous reasoning pipeline.
    GUARANTEED to always return (prediction, state)
    """

    state = BDHState()
    prediction = 0

    try:
        chunks = [c.strip() for c in narrative.split("\n\n") if c.strip()]
        chunks = chunks[:8]

        update_count = 0

        for chunk in chunks:
            claim, signal = analyze_chunk(model, chunk, backstory)

            if abs(s
