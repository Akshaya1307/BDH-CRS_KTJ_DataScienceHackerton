import json
import re
from bdh_core import BDHState


def analyze_chunk(model, narrative_chunk: str, backstory: str):
    """
    Gemini-based causal scoring.
    NEVER returns None.
    NEVER returns untracked signals.
    """

    prompt = f"""
You are evaluating whether the narrative evidence is CONSISTENT or CONTRADICTORY
with the given backstory.

Backstory:
{backstory}

Narrative evidence:
{narrative_chunk}

Scoring rules:
- Strong alignment → score = 1
- Mild alignment → score = 0.5
- Neutral / descriptive → score = 0
- Mild contradiction → score = -0.5
- Strong contradiction → score = -1

Respond ONLY with valid JSON:
{{
  "score": 1 | 0.5 | 0 | -0.5 | -1,
  "claim": "short causal claim"
}}
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text if hasattr(response, "text") else ""

        match = re.search(r"\{[\s\S]*?\}", raw)
        if not match:
            raise ValueError("No JSON found")

        data = json.loads(match.group())
        score = float(data.get("score", 0))
        claim = str(data.get("claim", "unspecified claim"))

        return claim, score

    except Exception as e:
        # 🔒 SAFE fallback: neutral evidence (still updates!)
        print("Gemini parse error:", e)
        return "uncertain evidence", 0.0


def run_bdh_pipeline(model, narrative: str, backstory: str):
    """
    FINAL BDH PIPELINE
    - Sentence-level updates
    - No filtering of updates
    - Always returns (prediction, state)
    """

    state = BDHState()

    # 🔑 Sentence-wise chunking = visible graph
    chunks = [c.strip() for c in narrative.split(".") if c.strip()]

    for chunk in chunks:
        claim, signal = analyze_chunk(model, chunk, backstory)

        # 🔥 ALWAYS update state (even for 0)
        state.sparse_update(claim, signal)

    final_score = state.global_score()

    # 🔑 Correct decision logic
    prediction = 1 if final_score >= 0 else 0

    return prediction, state
