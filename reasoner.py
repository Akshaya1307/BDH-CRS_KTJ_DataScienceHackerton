import google.generativeai as genai
from bdh_core import BDHState

def analyze_chunk(model, narrative_chunk, backstory):
    prompt = f"""
You are evaluating narrative causality.

Backstory claim:
{backstory}

Narrative evidence:
{narrative_chunk}

Does this evidence SUPPORT (+1), WEAKLY SUPPORT (+0.5),
WEAKLY CONTRADICT (-0.5), or CONTRADICT (-1) the backstory?
Respond with:
score: <number>
claim: <short causal claim>
"""
    response = model.generate_content(prompt).text
    score = float(response.split("score:")[1].split()[0])
    claim = response.split("claim:")[1].strip()
    return claim, score


def run_bdh_pipeline(model, narrative, backstory):
    state = BDHState()
    chunks = narrative.split("\n\n")[:8]  # simulate long narrative

    for chunk in chunks:
        claim, signal = analyze_chunk(model, chunk, backstory)
        # selective update: ignore weak noise
        if abs(signal) >= 0.5:
            state.sparse_update(claim, signal)

    final_score = state.global_score()
    prediction = 1 if final_score >= 0 else 0

    return prediction, state
