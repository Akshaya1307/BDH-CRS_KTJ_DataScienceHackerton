import json
import re
import nltk
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

# Download nltk data if needed (uncomment on first run)
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

@dataclass
class BeliefNode:
    """Represents a single belief claim with support/conflict tracking."""
    claim: str
    support: float = 0.0
    conflict: float = 0.0
    evidence_count: int = 0

    def update(self, signal: float):
        """Update belief with new evidence signal."""
        self.evidence_count += 1
        if signal > 0:
            self.support += signal
        elif signal < 0:
            self.conflict += abs(signal)
    
    def reset(self):
        """Reset the belief node to initial state."""
        self.support = 0.0
        self.conflict = 0.0
        self.evidence_count = 0

    @property
    def score(self):
        """Calculate net belief score."""
        return self.support - self.conflict
    
    @property
    def confidence(self):
        """Calculate confidence based on total evidence."""
        total = self.support + self.conflict
        return total / (total + 1) if total > 0 else 0.0


@dataclass
class BDHState:
    """Maintains belief state across narrative processing."""
    nodes: Dict[str, BeliefNode] = field(default_factory=dict)
    trajectory: List[dict] = field(default_factory=list)
    chunk_count: int = 0
    processed_count: int = 0
    
    def sparse_update(self, claim: str, signal: float):
        """Update belief state with new evidence."""
        if claim not in self.nodes:
            self.nodes[claim] = BeliefNode(claim)
        
        self.nodes[claim].update(signal)
        self.processed_count += 1
        
        # Record trajectory with global context
        self.trajectory.append({
            "step": len(self.trajectory) + 1,
            "claim": claim,
            "signal": signal,
            "current_score": self.global_score(),
            "normalized_score": self.normalized_score(),
            "node_count": len(self.nodes),
            "support": self.nodes[claim].support,
            "conflict": self.nodes[claim].conflict
        })
    
    def global_score(self):
        """Calculate total belief score across all nodes."""
        return sum(node.score for node in self.nodes.values())
    
    def normalized_score(self):
        """Calculate score normalized by evidence count."""
        total_evidence = sum(node.evidence_count for node in self.nodes.values())
        if total_evidence == 0:
            return 0.0
        return self.global_score() / total_evidence
    
    def confidence_score(self):
        """Calculate weighted confidence score."""
        if not self.nodes:
            return 0.0
        
        total_weight = 0
        weighted_sum = 0
        
        for node in self.nodes.values():
            weight = node.confidence
            weighted_sum += node.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def reset(self):
        """Reset entire state."""
        self.nodes.clear()
        self.trajectory.clear()
        self.chunk_count = 0
        self.processed_count = 0
    
    def summary(self):
        """Generate summary statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_evidence": sum(node.evidence_count for node in self.nodes.values()),
            "global_score": self.global_score(),
            "normalized_score": self.normalized_score(),
            "confidence_score": self.confidence_score(),
            "trajectory_length": len(self.trajectory)
        }


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using nltk or fallback regex."""
    try:
        # Try nltk for better sentence boundary detection
        return nltk.sent_tokenize(text)
    except:
        # Fallback regex for sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Clean up any empty strings
        return [s.strip() for s in sentences if s.strip()]


@lru_cache(maxsize=128)
def analyze_chunk_cached(model, narrative_chunk: str, backstory: str) -> Tuple[str, float]:
    """Cached version of analyze_chunk for repeated analyses."""
    return analyze_chunk(model, narrative_chunk, backstory)


def analyze_chunk(model, narrative_chunk: str, backstory: str) -> Tuple[str, float]:
    """
    Analyze a single narrative chunk against backstory.
    Returns: (claim, score) where score ∈ {1, 0.5, 0, -0.5, -1}
    """
    prompt = f"""
# BDH Consistency Analysis Task

## BACKSTORY (Hypothetical Context):
{backstory}

## NARRATIVE EVIDENCE (Observed Event/Fact):
{narrative_chunk}

## ANALYSIS INSTRUCTIONS:
Evaluate ONLY whether the narrative evidence is consistent with the backstory.
Focus on causal relationships, motivations, and implied states.

## SCORING GUIDELINES:
1.0  → CLEAR ALIGNMENT: Evidence directly confirms or strongly supports backstory
0.5  → PARTIAL ALIGNMENT: Evidence weakly supports or suggests backstory
0.0  → NEUTRAL: Evidence is unrelated, purely descriptive, or ambiguous
-0.5 → PARTIAL CONTRADICTION: Evidence weakly contradicts or casts doubt on backstory
-1.0 → CLEAR CONTRADICTION: Evidence directly contradicts or invalidates backstory

## OUTPUT REQUIREMENTS:
Respond with ONLY valid JSON in this exact format:
{{
  "score": 1 | 0.5 | 0 | -0.5 | -1,
  "claim": "brief summary of what this evidence implies about the backstory",
  "reasoning": "one sentence explaining the score (optional, for debugging)"
}}
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text if hasattr(response, "text") else str(response)
        
        # Extract JSON from response
        match = re.search(r'\{[\s\S]*\}', raw)
        if not match:
            raise ValueError("No JSON object found in response")
        
        data = json.loads(match.group())
        
        # Validate score
        valid_scores = {1.0, 0.5, 0.0, -0.5, -1.0}
        score = float(data.get("score", 0))
        if score not in valid_scores:
            score = round(score * 2) / 2  # Round to nearest valid score
            if score not in valid_scores:
                score = 0.0
        
        claim = str(data.get("claim", "unspecified claim")).strip()
        if not claim or claim == "unspecified claim":
            claim = narrative_chunk[:100] + "..." if len(narrative_chunk) > 100 else narrative_chunk
        
        return claim, score
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return "parsing error: invalid JSON format", 0.0
    except AttributeError as e:
        print(f"Model response error: {e}")
        return "model error: unexpected response format", 0.0
    except Exception as e:
        print(f"Unexpected error in analyze_chunk: {e}")
        return "analysis error: unable to process", 0.0


def run_bdh_pipeline(
    model, 
    narrative: str, 
    backstory: str,
    min_chunk_length: int = 3,
    signal_threshold: float = 0.1,
    decision_threshold: float = 0.05,
    use_caching: bool = True
) -> Tuple[int, BDHState, dict]:
    """
    Run the complete BDH reasoning pipeline.
    
    Args:
        model: Generative AI model
        narrative: Text to analyze
        backstory: Context to evaluate against
        min_chunk_length: Minimum words per chunk to process
        signal_threshold: Minimum abs(signal) to update belief
        decision_threshold: Threshold for final prediction
        use_caching: Whether to use cached analyses
    
    Returns:
        Tuple of (prediction, state, metadata)
        - prediction: 1 (consistent), 0 (contradict), or 0.5 (uncertain)
        - state: Complete BDHState object
        - metadata: Processing statistics
    """
    
    # Input validation
    if not narrative or not backstory:
        raise ValueError("Narrative and backstory must not be empty")
    
    if len(narrative.split()) < 5:
        raise ValueError("Narrative too short for meaningful analysis (min 5 words)")
    
    if len(backstory.split()) < 3:
        raise ValueError("Backstory too short (min 3 words)")
    
    # Initialize state
    state = BDHState()
    
    # Split narrative into chunks
    chunks = split_into_sentences(narrative)
    
    # Filter short chunks
    filtered_chunks = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) >= min_chunk_length:
            filtered_chunks.append(chunk)
        elif len(words) > 0:  # Very short chunks can be combined
            if filtered_chunks:
                filtered_chunks[-1] += " " + chunk
            else:
                filtered_chunks.append(chunk)
    
    state.chunk_count = len(filtered_chunks)
    
    # Process each chunk
    for i, chunk in enumerate(filtered_chunks):
        # Analyze chunk
        if use_caching:
            claim, signal = analyze_chunk_cached(model, chunk, backstory)
        else:
            claim, signal = analyze_chunk(model, chunk, backstory)
        
        # Only update state if signal is significant
        if abs(signal) >= signal_threshold:
            state.sparse_update(claim, signal)
        else:
            # Record neutral signals in trajectory for completeness
            state.trajectory.append({
                "step": len(state.trajectory) + 1,
                "claim": claim,
                "signal": signal,
                "current_score": state.global_score(),
                "normalized_score": state.normalized_score(),
                "node_count": len(state.nodes),
                "support": 0,
                "conflict": 0,
                "note": "below threshold"
            })
    
    # Calculate final decision with multiple strategies
    normalized_score = state.normalized_score()
    confidence_score = state.confidence_score()
    global_score = state.global_score()
    
    # Decision logic
    if abs(normalized_score) < decision_threshold:
        prediction = 0.5  # Uncertain/Neutral
    elif normalized_score >= decision_threshold:
        prediction = 1  # Consistent
    else:
        prediction = 0  # Contradict
    
    # Prepare metadata
    metadata = {
        "prediction": prediction,
        "normalized_score": normalized_score,
        "confidence_score": confidence_score,
        "global_score": global_score,
        "total_chunks": state.chunk_count,
        "processed_chunks": state.processed_count,
        "belief_nodes": len(state.nodes),
        "decision_threshold": decision_threshold,
        "decision_reason": f"normalized_score={normalized_score:.3f} vs threshold={decision_threshold}"
    }
    
    return prediction, state, metadata
