import re
import nltk
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Download nltk data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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


def analyze_chunk_semantic(narrative_chunk: str, backstory: str) -> Tuple[str, float]:
    """
    Analyze chunk using semantic similarity (rule-based).
    Returns: (claim, score) where score ∈ {1, 0.5, 0, -0.5, -1}
    """
    # Clean inputs
    narrative_chunk = narrative_chunk.strip().lower()
    backstory = backstory.strip().lower()
    
    if not narrative_chunk:
        return "empty chunk", 0.0
    
    # Extract key concepts from backstory
    backstory_keywords = extract_keywords(backstory)
    narrative_keywords = extract_keywords(narrative_chunk)
    
    # Calculate semantic alignment
    alignment_score = calculate_alignment(backstory_keywords, narrative_keywords)
    
    # Generate a meaningful claim
    claim = generate_claim(narrative_chunk, alignment_score)
    
    # Map alignment to BDH score
    score = map_to_bdh_score(alignment_score)
    
    return claim, score


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text."""
    # Remove common stopwords
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'was', 'were', 'be', 'been', 
                 'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'that',
                 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
                 'he', 'him', 'his', 'she', 'her', 'i', 'me', 'my', 'we', 'us', 'our',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                 'could', 'can', 'may', 'might', 'must', 'shall', 'not', 'no', 'yes',
                 'so', 'then', 'than', 'just', 'only', 'also', 'very', 'too', 'much',
                 'many', 'some', 'any', 'all', 'both', 'each', 'every', 'such', 'like',
                 'about', 'above', 'below', 'under', 'over', 'through', 'between',
                 'during', 'before', 'after', 'since', 'until', 'while', 'because',
                 'although', 'though', 'even', 'if', 'unless', 'whether', 'while'}
    
    # Extract words
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    
    # Filter and return unique keywords
    keywords = [w for w in words if w not in stopwords]
    return list(set(keywords))


def calculate_alignment(backstory_keywords: List[str], narrative_keywords: List[str]) -> float:
    """Calculate semantic alignment between keywords (-1 to 1)."""
    if not backstory_keywords or not narrative_keywords:
        return 0.0
    
    # Calculate Jaccard similarity
    set_a = set(backstory_keywords)
    set_b = set(narrative_keywords)
    
    if not set_a or not set_b:
        return 0.0
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    jaccard = intersection / union if union > 0 else 0.0
    
    # Check for contradictions (simple rule-based)
    contradictions = check_contradictions(set_a, set_b)
    
    # Adjust score based on contradictions
    if contradictions > 0:
        return -jaccard * (1 + contradictions)
    else:
        return jaccard


def check_contradictions(backstory_words: set, narrative_words: set) -> float:
    """Check for obvious contradictions between word sets."""
    contradiction_pairs = [
        # Positive vs negative
        {'love', 'hate'}, {'like', 'dislike'}, {'good', 'bad'}, {'happy', 'sad'},
        {'success', 'failure'}, {'win', 'lose'}, {'help', 'harm'}, {'friend', 'enemy'},
        {'trust', 'betray'}, {'honest', 'dishonest'}, {'loyal', 'disloyal'},
        {'brave', 'cowardly'}, {'kind', 'cruel'}, {'generous', 'selfish'},
        {'truth', 'lie'}, {'real', 'fake'}, {'sincere', 'insincere'},
        # Presence vs absence
        {'present', 'absent'}, {'here', 'gone'}, {'alive', 'dead'}, {'found', 'lost'},
        {'exist', 'nonexistent'}, {'have', 'lack'}, {'full', 'empty'},
        # Ability vs inability
        {'can', 'cannot'}, {'able', 'unable'}, {'possible', 'impossible'},
        {'capable', 'incapable'}, {'competent', 'incompetent'},
        # Truth vs falsehood
        {'true', 'false'}, {'real', 'fake'}, {'truth', 'lie'}, {'fact', 'fiction'},
        {'accurate', 'inaccurate'}, {'correct', 'wrong'}, {'right', 'wrong'},
        # Time-related contradictions
        {'always', 'never'}, {'forever', 'temporary'}, {'permanent', 'temporary'},
        # Quantity contradictions
        {'all', 'none'}, {'every', 'no'}, {'complete', 'incomplete'},
        {'many', 'few'}, {'most', 'least'}, {'more', 'less'}
    ]
    
    contradictions = 0
    for pair in contradiction_pairs:
        has_backstory = any(word in backstory_words for word in pair)
        has_narrative = any(word in narrative_words for word in pair)
        
        if has_backstory and has_narrative:
            # Check if they're opposite words
            backstory_word = next((w for w in pair if w in backstory_words), None)
            narrative_word = next((w for w in pair if w in narrative_words), None)
            
            if backstory_word and narrative_word and backstory_word != narrative_word:
                contradictions += 1
    
    return min(contradictions / 5, 1.0)  # Normalize to 0-1


def generate_claim(chunk: str, alignment: float) -> str:
    """Generate a claim based on the chunk and alignment score."""
    chunk_words = chunk.split()
    if len(chunk_words) > 10:
        summary = ' '.join(chunk_words[:10]) + '...'
    else:
        summary = chunk
    
    if alignment >= 0.7:
        return f"Strongly supports: {summary}"
    elif alignment >= 0.3:
        return f"Supports: {summary}"
    elif alignment >= -0.3:
        return f"Neutral: {summary}"
    elif alignment >= -0.7:
        return f"Contradicts: {summary}"
    else:
        return f"Strongly contradicts: {summary}"


def map_to_bdh_score(alignment: float) -> float:
    """Map semantic alignment to BDH score."""
    if alignment >= 0.7:
        return 1.0
    elif alignment >= 0.3:
        return 0.5
    elif alignment >= -0.3:
        return 0.0
    elif alignment >= -0.7:
        return -0.5
    else:
        return -1.0


def run_bdh_pipeline(
    narrative: str, 
    backstory: str,
    min_chunk_length: int = 2,
    signal_threshold: float = 0.01,
    decision_threshold: float = 0.05
) -> Tuple[int, BDHState, dict]:
    """
    Run the complete BDH reasoning pipeline without external AI.
    
    Args:
        narrative: Text to analyze
        backstory: Context to evaluate against
        min_chunk_length: Minimum words per chunk to process
        signal_threshold: Minimum abs(signal) to update belief
        decision_threshold: Threshold for final prediction
    
    Returns:
        Tuple of (prediction, state, metadata)
        - prediction: 1 (consistent), 0 (contradict), or 0.5 (uncertain)
        - state: Complete BDHState object
        - metadata: Processing statistics
    """
    
    # Input validation
    if not narrative or not backstory:
        raise ValueError("Narrative and backstory must not be empty")
    
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
        # Analyze chunk using rule-based semantic analysis
        claim, signal = analyze_chunk_semantic(chunk, backstory)
        
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
    
    # Calculate final decision
    normalized_score = state.normalized_score()
    confidence_score = state.confidence_score()
    
    # Decision logic
    if state.chunk_count == 0:
        prediction = 0.5  # No chunks processed
    elif len(state.nodes) == 0:
        prediction = 0.5  # No belief nodes formed
    elif normalized_score >= decision_threshold:
        prediction = 1  # Consistent
    elif normalized_score <= -decision_threshold:
        prediction = 0  # Contradict
    else:
        prediction = 0.5  # Uncertain
    
    # Prepare metadata
    metadata = {
        "prediction": prediction,
        "normalized_score": normalized_score,
        "confidence_score": confidence_score,
        "global_score": state.global_score(),
        "total_chunks": state.chunk_count,
        "processed_chunks": state.processed_count,
        "belief_nodes": len(state.nodes),
        "belief_density": len(state.nodes) / max(state.chunk_count, 1),
        "decision_threshold": decision_threshold,
        "decision_reason": f"normalized_score={normalized_score:.3f} vs threshold={decision_threshold}"
    }
    
    return prediction, state, metadata
