from dataclasses import dataclass, field
from typing import Dict, List
import json

@dataclass
class BeliefNode:
    """Core belief node for the BDH system."""
    claim: str
    support: float = 0.0
    conflict: float = 0.0
    evidence_count: int = 0

    def update(self, signal: float):
        """Update with new evidence signal."""
        self.evidence_count += 1
        if signal > 0:
            self.support += signal
        elif signal < 0:
            self.conflict += abs(signal)

    @property
    def score(self):
        """Net belief score."""
        return self.support - self.conflict
    
    @property
    def confidence(self):
        """Confidence level based on evidence volume."""
        total = self.support + self.conflict
        return total / (total + 1) if total > 0 else 0.0
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "claim": self.claim,
            "support": self.support,
            "conflict": self.conflict,
            "score": self.score,
            "evidence_count": self.evidence_count,
            "confidence": self.confidence
        }


@dataclass
class BDHState:
    """Main BDH state container."""
    nodes: Dict[str, BeliefNode] = field(default_factory=dict)
    trajectory: List[dict] = field(default_factory=list)
    
    def sparse_update(self, claim: str, signal: float):
        """Update belief state with sparse evidence."""
        if claim not in self.nodes:
            self.nodes[claim] = BeliefNode(claim)
        
        self.nodes[claim].update(signal)
        
        # Record in trajectory
        self.trajectory.append({
            "step": len(self.trajectory) + 1,
            "claim": claim,
            "signal": signal,
            "current_score": self.global_score(),
            "node_support": self.nodes[claim].support,
            "node_conflict": self.nodes[claim].conflict,
            "node_score": self.nodes[claim].score
        })
    
    def global_score(self):
        """Calculate global belief score."""
        return sum(node.score for node in self.nodes.values())
    
    def normalized_score(self):
        """Normalize score by evidence count."""
        total_evidence = sum(node.evidence_count for node in self.nodes.values())
        if total_evidence == 0:
            return 0.0
        return self.global_score() / total_evidence
    
    def to_json(self):
        """Serialize state to JSON."""
        return json.dumps({
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "trajectory": self.trajectory,
            "global_score": self.global_score(),
            "normalized_score": self.normalized_score()
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Load state from JSON."""
        data = json.loads(json_str)
        state = cls()
        
        # Recreate nodes
        for claim_str, node_data in data.get("nodes", {}).items():
            node = BeliefNode(
                claim=node_data["claim"],
                support=node_data["support"],
                conflict=node_data["conflict"],
                evidence_count=node_data.get("evidence_count", 0)
            )
            state.nodes[claim_str] = node
        
        # Restore trajectory
        state.trajectory = data.get("trajectory", [])
        
        return state
