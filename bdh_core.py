from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BeliefNode:
    claim: str
    support: float = 0.0
    conflict: float = 0.0

    def update(self, signal: float):
        if signal > 0:
            self.support += signal
        elif signal < 0:
            self.conflict += abs(signal)

    @property
    def score(self) -> float:
        return self.support - self.conflict


@dataclass
class BDHState:
    nodes: Dict[str, BeliefNode] = field(default_factory=dict)
    trajectory: List[dict] = field(default_factory=list)

    @property
    def beliefs(self):
        # Alias for compatibility with UI / reasoner
        return self.nodes

    def sparse_update(self, claim: str, signal: float):
        if claim not in self.nodes:
            self.nodes[claim] = BeliefNode(claim=claim)

        self.nodes[claim].update(signal)

        # Track updates for plotting in Streamlit
        self.trajectory.append({
            "step": len(self.trajectory) + 1,
            "claim": claim,
            "signal": signal,
            "current_score": self.nodes[claim].score,  # Match Streamlit app key
            "global_score": self.global_score()
        })

    def global_score(self) -> float:
        return sum(node.score for node in self.nodes.values())
