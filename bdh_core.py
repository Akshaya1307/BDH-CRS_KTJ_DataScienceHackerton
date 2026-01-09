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

    def sparse_update(self, claim: str, signal: float):
        if claim not in self.nodes:
            self.nodes[claim] = BeliefNode(claim=claim)

        self.nodes[claim].update(signal)

        self.trajectory.append({
            "claim": claim,
            "signal": signal,
            "current_score": self.nodes[claim].score
        })

    def global_score(self) -> float:
        return sum(node.score for node in self.nodes.values())
