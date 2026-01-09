from dataclasses import dataclass, field
from typing import List, Dict
import math

@dataclass
class BeliefNode:
    claim: str
    support: float = 0.0
    conflict: float = 0.0

    def update(self, signal: float):
        if signal > 0:
            self.support += signal
        else:
            self.conflict += abs(signal)

    @property
    def score(self):
        return self.support - self.conflict


@dataclass
class BDHState:
    nodes: Dict[str, BeliefNode] = field(default_factory=dict)
    trajectory: List[Dict] = field(default_factory=list)

    def sparse_update(self, claim: str, signal: float):
        if claim not in self.nodes:
            self.nodes[claim] = BeliefNode(claim)
        self.nodes[claim].update(signal)

        # track state evolution (important for Track B)
        snapshot = {
            "claim": claim,
            "signal": signal,
            "current_score": self.nodes[claim].score
        }
        self.trajectory.append(snapshot)

    def global_score(self):
        return sum(node.score for node in self.nodes.values())
