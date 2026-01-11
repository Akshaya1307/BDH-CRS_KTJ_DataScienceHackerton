import numpy as np
from typing import List, Tuple

class BinaryConsistencyChecker:
    def __init__(self, signal_threshold: float = 0.01, decision_threshold: float = 0.05):
        """
        Binary consistency checker that outputs:
        - 1 for consistent
        - 0 for contradictory
        
        Args:
            signal_threshold: Minimum threshold to detect meaningful signal
            decision_threshold: Threshold for making binary decision
        """
        self.signal_threshold = signal_threshold
        self.decision_threshold = decision_threshold
    
    def analyze_chunks(self, text_chunks: List[str]) -> Tuple[int, float, str]:
        """
        Analyze text chunks for consistency.
        
        Returns:
            Tuple of (binary_result, confidence_score, explanation)
        """
        if not text_chunks or len(text_chunks) == 0:
            return 0, 0.0, "No content to analyze"
        
        # Process chunks (this is where you'd add your actual analysis logic)
        chunk_count = len(text_chunks)
        processed_count = 0
        scores = []
        
        for chunk in text_chunks:
            if self._is_valid_chunk(chunk):
                score = self._analyze_single_chunk(chunk)
                scores.append(score)
                processed_count += 1
        
        # Calculate metrics
        if processed_count == 0:
            return 0, 0.0, "No valid chunks processed"
        
        # Calculate consistency score (replace with your actual logic)
        consistency_score = self._calculate_consistency(scores)
        
        # BINARY DECISION LOGIC
        if abs(consistency_score) < self.signal_threshold:
            # Signal too weak - default to contradictory (0)
            binary_result = 0
            explanation = "Evidence ambiguous or neutral"
        elif consistency_score >= self.decision_threshold:
            # Clearly consistent
            binary_result = 1
            explanation = "Evidence is consistent"
        elif consistency_score <= -self.decision_threshold:
            # Clearly contradictory
            binary_result = 0
            explanation = "Evidence is contradictory"
        else:
            # In the ambiguous zone
            binary_result = 0  # Default to contradictory when unsure
            explanation = "Weak signal, defaulting to contradictory"
        
        return binary_result, consistency_score, explanation
    
    def _is_valid_chunk(self, chunk: str, min_words: int = 10) -> bool:
        """Check if chunk meets minimum requirements."""
        words = chunk.strip().split()
        return len(words) >= min_words
    
    def _analyze_single_chunk(self, chunk: str) -> float:
        """
        Analyze a single chunk of text.
        Replace this with your actual analysis logic.
        Returns a score between -1 (contradictory) and 1 (consistent).
        """
        # Placeholder: Implement your actual analysis here
        # This could be semantic similarity, fact-checking, logic checking, etc.
        
        # For demonstration: random score
        return np.random.uniform(-1, 1)
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """Calculate overall consistency from individual scores."""
        if not scores:
            return 0.0
        
        # Simple average (replace with your aggregation logic)
        return np.mean(scores)
    
    def get_metrics(self, chunks_processed: int, total_chunks: int) -> dict:
        """Calculate analysis metrics."""
        if total_chunks == 0:
            return {
                "processing_time": 0.0,
                "chunks_processed": "0/0 (0%)",
                "belief_density": "0.0%"
            }
        
        percentage = (chunks_processed / total_chunks) * 100
        
        # Calculate belief density (example calculation)
        belief_density = min(100, max(0, chunks_processed * 10))
        
        return {
            "processing_time": 0.01,  # Replace with actual timing
            "chunks_processed": f"{chunks_processed}/{total_chunks} ({percentage:.1f}%)",
            "belief_density": f"{belief_density:.1f}%"
        }


# USAGE EXAMPLE
if __name__ == "__main__":
    # Example data
    example_chunks = [
        "The company reported increased profits this quarter.",
        "Revenue growth exceeded expectations by 15%.",
        "All departments showed improved performance metrics.",
        "The CEO announced expansion plans for next year.",
        "Employee satisfaction scores reached record highs."
    ]
    
    # Initialize checker with your thresholds
    checker = BinaryConsistencyChecker(
        signal_threshold=0.01,
        decision_threshold=0.05
    )
    
    # Run analysis
    binary_result, score, explanation = checker.analyze_chunks(example_chunks)
    
    # Get metrics
    metrics = checker.get_metrics(
        chunks_processed=len(example_chunks),
        total_chunks=len(example_chunks)
    )
    
    # Display results
    print("=" * 50)
    print("BINARY CONSISTENCY CHECKER RESULTS")
    print("=" * 50)
    print(f"Binary Result: {binary_result} ({'CONSISTENT' if binary_result == 1 else 'CONTRADICTORY'})")
    print(f"Normalized Score: {score:+.3f}")
    print(f"Explanation: {explanation}")
    print("-" * 50)
    print("ANALYSIS METRICS")
    print(f"Processing Time: {metrics['processing_time']}s")
    print(f"Chunks Processed: {metrics['chunks_processed']}")
    print(f"Belief-Density: {metrics['belief_density']}")
    print("=" * 50)
