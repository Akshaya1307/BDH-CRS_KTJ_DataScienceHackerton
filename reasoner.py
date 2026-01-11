# reasoner.py
import numpy as np
from typing import List, Tuple, Dict, Any
import time

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
        
        # Process chunks
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
        
        # Calculate consistency score
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
    
    def get_metrics(self, chunks_processed: int, total_chunks: int, processing_time: float) -> dict:
        """Calculate analysis metrics."""
        if total_chunks == 0:
            return {
                "processing_time": 0.0,
                "chunks_processed": "0/0 (0%)",
                "belief_density": "0.0%"
            }
        
        percentage = (chunks_processed / total_chunks) * 100
        
        # Calculate belief density
        belief_density = min(100, max(0, chunks_processed * 20))
        
        return {
            "processing_time": f"{processing_time:.3f}s",
            "chunks_processed": f"{chunks_processed}/{total_chunks} ({percentage:.1f}%)",
            "belief_density": f"{belief_density:.1f}%"
        }


# ADD THIS FUNCTION - The missing function that app.py is trying to import
def run_bdh_pipeline(text_chunks: List[str], min_chunk_length: int = 10, 
                     signal_threshold: float = 0.01, 
                     decision_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Main pipeline function for BDH analysis.
    
    Args:
        text_chunks: List of text chunks to analyze
        min_chunk_length: Minimum words per chunk
        signal_threshold: Signal detection threshold
        decision_threshold: Decision boundary threshold
        
    Returns:
        Dictionary with analysis results
    """
    start_time = time.time()
    
    # Initialize checker
    checker = BinaryConsistencyChecker(
        signal_threshold=signal_threshold,
        decision_threshold=decision_threshold
    )
    
    # Run analysis
    binary_result, score, explanation = checker.analyze_chunks(text_chunks)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Get metrics
    valid_chunks = sum(1 for chunk in text_chunks if len(chunk.strip().split()) >= min_chunk_length)
    metrics = checker.get_metrics(valid_chunks, len(text_chunks), processing_time)
    
    # Format result
    result = {
        "binary_result": binary_result,
        "result_label": "CONSISTENT" if binary_result == 1 else "CONTRADICTORY",
        "normalized_score": float(f"{score:.3f}"),
        "explanation": explanation,
        "processing_time": metrics["processing_time"],
        "chunks_processed": metrics["chunks_processed"],
        "belief_density": metrics["belief_density"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result


# For backward compatibility with your existing imports
if __name__ == "__main__":
    from datetime import datetime
    
    # Example usage
    example_chunks = [
        "The company reported increased profits this quarter.",
        "Revenue growth exceeded expectations by 15%.",
        "All departments showed improved performance metrics.",
        "The CEO announced expansion plans for next year.",
        "Employee satisfaction scores reached record highs."
    ]
    
    result = run_bdh_pipeline(example_chunks)
    print("BDH Pipeline Result:")
    print(json.dumps(result, indent=2))
