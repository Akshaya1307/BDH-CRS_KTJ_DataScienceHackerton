#!/usr/bin/env python3
"""
BDH Continuous Reasoner - Prediction Generation Script
For Track B competition submission
"""

import pandas as pd
import json
import sys
import os
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent))

from reasoner import run_bdh_pipeline
import google.generativeai as genai


def load_test_data(test_file: str) -> pd.DataFrame:
    """
    Load test data from CSV file.
    Expected columns: id, narrative, backstory
    """
    try:
        df = pd.read_csv(test_file)
        
        # Validate required columns
        required_cols = ['id', 'narrative', 'backstory']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print(f"✅ Loaded {len(df)} test cases from {test_file}")
        return df
    
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        sys.exit(1)


def initialize_model(api_key: str):
    """
    Initialize Gemini model.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Model initialized successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        sys.exit(1)


def process_batch(
    model,
    test_df: pd.DataFrame,
    batch_size: int = 10,
    min_chunk_length: int = 3,
    signal_threshold: float = 0.1,
    decision_threshold: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Process test cases in batches.
    """
    results = []
    total_cases = len(test_df)
    
    print(f"\n📊 Starting batch processing...")
    print(f"   Total cases: {total_cases}")
    print(f"   Batch size: {batch_size}")
    print(f"   Parameters: min_chunk={min_chunk_length}, "
          f"signal_thresh={signal_threshold}, "
          f"decision_thresh={decision_threshold}")
    
    for i, (_, row) in enumerate(test_df.iterrows()):
        try:
            # Process each test case
            prediction, state, metadata = run_bdh_pipeline(
                model=model,
                narrative=str(row['narrative']),
                backstory=str(row['backstory']),
                min_chunk_length=min_chunk_length,
                signal_threshold=signal_threshold,
                decision_threshold=decision_threshold,
                use_caching=True
            )
            
            # Convert prediction to binary (0 or 1)
            # Note: Our system returns 0.5 for uncertain, which we map to 0
            binary_prediction = 1 if prediction == 1 else 0
            
            results.append({
                'id': row['id'],
                'prediction': binary_prediction,
                'normalized_score': metadata['normalized_score'],
                'confidence': metadata['confidence_score'],
                'total_chunks': metadata['total_chunks'],
                'belief_nodes': metadata['belief_nodes'],
                'processing_time': metadata['processing_time']
            })
            
            # Progress update
            if (i + 1) % batch_size == 0 or (i + 1) == total_cases:
                print(f"   Processed {i + 1}/{total_cases} cases "
                      f"({(i + 1)/total_cases*100:.1f}%)")
        
        except Exception as e:
            print(f"❌ Error processing case {row['id']}: {e}")
            # Default to 0 (contradict) on error
            results.append({
                'id': row['id'],
                'prediction': 0,
                'normalized_score': 0.0,
                'confidence': 0.0,
                'total_chunks': 0,
                'belief_nodes': 0,
                'processing_time': 0.0,
                'error': str(e)
            })
    
    return results


def save_predictions(results: List[Dict[str, Any]], output_file: str):
    """
    Save predictions to CSV file.
    """
    try:
        # Create DataFrame with required columns
        df = pd.DataFrame(results)
        
        # Save required columns
        submission_df = df[['id', 'prediction']].copy()
        submission_df.to_csv(output_file, index=False)
        
        # Save detailed results for analysis
        detailed_file = output_file.replace('.csv', '_detailed.csv')
        df.to_csv(detailed_file, index=False)
        
        print(f"\n✅ Predictions saved to {output_file}")
        print(f"✅ Detailed results saved to {detailed_file}")
        
        # Print summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"   Total predictions: {len(df)}")
        print(f"   Consistent (1): {len(df[df['prediction'] == 1])} "
              f"({len(df[df['prediction'] == 1])/len(df)*100:.1f}%)")
        print(f"   Contradict (0): {len(df[df['prediction'] == 0])} "
              f"({len(df[df['prediction'] == 0])/len(df)*100:.1f}%)")
        
        if 'normalized_score' in df.columns:
            print(f"   Avg normalized score: {df['normalized_score'].mean():.3f}")
            print(f"   Avg confidence: {df['confidence'].mean():.3f}")
        
        return submission_df
    
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
        sys.exit(1)


def main():
    """
    Main execution function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='BDH Continuous Reasoner - Generate predictions for test data'
    )
    
    parser.add_argument(
        '--test_file',
        type=str,
        required=True,
        help='Path to test data CSV file'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default='submission.csv',
        help='Path to output predictions CSV file'
    )
    
    parser.add_argument(
        '--api_key',
        type=str,
        help='Gemini API key (or set GEMINI_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Number of cases to process between progress updates'
    )
    
    parser.add_argument(
        '--min_chunk_length',
        type=int,
        default=3,
        help='Minimum words per chunk'
    )
    
    parser.add_argument(
        '--signal_threshold',
        type=float,
        default=0.1,
        help='Minimum absolute signal value to update beliefs'
    )
    
    parser.add_argument(
        '--decision_threshold',
        type=float,
        default=0.05,
        help='Threshold for final decision'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("BDH CONTINUOUS REASONER - PREDICTION GENERATION")
    print("=" * 60)
    
    # Get API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: Gemini API key required")
        print("   Provide via --api_key argument or GEMINI_API_KEY environment variable")
        sys.exit(1)
    
    # Load test data
    test_df = load_test_data(args.test_file)
    
    # Initialize model
    model = initialize_model(api_key)
    
    # Process test cases
    results = process_batch(
        model=model,
        test_df=test_df,
        batch_size=args.batch_size,
        min_chunk_length=args.min_chunk_length,
        signal_threshold=args.signal_threshold,
        decision_threshold=args.decision_threshold
    )
    
    # Save predictions
    save_predictions(results, args.output_file)
    
    print("\n" + "=" * 60)
    print("🎉 PREDICTION GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
