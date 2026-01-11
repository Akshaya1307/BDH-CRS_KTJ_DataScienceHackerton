#!/usr/bin/env python3
"""
BDH Continuous Reasoner - Prediction Generation Script
For Track B competition submission (Rule-based version)
"""

import pandas as pd
import sys
import os
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent))

from reasoner import run_bdh_pipeline


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


def process_batch(
    test_df: pd.DataFrame,
    batch_size: int = 10,
    min_chunk_length: int = 2,
    signal_threshold: float = 0.01,
    decision_th
