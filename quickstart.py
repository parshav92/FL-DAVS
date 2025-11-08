"""
Quick Start Script - Small test run for Phase 1
Tests the FL pipeline with minimal configuration
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override config for quick testing
os.environ['FL_NUM_ROUNDS'] = '5'
os.environ['FL_NUM_CLIENTS'] = '5'
os.environ['FL_LOCAL_EPOCHS'] = '1'

from train import main

if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUICK START - Testing FL Pipeline")
    print("Running with minimal configuration (5 clients, 5 rounds)")
    print("="*70 + "\n")
    
    main()
    
    print("\n" + "="*70)
    print("Quick test complete! Check results/ directory")
    print("To run full experiment, use: python train.py")
    print("="*70 + "\n")
