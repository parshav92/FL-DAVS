#!/usr/bin/env python3
"""Direct verification of SimpleCNN parameter count using PyTorch"""

import sys
sys.path.insert(0, '/home/parshav/Desktop/D/Projects/final_year_project')

from models.cnn_model import SimpleCNN
import torch

print("="*70)
print("PYTORCH MODEL PARAMETER COUNT")
print("="*70)

model = SimpleCNN(num_classes=9)

# Total count
total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")

# Layer breakdown
print("\nPer-layer breakdown:")
for name, param in model.named_parameters():
    print(f"  {name:25s}: {param.shape} = {param.numel():,} params")

print("\n" + "="*70)
print(f"ANSWER: The model has exactly {total:,} parameters")
print("="*70)
