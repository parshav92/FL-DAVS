"""
Regenerate missing plots and summary from existing attack_log.json
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.attack_logger import AttackLogger

# Load the experiment data
result_dir = "results/hybrid_davs_pbft_blockchain_pathmnist_non_iid_attack20"
log_file = os.path.join(result_dir, "attack_log.json")

print("Loading experiment data...")
with open(log_file, 'r') as f:
    data = json.load(f)

# Create logger instance
logger = AttackLogger(
    experiment_name=data['experiment'],
    save_dir="results",
    malicious_nodes=data['config']['malicious_nodes'],
    total_nodes=data['config']['total_nodes']
)

# Restore the rounds data
logger.rounds_data = data['rounds']

print(f"Loaded {len(logger.rounds_data)} rounds of data")
print("\nRegenerating missing outputs...")

# Generate all plots and reports
logger.plot_davs_score_distribution()
logger.plot_accuracy_trends()
summary = logger.generate_summary_report()

print("\n✅ All outputs regenerated successfully!")
print(f"\nResults saved to: {logger.exp_dir}")
