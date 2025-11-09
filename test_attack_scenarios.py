"""
Test Different Attack Scenarios
Run experiments with varying attack percentages and types
"""

import os
import sys
import subprocess
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attacks.malicious_client import AttackType


def run_experiment(
    malicious_ratio: float,
    attack_type: AttackType,
    num_rounds: int = 20,
    num_clients: int = 10
):
    """
    Run a single experiment with specific attack configuration
    
    Args:
        malicious_ratio: Fraction of malicious clients (0.0-1.0)
        attack_type: Type of attack
        num_rounds: Number of FL rounds
        num_clients: Total number of clients
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {int(malicious_ratio*100)}% {attack_type.value.upper()} Attack")
    print(f"{'='*80}\n")
    
    # This would be integrated into the main script
    # For now, we'll create a configuration
    config = {
        'malicious_ratio': malicious_ratio,
        'attack_type': attack_type.value,
        'num_rounds': num_rounds,
        'num_clients': num_clients
    }
    
    print(f"Configuration: {config}")
    print("Run train_davs_pbft_blockchain.py with these parameters")
    
    return config


def main():
    """
    Run comprehensive attack scenario tests
    """
    print("="*80)
    print("ATTACK SCENARIO TESTING SUITE")
    print("="*80)
    
    # Test scenarios
    scenarios = [
        # Scenario 1: No attack (baseline)
        {
            'name': 'Baseline (No Attack)',
            'malicious_ratio': 0.0,
            'attack_type': AttackType.NONE,
            'description': 'All honest clients - establish baseline performance'
        },
        
        # Scenario 2: 10% malicious - Gradient Flip
        {
            'name': '10% Gradient Flip Attack',
            'malicious_ratio': 0.1,
            'attack_type': AttackType.FLIP,
            'description': '1 out of 10 clients performs gradient flipping'
        },
        
        # Scenario 3: 20% malicious - Gradient Flip
        {
            'name': '20% Gradient Flip Attack',
            'malicious_ratio': 0.2,
            'attack_type': AttackType.FLIP,
            'description': '2 out of 10 clients perform gradient flipping'
        },
        
        # Scenario 4: 30% malicious - Gradient Flip
        {
            'name': '30% Gradient Flip Attack',
            'malicious_ratio': 0.3,
            'attack_type': AttackType.FLIP,
            'description': '3 out of 10 clients perform gradient flipping'
        },
        
        # Scenario 5: 40% malicious - Gradient Flip
        {
            'name': '40% Gradient Flip Attack',
            'malicious_ratio': 0.4,
            'attack_type': AttackType.FLIP,
            'description': '4 out of 10 clients perform gradient flipping (High threat)'
        },
        
        # Scenario 6: 20% malicious - Gaussian Noise
        {
            'name': '20% Gaussian Noise Attack',
            'malicious_ratio': 0.2,
            'attack_type': AttackType.GAUSSIAN,
            'description': '2 out of 10 clients add heavy Gaussian noise'
        },
        
        # Scenario 7: 20% malicious - Targeted Attack
        {
            'name': '20% Targeted Attack',
            'malicious_ratio': 0.2,
            'attack_type': AttackType.TARGETED,
            'description': '2 out of 10 clients target specific layers'
        },
        
        # Scenario 8: 20% malicious - Byzantine
        {
            'name': '20% Byzantine Attack',
            'malicious_ratio': 0.2,
            'attack_type': AttackType.BYZANTINE,
            'description': '2 out of 10 clients exhibit random Byzantine behavior'
        },
    ]
    
    print(f"\nTotal scenarios to test: {len(scenarios)}\n")
    
    # Display scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Malicious ratio: {scenario['malicious_ratio']*100:.0f}%")
        print(f"   Attack type: {scenario['attack_type'].value}")
        print(f"   Description: {scenario['description']}")
        print()
    
    # Expected results
    print("="*80)
    print("EXPECTED RESULTS")
    print("="*80)
    print("""
Key Hypotheses to Validate:

1. DAVS Filtering Efficacy:
   - Malicious nodes should have significantly lower DAVS scores
   - Expected score separation: 0.5-0.7 (honest avg - malicious avg)
   - Malicious selection rate should be < baseline (random selection)

2. Committee Size vs Security:
   - k=5 committee should tolerate f=1 Byzantine node
   - Consensus should fail gracefully if f+1 malicious nodes selected
   - Attack success rate should be < 10% with DAVS+PBFT

3. Attack Type Comparison:
   - Gradient Flip: Most detectable (largest DAVS score drop)
   - Gaussian Noise: Moderately detectable
   - Targeted Attack: May be harder to detect
   - Byzantine: Random, unpredictable

4. Model Performance:
   - Baseline (0% attack): ~75-80% accuracy
   - 10-20% attack with DAVS+PBFT: ~70-75% accuracy
   - 30-40% attack with DAVS+PBFT: ~60-70% accuracy
   - Without DAVS/PBFT: Severe degradation (< 50%)

5. Communication Efficiency:
   - PBFT with k=5: ~25 messages (O(k²))
   - Full PBFT with N=10: ~100 messages (O(N²))
   - Bandwidth reduction: 75%
    """)
    
    print("="*80)
    print("TO RUN EXPERIMENTS")
    print("="*80)
    print("""
Modify train_davs_pbft_blockchain.py to accept command-line arguments:

    python train_davs_pbft_blockchain.py \\
        --malicious-ratio 0.2 \\
        --attack-type flip \\
        --num-rounds 20 \\
        --committee-size 5

Or manually edit the configuration in train_davs_pbft_blockchain.py
for each scenario and run separately.
    """)
    
    # Save scenarios to JSON
    results_dir = "results"
    Path(results_dir).mkdir(exist_ok=True)
    
    scenarios_file = os.path.join(results_dir, "attack_scenarios.json")
    with open(scenarios_file, 'w') as f:
        scenarios_json = []
        for scenario in scenarios:
            scenarios_json.append({
                'name': scenario['name'],
                'malicious_ratio': scenario['malicious_ratio'],
                'attack_type': scenario['attack_type'].value,
                'description': scenario['description']
            })
        json.dump(scenarios_json, f, indent=2)
    
    print(f"\n✓ Scenarios saved to {scenarios_file}")
    print(f"\nRun train_davs_pbft_blockchain.py to execute experiments!")


if __name__ == "__main__":
    main()
