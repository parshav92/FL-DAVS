"""
DAVS (Data-Aware Verifier Selection) - Phase 3
Correct implementation following DAVS specification:
- Amnesic (no historical tracking)
- Sketch-based (works on 128-dim gradient sketches)
- Single metric: cosine similarity for representativeness
- No separate Byzantine detection (implicit via low similarity scores)
"""

import torch
import numpy as np
from typing import Dict, Tuple, List


class DAVSSelector:
    """
    Data-Aware Verifier Selection (DAVS)
    
    CRITICAL: DAVS is AMNESIC - no historical tracking whatsoever.
    Selection based ONLY on real-time gradient sketch similarity.
    
    Reference: "DAVS operates on the principle that a node's suitability 
    for consensus in the current round should be judged solely on the 
    mathematical properties of its contribution to that round."
    """
    
    def __init__(self, committee_size: int = 5):
        """
        Initialize DAVS selector
        
        Args:
            committee_size: Size of verification committee
        
        Note: NO historical tracking, NO reputation system
        """
        self.committee_size = committee_size
    
    @staticmethod
    def compute_cosine_similarity(sketch1: torch.Tensor, sketch2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two gradient sketches
        
        Args:
            sketch1: First gradient sketch (128-dim)
            sketch2: Second gradient sketch (128-dim)
        
        Returns:
            Cosine similarity in [-1, 1]
        """
        # Ensure same device
        if sketch1.device != sketch2.device:
            sketch2 = sketch2.to(sketch1.device)
        
        # Compute cosine similarity
        dot_product = torch.dot(sketch1, sketch2)
        norm1 = torch.norm(sketch1)
        norm2 = torch.norm(sketch2)
        
        similarity = dot_product / (norm1 * norm2 + 1e-10)
        return similarity.item()
    
    def compute_representativeness_score(self, 
                                         client_sketch: torch.Tensor,
                                         all_sketches: List[torch.Tensor]) -> float:
        """
        Compute Representativeness Score as defined in DAVS specification.
        
        "The Representativeness Score is defined as the mean cosine similarity 
        between a client's sketch and the sketches of all other participating clients."
        
        Args:
            client_sketch: Gradient sketch of target client (128-dim)
            all_sketches: List of all gradient sketches (128-dim each)
        
        Returns:
            Representativeness score (mean cosine similarity)
        """
        similarities = []
        
        for other_sketch in all_sketches:
            # Skip self-comparison
            if not torch.equal(client_sketch, other_sketch):
                sim = self.compute_cosine_similarity(client_sketch, other_sketch)
                similarities.append(sim)
        
        if len(similarities) == 0:
            return 0.0
        
        # Mean similarity = representativeness
        return float(np.mean(similarities))
    
    def select_committee(self, 
                        client_sketches: Dict[int, torch.Tensor]) -> Tuple[List[int], Dict[int, float]]:
        """
        Select verification committee using DAVS.
        
        CRITICAL: This is the ONLY method that matters in DAVS.
        - Input: gradient sketches (128-dim) ONLY
        - Output: top-k clients by representativeness score
        - NO historical data, NO heuristics, NO multi-factor scoring
        
        Args:
            client_sketches: Dictionary mapping client_id to gradient sketch (128-dim)
        
        Returns:
            Tuple of (committee_ids, representativeness_scores)
        """
        scores = {}
        all_sketches = list(client_sketches.values())
        
        # Compute representativeness for each client
        for client_id, sketch in client_sketches.items():
            scores[client_id] = self.compute_representativeness_score(sketch, all_sketches)
        
        # Select top-k by representativeness (highest similarity = most representative)
        sorted_clients = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        committee = [client_id for client_id, _ in sorted_clients[:self.committee_size]]
        
        return committee, scores


if __name__ == "__main__":
    # Test DAVS committee selection with gradient sketches
    print("="*70)
    print("Testing DAVS Committee Selection (Correct Implementation)")
    print("="*70)
    
    # Simulate 10 clients with gradient sketches
    num_clients = 10
    sketch_dim = 128
    
    print("\n" + "="*70)
    print("Initializing DAVS Selector")
    print("="*70)
    selector = DAVSSelector(committee_size=5)
    print(f"✓ Committee size: {selector.committee_size}")
    print(f"✓ Algorithm: Amnesic (no historical tracking)")
    print(f"✓ Metric: Cosine similarity only")
    
    # Create gradient sketches for clients
    print("\n" + "="*70)
    print("Simulating Gradient Sketches")
    print("="*70)
    
    client_sketches = {}
    
    # Honest clients (similar sketches)
    honest_base = torch.randn(sketch_dim)
    for i in range(7):
        # Add small perturbations to base
        honest_sketch = honest_base + torch.randn(sketch_dim) * 0.2
        client_sketches[i] = honest_sketch
    
    # Malicious clients (very different sketches)
    for i in range(7, 10):
        malicious_sketch = torch.randn(sketch_dim) * 5.0  # Large, different
        client_sketches[i] = malicious_sketch
    
    print(f"✓ Created {num_clients} gradient sketches ({sketch_dim}-dim)")
    print(f"  Honest clients: 0-6 (similar gradients)")
    print(f"  Malicious clients: 7-9 (different gradients)")
    
    # Select committee
    print("\n" + "="*70)
    print("Computing Representativeness Scores")
    print("="*70)
    
    committee, scores = selector.select_committee(client_sketches)
    
    print("\nRepresentativeness Scores (mean cosine similarity):")
    for client_id in sorted(scores.keys()):
        is_malicious = "⚠️  MALICIOUS" if client_id >= 7 else "✓ Honest"
        in_committee = "✓ SELECTED" if client_id in committee else ""
        print(f"  Client {client_id}: {scores[client_id]:6.4f}  {is_malicious}  {in_committee}")
    
    # Results
    print("\n" + "="*70)
    print("Committee Selection Results")
    print("="*70)
    
    print(f"✓ Selected committee: {committee}")
    
    malicious_in_committee = [c for c in committee if c >= 7]
    print(f"\nByzantine Resilience:")
    print(f"  Malicious clients in committee: {len(malicious_in_committee)}/{len(committee)}")
    
    if len(malicious_in_committee) == 0:
        print("  ✓✓✓ SUCCESS: All malicious clients filtered out!")
        print("  (Low representativeness scores excluded them automatically)")
    else:
        print(f"  ⚠️  Some malicious clients selected: {malicious_in_committee}")
    
    committee_scores = [scores[i] for i in committee]
    print(f"\nCommittee Statistics:")
    print(f"  Average representativeness: {np.mean(committee_scores):.4f}")
    print(f"  Min representativeness: {np.min(committee_scores):.4f}")
    print(f"  Max representativeness: {np.max(committee_scores):.4f}")
    
    print("\n" + "="*70)
    print("✓ DAVS test complete - Correct amnesic implementation!")
    print("="*70)
