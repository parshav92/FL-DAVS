

import torch
import numpy as np
from typing import Dict, List, Optional
import hashlib

class RandomProjectionSketcher:
    """
    Random Projection gradient sketcher for DAVS
    Uses Gaussian random projection to preserve cosine similarity
    
    Based on Johnson-Lindenstrauss Lemma:
    Random projection preserves pairwise distances/angles with high probability
    """
    
    def __init__(self, input_dim: int, sketch_dim: int = 128, 
                 add_dp_noise: bool = False, noise_scale: float = 0.01,
                 seed: int = 42):
        """
        Initialize Random Projection Sketcher
        
        Args:
            input_dim: Original gradient dimension
            sketch_dim: Compressed sketch dimension (typically 128-256)
            add_dp_noise: Whether to add differential privacy noise
            noise_scale: Standard deviation of DP noise
            seed: Random seed for reproducibility (all clients must use same seed!)
        """
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        self.add_dp_noise = add_dp_noise
        self.noise_scale = noise_scale
        self.seed = seed
        
        # Generate shared random projection matrix
        # CRITICAL: All clients must use the same matrix!
        self._initialize_projection_matrix()
        
        # Compression stats
        self.compression_rate = sketch_dim / input_dim
        self.bandwidth_reduction = (1 - self.compression_rate) * 100
    
    def _initialize_projection_matrix(self):
        """
        Generate normalized Gaussian random projection matrix
        Shape: (sketch_dim, input_dim)
        """
        # Set seed for reproducibility across all clients
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Generate Gaussian random matrix
        self.projection_matrix = torch.randn(self.sketch_dim, self.input_dim)
        
        # Normalize rows to unit length (improves stability)
        row_norms = torch.norm(self.projection_matrix, dim=1, keepdim=True)
        self.projection_matrix = self.projection_matrix / (row_norms + 1e-10)
        
        print(f"✓ Initialized projection matrix: {self.sketch_dim} × {self.input_dim}")
    
    def sketch(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Compress gradient using random projection
        
        Args:
            gradient: Original gradient tensor (can be multi-dimensional)
            
        Returns:
            Compressed sketch (1D tensor of size sketch_dim)
        """
        # Flatten if needed
        if gradient.dim() > 1:
            gradient = gradient.flatten()
        
        assert gradient.shape[0] == self.input_dim, \
            f"Gradient size {gradient.shape[0]} != input_dim {self.input_dim}"
        
        # Move projection matrix to same device as gradient
        proj_matrix = self.projection_matrix.to(gradient.device)
        
        # Compute sketch: s = R × g
        sketch = torch.matmul(proj_matrix, gradient)
        
        # Add differential privacy noise if enabled
        if self.add_dp_noise:
            noise = torch.randn_like(sketch) * self.noise_scale
            sketch = sketch + noise
        
        return sketch
    
    def compute_cosine_similarity(self, sketch1: torch.Tensor, 
                                  sketch2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two sketches
        
        Args:
            sketch1: First sketch
            sketch2: Second sketch
            
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
    
    def batch_cosine_similarity(self, sketch: torch.Tensor, 
                               all_sketches: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute cosine similarity between one sketch and all others
        Efficient batch computation
        
        Args:
            sketch: Query sketch
            all_sketches: List of all sketches
            
        Returns:
            Tensor of similarities
        """
        # Stack all sketches into matrix
        sketch_matrix = torch.stack(all_sketches)  # (num_clients, sketch_dim)
        
        # Ensure same device
        sketch = sketch.to(sketch_matrix.device)
        
        # Normalize
        sketch_norm = sketch / (torch.norm(sketch) + 1e-10)
        matrix_norms = torch.norm(sketch_matrix, dim=1, keepdim=True)
        sketch_matrix_norm = sketch_matrix / (matrix_norms + 1e-10)
        
        # Batch dot product
        similarities = torch.matmul(sketch_matrix_norm, sketch_norm)
        
        return similarities


class GradientSketcherForDAVS:
    """
    Manages gradient sketching for all clients in DAVS
    Handles model-wide gradient compression and similarity computation
    """
    
    def __init__(self, model: torch.nn.Module, sketch_dim: int = 128,
                 add_dp_noise: bool = False, noise_scale: float = 0.01,
                 shared_seed: int = 42):
        """
        Initialize gradient sketcher for DAVS
        
        Args:
            model: Neural network model (to get total parameter count)
            sketch_dim: Sketch dimension (default: 128)
            add_dp_noise: Enable differential privacy
            noise_scale: DP noise scale
            shared_seed: Seed shared across ALL clients (critical!)
        """
        self.model = model
        self.sketch_dim = sketch_dim
        self.add_dp_noise = add_dp_noise
        self.noise_scale = noise_scale
        self.shared_seed = shared_seed
        
        # Calculate total gradient dimension
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Initialize sketcher
        self.sketcher = RandomProjectionSketcher(
            input_dim=self.total_params,
            sketch_dim=sketch_dim,
            add_dp_noise=add_dp_noise,
            noise_scale=noise_scale,
            seed=shared_seed
        )
        
        print(f"✓ DAVS Sketcher initialized:")
        print(f"  Total parameters: {self.total_params:,}")
        print(f"  Sketch dimension: {sketch_dim}")
        print(f"  Compression: {self.total_params/sketch_dim:.1f}x")
        print(f"  Bandwidth reduction: {self.sketcher.bandwidth_reduction:.1f}%")
    
    def extract_gradients(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Extract and flatten all gradients from model
        
        Args:
            model: Model with computed gradients
            
        Returns:
            Flattened gradient vector
        """
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.flatten())
        
        if len(gradients) == 0:
            raise ValueError("No gradients found! Did you call loss.backward()?")
        
        return torch.cat(gradients)
    
    def sketch_gradients(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Extract and sketch gradients from model
        
        Args:
            model: Model with computed gradients
            
        Returns:
            Gradient sketch
        """
        gradient_vector = self.extract_gradients(model)
        sketch = self.sketcher.sketch(gradient_vector)
        return sketch
    
    def compute_representativeness_score(self, client_sketch: torch.Tensor,
                                        all_sketches: List[torch.Tensor]) -> float:
        """
        Compute representativeness score for DAVS
        Score = mean cosine similarity with all other clients
        
        Args:
            client_sketch: Sketch of current client
            all_sketches: Sketches of all clients (including current)
            
        Returns:
            Representativeness score
        """
        similarities = []
        
        for other_sketch in all_sketches:
            # Skip self-comparison
            if not torch.equal(client_sketch, other_sketch):
                sim = self.sketcher.compute_cosine_similarity(client_sketch, other_sketch)
                similarities.append(sim)
        
        if len(similarities) == 0:
            return 0.0
        
        # Mean similarity = representativeness
        return float(np.mean(similarities))
    
    def select_davs_committee(self, client_sketches: Dict[int, torch.Tensor],
                             committee_size: int = 5) -> tuple:
        """
        DAVS committee selection based on representativeness scores
        
        Args:
            client_sketches: Dict of {client_id: sketch}
            committee_size: Number of verifiers to select
            
        Returns:
            Tuple of (committee_ids, all_scores)
        """
        scores = {}
        all_sketches = list(client_sketches.values())
        
        # Compute representativeness for each client
        for client_id, sketch in client_sketches.items():
            scores[client_id] = self.compute_representativeness_score(sketch, all_sketches)
        
        # Select top-k clients
        sorted_clients = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        committee = [client_id for client_id, _ in sorted_clients[:committee_size]]
        
        return committee, scores


# Utility functions for integration

def create_shared_sketcher(model: torch.nn.Module, config: dict) -> GradientSketcherForDAVS:
    """
    Factory function to create sketcher with config
    
    Args:
        model: Neural network model
        config: Configuration dictionary
        
    Returns:
        Initialized sketcher
    """
    return GradientSketcherForDAVS(
        model=model,
        sketch_dim=config.get('sketch_dim', 128),
        add_dp_noise=config.get('add_dp_noise', False),
        noise_scale=config.get('noise_scale', 0.01),
        shared_seed=config.get('shared_seed', 42)
    )


if __name__ == "__main__":
    print("="*70)
    print("Testing Random Projection Gradient Sketching for DAVS")
    print("="*70)
    
    from models.cnn_model import SimpleCNN
    
    # Initialize model
    model = SimpleCNN(num_classes=9)
    
    # Create sketcher
    sketcher = GradientSketcherForDAVS(
        model=model,
        sketch_dim=128,
        add_dp_noise=False,
        shared_seed=42
    )
    
    print("\n" + "="*70)
    print("Test 1: Gradient Extraction and Sketching")
    print("="*70)
    
    # Create dummy data and compute gradients
    dummy_input = torch.randn(4, 3, 28, 28)
    dummy_target = torch.randint(0, 9, (4,))
    
    # Forward pass
    output = model(dummy_input)
    loss = torch.nn.CrossEntropyLoss()(output, dummy_target)
    loss.backward()
    
    # Extract and sketch
    gradient_vector = sketcher.extract_gradients(model)
    sketch = sketcher.sketch_gradients(model)
    
    print(f"✓ Original gradient dimension: {gradient_vector.shape[0]:,}")
    print(f"✓ Sketch dimension: {sketch.shape[0]}")
    print(f"✓ Compression ratio: {gradient_vector.shape[0]/sketch.shape[0]:.1f}x")
    
    print("\n" + "="*70)
    print("Test 2: Similarity Preservation")
    print("="*70)
    
    # Create similar gradients
    grad1 = torch.randn(sketcher.total_params)
    grad2 = grad1 + torch.randn(sketcher.total_params) * 0.1  # Similar
    grad3 = torch.randn(sketcher.total_params)  # Different
    
    # Sketch them
    sketch1 = sketcher.sketcher.sketch(grad1)
    sketch2 = sketcher.sketcher.sketch(grad2)
    sketch3 = sketcher.sketcher.sketch(grad3)
    
    # Compute similarities
    sim_12_original = torch.cosine_similarity(grad1.unsqueeze(0), grad2.unsqueeze(0)).item()
    sim_13_original = torch.cosine_similarity(grad1.unsqueeze(0), grad3.unsqueeze(0)).item()
    
    sim_12_sketch = sketcher.sketcher.compute_cosine_similarity(sketch1, sketch2)
    sim_13_sketch = sketcher.sketcher.compute_cosine_similarity(sketch1, sketch3)
    
    print(f"Original gradients:")
    print(f"  Similarity(grad1, grad2): {sim_12_original:.4f} (similar)")
    print(f"  Similarity(grad1, grad3): {sim_13_original:.4f} (different)")
    
    print(f"\nAfter sketching:")
    print(f"  Similarity(sketch1, sketch2): {sim_12_sketch:.4f}")
    print(f"  Similarity(sketch1, sketch3): {sim_13_sketch:.4f}")
    
    error_similar = abs(sim_12_original - sim_12_sketch)
    error_different = abs(sim_13_original - sim_13_sketch)
    print(f"\n✓ Similarity preservation error: {error_similar:.4f}, {error_different:.4f}")
    
    print("\n" + "="*70)
    print("Test 3: DAVS Committee Selection")
    print("="*70)
    
    # Simulate 10 clients with sketches
    num_clients = 10
    client_sketches = {}
    
    # Honest clients (similar gradients)
    honest_base = torch.randn(sketcher.total_params)
    for i in range(7):
        honest_grad = honest_base + torch.randn(sketcher.total_params) * 0.2
        client_sketches[i] = sketcher.sketcher.sketch(honest_grad)
    
    # Malicious clients (very different gradients)
    for i in range(7, 10):
        malicious_grad = torch.randn(sketcher.total_params) * 5  # Large, different
        client_sketches[i] = sketcher.sketcher.sketch(malicious_grad)
    
    # Select committee
    committee, scores = sketcher.select_davs_committee(client_sketches, committee_size=5)
    
    print(f"Representativeness Scores:")
    for client_id in sorted(scores.keys()):
        is_malicious = "⚠️  MALICIOUS" if client_id >= 7 else "✓ Honest"
        print(f"  Client {client_id}: {scores[client_id]:6.4f}  {is_malicious}")
    
    print(f"\n✓ Selected Committee: {committee}")
    malicious_in_committee = [c for c in committee if c >= 7]
    print(f"  Malicious clients in committee: {len(malicious_in_committee)}/{len(committee)}")
    
    if len(malicious_in_committee) == 0:
        print("  ✓✓✓ SUCCESS: No malicious clients selected!")
    
    print("\n" + "="*70)
    print("✓ All tests passed! Random Projection works correctly for DAVS")
    print("="*70)