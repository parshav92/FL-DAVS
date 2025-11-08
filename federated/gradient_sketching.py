"""
Gradient Sketching - Phase 2
Implements Count-Sketch algorithm for gradient compression
Reduces communication overhead in federated learning
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


class CountSketch:
    """
    Count-Sketch algorithm for gradient compression
    Provides dimensionality reduction while preserving important gradient information
    """
    
    def __init__(self, input_dim: int, sketch_dim: int, num_hash: int = 2):
        """
        Initialize Count-Sketch
        
        Args:
            input_dim: Original gradient dimension
            sketch_dim: Compressed sketch dimension (compression rate = input_dim/sketch_dim)
            num_hash: Number of hash functions (more = better accuracy, slower)
        """
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        self.num_hash = num_hash
        self.compression_rate = input_dim / sketch_dim
        
        # Initialize hash functions
        self.hash_indices = []
        self.hash_signs = []
        
        for _ in range(num_hash):
            # Random hash mapping: each element maps to random bucket
            indices = torch.randint(0, sketch_dim, (input_dim,))
            # Random sign: +1 or -1
            signs = torch.randint(0, 2, (input_dim,)) * 2 - 1
            
            self.hash_indices.append(indices)
            self.hash_signs.append(signs.float())
    
    def sketch(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Compress gradient using Count-Sketch
        
        Args:
            gradient: Original gradient tensor (flattened)
            
        Returns:
            Compressed sketch tensor
        """
        if gradient.dim() > 1:
            gradient = gradient.flatten()
        
        assert gradient.shape[0] == self.input_dim, \
            f"Gradient dimension {gradient.shape[0]} doesn't match input_dim {self.input_dim}"
        
        # Create sketch using multiple hash functions
        sketches = []
        for indices, signs in zip(self.hash_indices, self.hash_signs):
            sketch = torch.zeros(self.sketch_dim, device=gradient.device)
            signed_gradient = gradient * signs.to(gradient.device)
            
            # Accumulate values in sketch buckets
            sketch.index_add_(0, indices.to(gradient.device), signed_gradient)
            sketches.append(sketch)
        
        # Average across hash functions for better estimation
        final_sketch = torch.stack(sketches).mean(dim=0)
        return final_sketch
    
    def unSketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Decompress sketch back to original dimension
        
        Args:
            sketch: Compressed sketch tensor
            
        Returns:
            Decompressed gradient (approximate)
        """
        assert sketch.shape[0] == self.sketch_dim, \
            f"Sketch dimension {sketch.shape[0]} doesn't match sketch_dim {self.sketch_dim}"
        
        # Reconstruct using hash functions
        reconstructed = []
        for indices, signs in zip(self.hash_indices, self.hash_signs):
            # Lookup sketch values and apply signs
            values = sketch[indices.to(sketch.device)]
            recon = values * signs.to(sketch.device)
            reconstructed.append(recon)
        
        # Median for robustness (or mean for speed)
        if len(reconstructed) > 1:
            final_recon = torch.stack(reconstructed).median(dim=0)[0]
        else:
            final_recon = reconstructed[0]
        
        return final_recon


class GradientCompressor:
    """
    Manages gradient compression for all model parameters
    """
    
    def __init__(self, model: torch.nn.Module, compression_rate: float = 0.1, num_hash: int = 2):
        """
        Initialize gradient compressor
        
        Args:
            model: Neural network model
            compression_rate: Target compression rate (0.1 = 10x compression)
            num_hash: Number of hash functions for Count-Sketch
        """
        self.model = model
        self.compression_rate = compression_rate
        self.num_hash = num_hash
        
        # Create sketcher for each parameter
        self.sketchers = {}
        self.param_shapes = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                input_dim = param.numel()
                sketch_dim = max(int(input_dim * compression_rate), 1)
                
                self.sketchers[name] = CountSketch(input_dim, sketch_dim, num_hash)
                self.param_shapes[name] = param.shape
        
        # Calculate compression statistics
        self._calculate_stats()
    
    def _calculate_stats(self):
        """Calculate compression statistics"""
        original_size = sum(s.input_dim for s in self.sketchers.values())
        compressed_size = sum(s.sketch_dim for s in self.sketchers.values())
        
        self.original_size = original_size
        self.compressed_size = compressed_size
        self.actual_compression_rate = compressed_size / original_size
        self.bandwidth_reduction = (1 - self.actual_compression_rate) * 100
    
    def compress_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compress all gradients
        
        Args:
            gradients: Dictionary of parameter name -> gradient tensor
            
        Returns:
            Dictionary of parameter name -> compressed sketch
        """
        compressed = {}
        for name, grad in gradients.items():
            if name in self.sketchers:
                flat_grad = grad.flatten()
                sketch = self.sketchers[name].sketch(flat_grad)
                compressed[name] = sketch
        
        return compressed
    
    def decompress_gradients(self, compressed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decompress sketches back to gradients
        
        Args:
            compressed: Dictionary of parameter name -> sketch
            
        Returns:
            Dictionary of parameter name -> decompressed gradient
        """
        decompressed = {}
        for name, sketch in compressed.items():
            if name in self.sketchers:
                flat_grad = self.sketchers[name].unSketch(sketch)
                original_shape = self.param_shapes[name]
                grad = flat_grad.reshape(original_shape)
                decompressed[name] = grad
        
        return decompressed
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics"""
        return {
            'original_size': self.original_size,
            'compressed_size': self.compressed_size,
            'compression_rate': self.actual_compression_rate,
            'bandwidth_reduction_percent': self.bandwidth_reduction
        }


def compress_model_updates(model_updates: List[Dict[str, torch.Tensor]], 
                          compressor: GradientCompressor) -> List[Dict[str, torch.Tensor]]:
    """
    Compress model updates from multiple clients
    
    Args:
        model_updates: List of model update dictionaries
        compressor: Gradient compressor instance
        
    Returns:
        List of compressed updates
    """
    compressed_updates = []
    for update in model_updates:
        compressed = compressor.compress_gradients(update)
        compressed_updates.append(compressed)
    
    return compressed_updates


def decompress_and_aggregate(compressed_updates: List[Dict[str, torch.Tensor]], 
                            weights: List[float],
                            compressor: GradientCompressor) -> Dict[str, torch.Tensor]:
    """
    Decompress and aggregate compressed updates
    
    Args:
        compressed_updates: List of compressed updates
        weights: Weight for each update (typically based on data size)
        compressor: Gradient compressor instance
        
    Returns:
        Aggregated decompressed gradients
    """
    # Decompress all updates
    decompressed_updates = []
    for compressed in compressed_updates:
        decompressed = compressor.decompress_gradients(compressed)
        decompressed_updates.append(decompressed)
    
    # Weighted aggregation
    aggregated = {}
    total_weight = sum(weights)
    
    for name in decompressed_updates[0].keys():
        weighted_sum = sum(
            update[name] * (weight / total_weight) 
            for update, weight in zip(decompressed_updates, weights)
        )
        aggregated[name] = weighted_sum
    
    return aggregated


if __name__ == "__main__":
    # Test gradient sketching
    print("Testing Gradient Sketching...\n")
    
    # Create a simple test model
    from models.cnn_model import SimpleCNN
    model = SimpleCNN(num_classes=9)
    
    # Initialize compressor
    print("="*60)
    print("Initializing Gradient Compressor")
    print("="*60)
    compressor = GradientCompressor(model, compression_rate=0.1, num_hash=3)
    
    stats = compressor.get_compression_stats()
    print(f"Original size: {stats['original_size']:,} parameters")
    print(f"Compressed size: {stats['compressed_size']:,} parameters")
    print(f"Compression rate: {stats['compression_rate']:.2%}")
    print(f"Bandwidth reduction: {stats['bandwidth_reduction_percent']:.1f}%")
    
    # Test compression/decompression
    print("\n" + "="*60)
    print("Testing Compression/Decompression")
    print("="*60)
    
    # Create dummy gradients
    dummy_gradients = {
        name: torch.randn_like(param) 
        for name, param in model.named_parameters()
    }
    
    # Compress
    compressed = compressor.compress_gradients(dummy_gradients)
    print(f"✓ Compressed {len(compressed)} parameter groups")
    
    # Decompress
    decompressed = compressor.decompress_gradients(compressed)
    print(f"✓ Decompressed to {len(decompressed)} parameter groups")
    
    # Calculate reconstruction error
    total_error = 0.0
    total_norm = 0.0
    for name in dummy_gradients.keys():
        if name in decompressed:
            error = torch.norm(dummy_gradients[name] - decompressed[name]).item()
            norm = torch.norm(dummy_gradients[name]).item()
            total_error += error
            total_norm += norm
    
    relative_error = (total_error / total_norm) * 100
    print(f"\nReconstruction error: {relative_error:.2f}%")
    print("="*60)
    print("\n✓ Gradient sketching test complete!")
