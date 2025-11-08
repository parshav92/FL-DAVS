import torch
from copy import deepcopy


def fedavg(client_parameters, client_weights=None):
    """
    Federated Averaging (FedAvg) aggregation
    
    Computes weighted average of client model parameters
    
    Args:
        client_parameters: List of model parameter dictionaries from clients
        client_weights: List of weights for each client (e.g., number of samples)
                       If None, uses uniform weighting
    
    Returns:
        Aggregated model parameters (state_dict)
    """
    if not client_parameters:
        raise ValueError("No client parameters provided")
    
    # Use uniform weights if not provided
    if client_weights is None:
        client_weights = [1.0] * len(client_parameters)
    
    # Normalize weights to sum to 1
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    # Initialize aggregated parameters with zeros
    aggregated_params = deepcopy(client_parameters[0])
    for key in aggregated_params.keys():
        aggregated_params[key] = torch.zeros_like(aggregated_params[key])
    
    # Weighted sum of all client parameters
    for client_param, weight in zip(client_parameters, normalized_weights):
        for key in aggregated_params.keys():
            aggregated_params[key] += weight * client_param[key]
    
    return aggregated_params


def weighted_average(values, weights):
    """
    Compute weighted average of values
    
    Args:
        values: List of values to average
        weights: List of weights
    
    Returns:
        Weighted average
    """
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")
    
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def aggregate_metrics(client_metrics, client_weights):
    """
    Aggregate metrics (loss, accuracy) from multiple clients
    
    Args:
        client_metrics: List of tuples (loss, accuracy) from each client
        client_weights: List of weights (typically number of samples)
    
    Returns:
        Tuple of (avg_loss, avg_accuracy)
    """
    losses = [metric[0] for metric in client_metrics]
    accuracies = [metric[1] for metric in client_metrics]
    
    avg_loss = weighted_average(losses, client_weights)
    avg_accuracy = weighted_average(accuracies, client_weights)
    
    return avg_loss, avg_accuracy


def krum(client_parameters, num_malicious=0, multi_krum=False):
    """
    Krum aggregation - selects parameters with smallest distance to others
    Robust against Byzantine attacks
    
    Args:
        client_parameters: List of model parameter dictionaries
        num_malicious: Number of malicious clients to tolerate
        multi_krum: If True, averages top-k clients; if False, returns single best
    
    Returns:
        Aggregated parameters using Krum selection
    
    Note: This is a placeholder for future Byzantine-robust aggregation
    """
    # For Phase 1, we'll just use FedAvg
    # Krum will be fully implemented in later phases for attack resistance
    print("Warning: Krum not fully implemented yet, using FedAvg")
    return fedavg(client_parameters)


class AggregationStrategy:
    """
    Base class for aggregation strategies
    Allows easy switching between different aggregation methods
    """
    
    def __init__(self, strategy='fedavg'):
        """
        Args:
            strategy: 'fedavg', 'krum', etc.
        """
        self.strategy = strategy
    
    def aggregate(self, client_parameters, client_weights=None, **kwargs):
        """
        Aggregate client parameters using selected strategy
        
        Args:
            client_parameters: List of parameter dictionaries
            client_weights: Optional weights for clients
            **kwargs: Additional strategy-specific parameters
        
        Returns:
            Aggregated parameters
        """
        if self.strategy == 'fedavg':
            return fedavg(client_parameters, client_weights)
        elif self.strategy == 'krum':
            return krum(client_parameters, **kwargs)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")


if __name__ == "__main__":
    # Test aggregation
    print("Testing FedAvg Aggregation...\n")
    
    # Create dummy model parameters
    dummy_params = [
        {'layer1': torch.randn(10, 5), 'layer2': torch.randn(5, 2)},
        {'layer1': torch.randn(10, 5), 'layer2': torch.randn(5, 2)},
        {'layer1': torch.randn(10, 5), 'layer2': torch.randn(5, 2)},
    ]
    
    # Test uniform weighting
    print("Test 1: Uniform weights")
    aggregated = fedavg(dummy_params)
    print(f"  ✓ Aggregated {len(dummy_params)} clients")
    print(f"  Output shapes: {[v.shape for v in aggregated.values()]}")
    
    # Test weighted aggregation
    print("\nTest 2: Weighted by number of samples")
    weights = [100, 200, 150]  # Different sample counts
    aggregated = fedavg(dummy_params, weights)
    print(f"  ✓ Weighted aggregation with weights {weights}")
    
    # Test metric aggregation
    print("\nTest 3: Metric aggregation")
    metrics = [(0.5, 85.0), (0.4, 90.0), (0.6, 80.0)]  # (loss, accuracy)
    avg_loss, avg_accuracy = aggregate_metrics(metrics, weights)
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Average Accuracy: {avg_accuracy:.2f}%")
    
    # Test strategy class
    print("\nTest 4: AggregationStrategy class")
    strategy = AggregationStrategy(strategy='fedavg')
    aggregated = strategy.aggregate(dummy_params, weights)
    print(f"  ✓ Strategy-based aggregation successful")
    
    print("\n✓ All aggregation tests passed!")
