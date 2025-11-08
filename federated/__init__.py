"""
Federated Learning package for FL+DAVS
"""

from .client import FederatedClient, create_clients
from .server import FederatedServer
from .aggregation import fedavg, weighted_average, aggregate_metrics, AggregationStrategy
from .gradient_sketching import CountSketch, GradientCompressor
from .davs_selection import DAVSSelector, DataQualityMetrics

__all__ = [
    'FederatedClient',
    'create_clients',
    'FederatedServer',
    'fedavg',
    'weighted_average',
    'aggregate_metrics',
    'AggregationStrategy',
    'CountSketch',
    'GradientCompressor',
    'DAVSSelector',
    'DataQualityMetrics'
]
