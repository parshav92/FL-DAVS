"""
Federated Learning package for FL+DAVS
"""

from .client import FederatedClient, create_clients
from .server import FederatedServer
from .aggregation import fedavg, weighted_average, aggregate_metrics, AggregationStrategy

__all__ = [
    'FederatedClient',
    'create_clients',
    'FederatedServer',
    'fedavg',
    'weighted_average',
    'aggregate_metrics',
    'AggregationStrategy'
]
