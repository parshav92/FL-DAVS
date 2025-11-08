"""
DAVS (Data-Aware Verifier Selection) - Phase 3
Implements intelligent committee selection based on data quality and contribution
Byzantine-resilient verifier selection for federated learning
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class DataQualityMetrics:
    """
    Calculates data quality metrics for client selection
    """
    
    @staticmethod
    def calculate_gradient_norm(gradients: Dict[str, torch.Tensor]) -> float:
        """Calculate L2 norm of gradients"""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad).item() ** 2
        return np.sqrt(total_norm)
    
    @staticmethod
    def calculate_gradient_variance(client_gradients: List[Dict[str, torch.Tensor]]) -> Dict[int, float]:
        """
        Calculate variance of each client's gradients from mean
        Lower variance = more consistent with others
        """
        if not client_gradients:
            return {}
        
        # Calculate mean gradient
        mean_gradients = {}
        for param_name in client_gradients[0].keys():
            param_grads = [cg[param_name] for cg in client_gradients]
            mean_gradients[param_name] = torch.stack(param_grads).mean(dim=0)
        
        # Calculate variance for each client
        variances = {}
        for client_id, client_grad in enumerate(client_gradients):
            variance = 0.0
            for param_name in client_grad.keys():
                diff = client_grad[param_name] - mean_gradients[param_name]
                variance += torch.norm(diff).item() ** 2
            variances[client_id] = variance
        
        return variances
    
    @staticmethod
    def calculate_loss_improvement(prev_loss: float, curr_loss: float) -> float:
        """Calculate improvement in loss"""
        if prev_loss == 0:
            return 0.0
        return (prev_loss - curr_loss) / prev_loss
    
    @staticmethod
    def calculate_data_diversity(data_sizes: Dict[int, int], 
                                class_distributions: Dict[int, np.ndarray]) -> Dict[int, float]:
        """
        Calculate data diversity score for each client
        Higher diversity = better representation
        """
        diversity_scores = {}
        
        for client_id, class_dist in class_distributions.items():
            # Shannon entropy as diversity measure
            probabilities = class_dist / (class_dist.sum() + 1e-10)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            
            # Normalize by data size (larger datasets with diversity are better)
            data_size = data_sizes.get(client_id, 1)
            diversity_scores[client_id] = entropy * np.log(data_size + 1)
        
        return diversity_scores


class DAVSSelector:
    """
    Data-Aware Verifier Selection
    Selects optimal committee of verifiers based on multiple criteria
    """
    
    def __init__(self, num_clients: int, committee_size: int = None, 
                 selection_strategy: str = 'weighted'):
        """
        Initialize DAVS selector
        
        Args:
            num_clients: Total number of clients
            committee_size: Size of verification committee (default: sqrt(num_clients))
            selection_strategy: 'weighted', 'top_k', or 'probabilistic'
        """
        self.num_clients = num_clients
        self.committee_size = committee_size or max(int(np.sqrt(num_clients)), 3)
        self.selection_strategy = selection_strategy
        
        # Track client performance over time
        self.client_history = defaultdict(lambda: {
            'contributions': 0,
            'total_loss_improvement': 0.0,
            'avg_gradient_norm': 0.0,
            'reliability_score': 1.0,
            'data_quality_score': 1.0
        })
        
        # Byzantine detection
        self.suspected_byzantine = set()
    
    def calculate_client_scores(self, 
                                client_gradients: List[Dict[str, torch.Tensor]],
                                data_sizes: List[int],
                                loss_improvements: List[float],
                                class_distributions: Dict[int, np.ndarray] = None) -> Dict[int, float]:
        """
        Calculate comprehensive scores for each client
        
        Args:
            client_gradients: List of gradient dictionaries from clients
            data_sizes: List of data sizes for each client
            loss_improvements: List of loss improvements for each client
            class_distributions: Optional class distribution per client
            
        Returns:
            Dictionary mapping client_id to overall score
        """
        num_active_clients = len(client_gradients)
        
        # 1. Gradient-based metrics
        gradient_norms = [
            DataQualityMetrics.calculate_gradient_norm(grad) 
            for grad in client_gradients
        ]
        gradient_variances = DataQualityMetrics.calculate_gradient_variance(client_gradients)
        
        # 2. Data quality metrics
        data_sizes_dict = {i: size for i, size in enumerate(data_sizes)}
        if class_distributions is not None:
            diversity_scores = DataQualityMetrics.calculate_data_diversity(
                data_sizes_dict, class_distributions
            )
        else:
            diversity_scores = {i: 1.0 for i in range(num_active_clients)}
        
        # 3. Calculate composite scores
        scores = {}
        for client_id in range(num_active_clients):
            # Skip suspected Byzantine nodes
            if client_id in self.suspected_byzantine:
                scores[client_id] = 0.0
                continue
            
            # Normalize metrics
            norm_grad = gradient_norms[client_id] / (max(gradient_norms) + 1e-10)
            norm_var = 1.0 - (gradient_variances[client_id] / (max(gradient_variances.values()) + 1e-10))
            norm_loss = max(0.0, loss_improvements[client_id])
            norm_diversity = diversity_scores.get(client_id, 0.5)
            norm_data_size = data_sizes[client_id] / (max(data_sizes) + 1)
            
            # Weighted combination (tunable weights)
            score = (
                0.25 * norm_grad +           # Gradient magnitude
                0.25 * norm_var +            # Consistency with others
                0.20 * norm_loss +           # Loss improvement
                0.15 * norm_diversity +      # Data diversity
                0.15 * norm_data_size        # Data volume
            )
            
            # Apply historical reliability
            history = self.client_history[client_id]
            reliability_factor = history['reliability_score']
            score *= reliability_factor
            
            scores[client_id] = score
        
        return scores
    
    def select_committee(self, 
                        client_gradients: List[Dict[str, torch.Tensor]],
                        data_sizes: List[int],
                        loss_improvements: List[float],
                        class_distributions: Dict[int, np.ndarray] = None) -> List[int]:
        """
        Select verification committee using DAVS
        
        Args:
            client_gradients: Gradients from all clients
            data_sizes: Data sizes for each client
            loss_improvements: Loss improvements for each client
            class_distributions: Optional class distributions
            
        Returns:
            List of selected client IDs
        """
        # Calculate scores
        scores = self.calculate_client_scores(
            client_gradients, data_sizes, loss_improvements, class_distributions
        )
        
        # Select based on strategy
        if self.selection_strategy == 'top_k':
            # Select top K clients
            selected = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            committee = [client_id for client_id, _ in selected[:self.committee_size]]
        
        elif self.selection_strategy == 'probabilistic':
            # Probabilistic selection based on scores
            total_score = sum(scores.values())
            if total_score == 0:
                # Fallback to random selection
                committee = np.random.choice(
                    list(scores.keys()), 
                    size=min(self.committee_size, len(scores)), 
                    replace=False
                ).tolist()
            else:
                probabilities = np.array([scores[i] for i in range(len(scores))])
                probabilities = probabilities / probabilities.sum()
                committee = np.random.choice(
                    len(scores), 
                    size=min(self.committee_size, len(scores)), 
                    replace=False,
                    p=probabilities
                ).tolist()
        
        else:  # 'weighted' (default)
            # Weighted selection ensuring diversity
            committee = self._weighted_diverse_selection(scores, data_sizes)
        
        return committee
    
    def _weighted_diverse_selection(self, scores: Dict[int, float], 
                                   data_sizes: List[int]) -> List[int]:
        """Select committee with both high scores and diversity"""
        # Sort by score
        sorted_clients = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        committee = []
        for client_id, score in sorted_clients:
            if len(committee) >= self.committee_size:
                break
            
            # Add if score is positive and not Byzantine
            if score > 0 and client_id not in self.suspected_byzantine:
                committee.append(client_id)
        
        # If committee too small, add random non-Byzantine clients
        if len(committee) < self.committee_size:
            remaining = [
                i for i in range(len(scores)) 
                if i not in committee and i not in self.suspected_byzantine
            ]
            needed = min(self.committee_size - len(committee), len(remaining))
            committee.extend(np.random.choice(remaining, size=needed, replace=False).tolist())
        
        return committee
    
    def update_client_history(self, client_id: int, 
                             gradient_norm: float,
                             loss_improvement: float,
                             was_selected: bool):
        """Update historical performance of a client"""
        history = self.client_history[client_id]
        
        # Update contribution count
        if was_selected:
            history['contributions'] += 1
        
        # Update moving averages
        alpha = 0.7  # Smoothing factor
        history['avg_gradient_norm'] = (
            alpha * history['avg_gradient_norm'] + (1 - alpha) * gradient_norm
        )
        history['total_loss_improvement'] += loss_improvement
        
        # Update reliability score (decays if not selected or poor performance)
        if was_selected and loss_improvement > 0:
            history['reliability_score'] = min(1.0, history['reliability_score'] + 0.1)
        else:
            history['reliability_score'] = max(0.1, history['reliability_score'] - 0.05)
    
    def detect_byzantine(self, 
                        client_gradients: List[Dict[str, torch.Tensor]],
                        threshold: float = 3.0) -> List[int]:
        """
        Detect potential Byzantine (malicious) clients
        Using gradient variance and statistical outlier detection
        
        Args:
            client_gradients: Gradients from all clients
            threshold: Standard deviations from mean to flag as Byzantine
            
        Returns:
            List of suspected Byzantine client IDs
        """
        if len(client_gradients) < 3:
            return []
        
        # Calculate gradient norms
        norms = [
            DataQualityMetrics.calculate_gradient_norm(grad) 
            for grad in client_gradients
        ]
        
        # Statistical outlier detection
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        byzantine_clients = []
        for client_id, norm in enumerate(norms):
            # Flag if too far from mean
            if abs(norm - mean_norm) > threshold * std_norm:
                byzantine_clients.append(client_id)
                self.suspected_byzantine.add(client_id)
        
        return byzantine_clients
    
    def get_committee_stats(self, committee: List[int], 
                           scores: Dict[int, float]) -> Dict[str, float]:
        """Get statistics about selected committee"""
        committee_scores = [scores[i] for i in committee]
        
        return {
            'size': len(committee),
            'avg_score': np.mean(committee_scores),
            'min_score': np.min(committee_scores),
            'max_score': np.max(committee_scores),
            'score_std': np.std(committee_scores)
        }


if __name__ == "__main__":
    # Test DAVS committee selection
    print("Testing DAVS Committee Selection...\n")
    
    from models.cnn_model import SimpleCNN
    
    # Simulate 10 clients
    num_clients = 10
    model = SimpleCNN(num_classes=9)
    
    print("="*60)
    print("Initializing DAVS Selector")
    print("="*60)
    selector = DAVSSelector(num_clients=num_clients, committee_size=5, 
                           selection_strategy='weighted')
    print(f"Total clients: {num_clients}")
    print(f"Committee size: {selector.committee_size}")
    print(f"Selection strategy: {selector.selection_strategy}")
    
    # Create dummy client data
    print("\n" + "="*60)
    print("Simulating Client Gradients and Metrics")
    print("="*60)
    
    client_gradients = []
    data_sizes = []
    loss_improvements = []
    class_distributions = {}
    
    for i in range(num_clients):
        # Create dummy gradients
        dummy_grad = {
            name: torch.randn_like(param) * (1.0 + 0.1 * np.random.randn())
            for name, param in model.named_parameters()
        }
        client_gradients.append(dummy_grad)
        
        # Random data sizes (mimicking non-IID)
        data_sizes.append(int(5000 + 5000 * np.random.rand()))
        
        # Random loss improvements
        loss_improvements.append(0.1 * np.random.rand())
        
        # Random class distributions
        class_dist = np.random.dirichlet(np.ones(9) * 0.5)
        class_distributions[i] = class_dist * 1000
    
    print(f"✓ Created gradients for {num_clients} clients")
    print(f"✓ Data sizes range: {min(data_sizes)} - {max(data_sizes)}")
    print(f"✓ Loss improvements range: {min(loss_improvements):.3f} - {max(loss_improvements):.3f}")
    
    # Calculate scores
    print("\n" + "="*60)
    print("Calculating Client Scores")
    print("="*60)
    
    scores = selector.calculate_client_scores(
        client_gradients, data_sizes, loss_improvements, class_distributions
    )
    
    for client_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"Client {client_id}: Score = {score:.4f}, Data size = {data_sizes[client_id]}")
    
    # Select committee
    print("\n" + "="*60)
    print("Selecting Verification Committee")
    print("="*60)
    
    committee = selector.select_committee(
        client_gradients, data_sizes, loss_improvements, class_distributions
    )
    
    print(f"✓ Selected committee: {committee}")
    
    stats = selector.get_committee_stats(committee, scores)
    print(f"\nCommittee Statistics:")
    print(f"  Size: {stats['size']}")
    print(f"  Average score: {stats['avg_score']:.4f}")
    print(f"  Score range: [{stats['min_score']:.4f}, {stats['max_score']:.4f}]")
    print(f"  Score std dev: {stats['score_std']:.4f}")
    
    # Test Byzantine detection
    print("\n" + "="*60)
    print("Testing Byzantine Detection")
    print("="*60)
    
    # Add a Byzantine client (extreme gradient)
    byzantine_grad = {
        name: torch.randn_like(param) * 10.0  # 10x larger gradients
        for name, param in model.named_parameters()
    }
    client_gradients.append(byzantine_grad)
    
    byzantine = selector.detect_byzantine(client_gradients, threshold=2.5)
    if byzantine:
        print(f"⚠️  Detected Byzantine clients: {byzantine}")
    else:
        print("✓ No Byzantine clients detected")
    
    print("\n" + "="*60)
    print("✓ DAVS committee selection test complete!")
    print("="*60)
