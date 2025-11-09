"""
Malicious Client Implementation
Simulates various gradient poisoning attacks
"""

from enum import Enum
import torch
import numpy as np
from typing import Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.client import FederatedClient


class AttackType(Enum):
    """Types of gradient poisoning attacks"""
    NONE = "none"  # Honest behavior
    FLIP = "flip"  # Flip gradients (multiply by -scale)
    GAUSSIAN = "gaussian"  # Add Gaussian noise
    TARGETED = "targeted"  # Target specific classes
    BYZANTINE = "byzantine"  # Random malicious behavior
    ZERO = "zero"  # Send zero gradients (lazy attack)


class MaliciousClient(FederatedClient):
    """
    Malicious federated learning client that can perform gradient poisoning attacks
    
    Inherits from FederatedClient and overrides compute_gradients()
    """
    
    def __init__(
        self,
        client_id: int,
        model,
        train_loader,
        device='cpu',
        attack_type: AttackType = AttackType.FLIP,
        attack_scale: float = 10.0,
        attack_probability: float = 1.0,
        target_class: Optional[int] = None
    ):
        """
        Args:
            client_id: Unique identifier
            model: PyTorch model
            train_loader: DataLoader for training
            device: 'cpu' or 'cuda'
            attack_type: Type of attack to perform
            attack_scale: Scaling factor for attack intensity
            attack_probability: Probability of attacking (0.0-1.0)
            target_class: Specific class to target (for targeted attacks)
        """
        super().__init__(client_id, model, train_loader, device)
        
        self.attack_type = attack_type
        self.attack_scale = attack_scale
        self.attack_probability = attack_probability
        self.target_class = target_class
        
        self.is_malicious = (attack_type != AttackType.NONE)
        self.attack_count = 0  # Track number of attacks performed
    
    def compute_gradients(self, global_params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Override to poison gradients before returning
        
        Args:
            global_params: Global model parameters
        
        Returns:
            Poisoned gradient dictionary
        """
        # First, compute honest gradients using parent method
        honest_gradients = super().compute_gradients(global_params)
        
        # Decide whether to attack this round
        if not self.is_malicious or np.random.rand() > self.attack_probability:
            return honest_gradients
        
        # Poison the gradients based on attack type
        poisoned_gradients = self._poison_gradients(honest_gradients)
        
        self.attack_count += 1
        
        return poisoned_gradients
    
    def _poison_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply gradient poisoning based on attack type
        
        Args:
            gradients: Original gradients
        
        Returns:
            Poisoned gradients
        """
        poisoned = {}
        
        if self.attack_type == AttackType.FLIP:
            # Gradient flipping: negate and scale
            for name, grad in gradients.items():
                poisoned[name] = -self.attack_scale * grad
        
        elif self.attack_type == AttackType.GAUSSIAN:
            # Add Gaussian noise
            for name, grad in gradients.items():
                noise = torch.randn_like(grad) * self.attack_scale
                poisoned[name] = grad + noise
        
        elif self.attack_type == AttackType.TARGETED:
            # Targeted attack: manipulate specific layer
            # For CNN, target the last fully connected layer
            for name, grad in gradients.items():
                if 'fc' in name or 'classifier' in name:
                    # Flip gradients for classification layer
                    poisoned[name] = -self.attack_scale * grad
                else:
                    # Keep other layers honest
                    poisoned[name] = grad
        
        elif self.attack_type == AttackType.BYZANTINE:
            # Random Byzantine behavior
            for name, grad in gradients.items():
                attack_choice = np.random.choice(['flip', 'noise', 'scale', 'zero'])
                
                if attack_choice == 'flip':
                    poisoned[name] = -grad * self.attack_scale
                elif attack_choice == 'noise':
                    poisoned[name] = torch.randn_like(grad) * self.attack_scale
                elif attack_choice == 'scale':
                    poisoned[name] = grad * self.attack_scale
                else:  # zero
                    poisoned[name] = torch.zeros_like(grad)
        
        elif self.attack_type == AttackType.ZERO:
            # Lazy attack: send zero gradients
            for name, grad in gradients.items():
                poisoned[name] = torch.zeros_like(grad)
        
        else:
            # Default: return original gradients
            poisoned = gradients
        
        return poisoned
    
    def get_attack_stats(self) -> Dict:
        """
        Get statistics about attacks performed
        
        Returns:
            Dictionary with attack statistics
        """
        return {
            'client_id': self.client_id,
            'is_malicious': self.is_malicious,
            'attack_type': self.attack_type.value,
            'attack_scale': self.attack_scale,
            'attack_probability': self.attack_probability,
            'attack_count': self.attack_count
        }
    
    def __repr__(self) -> str:
        if self.is_malicious:
            return f"MaliciousClient(id={self.client_id}, attack={self.attack_type.value}, scale={self.attack_scale})"
        return f"MaliciousClient(id={self.client_id}, HONEST)"


def create_mixed_clients(
    client_datasets: Dict,
    model_fn,
    malicious_ids: list,
    attack_type: AttackType = AttackType.FLIP,
    attack_scale: float = 10.0,
    device='cpu'
) -> Dict:
    """
    Create a mix of honest and malicious clients
    
    Args:
        client_datasets: Dictionary of {client_id: DataLoader}
        model_fn: Function that returns a new model instance
        malicious_ids: List of client IDs that should be malicious
        attack_type: Type of attack for malicious clients
        attack_scale: Attack intensity
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary of {client_id: Client} (mix of honest and malicious)
    """
    clients = {}
    
    for client_id, train_loader in client_datasets.items():
        model = model_fn()
        
        if client_id in malicious_ids:
            # Create malicious client
            clients[client_id] = MaliciousClient(
                client_id=client_id,
                model=model,
                train_loader=train_loader,
                device=device,
                attack_type=attack_type,
                attack_scale=attack_scale,
                attack_probability=1.0  # Always attack
            )
        else:
            # Create honest client
            clients[client_id] = FederatedClient(
                client_id=client_id,
                model=model,
                train_loader=train_loader,
                device=device
            )
    
    return clients
