"""
Practical Byzantine Fault Tolerance (PBFT) Consensus
Adapted for Federated Learning with DAVS-selected committee
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import json
import time
import torch
import numpy as np


class PBFTPhase(Enum):
    """PBFT protocol phases"""
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    EXECUTED = "executed"


class PBFTMessage:
    """
    Message exchanged during PBFT protocol
    """
    
    def __init__(
        self,
        phase: PBFTPhase,
        round_num: int,
        sender_id: int,
        model_hash: str,
        sequence_num: int = 0,
        view_num: int = 0
    ):
        """
        Args:
            phase: PBFT phase (PRE_PREPARE, PREPARE, COMMIT)
            round_num: FL round number
            sender_id: Node ID sending the message
            model_hash: Hash of the model update being validated
            sequence_num: Sequence number for ordering
            view_num: View number (for view changes, not used in PoC)
        """
        self.phase = phase
        self.round_num = round_num
        self.sender_id = sender_id
        self.model_hash = model_hash
        self.sequence_num = sequence_num
        self.view_num = view_num
        self.timestamp = time.time()
    
    def __repr__(self) -> str:
        return f"PBFTMessage(phase={self.phase.value}, sender={self.sender_id}, hash={self.model_hash[:8]}...)"


class PBFTValidator:
    """
    Single validator node in PBFT committee
    """
    
    def __init__(
        self,
        node_id: int,
        model_weights: Dict[str, torch.Tensor],
        is_primary: bool = False
    ):
        """
        Args:
            node_id: Unique identifier for this validator
            model_weights: The aggregated model weights to validate
            is_primary: Whether this node is the primary (leader)
        """
        self.node_id = node_id
        self.model_weights = model_weights
        self.is_primary = is_primary
        
        # Track received messages
        self.prepare_messages: List[PBFTMessage] = []
        self.commit_messages: List[PBFTMessage] = []
        
        # State
        self.prepared = False
        self.committed = False
    
    def validate_model_update(self, model_weights: Dict[str, torch.Tensor]) -> bool:
        """
        Validate the quality of the aggregated model update
        
        Checks:
        1. No NaN or Inf values
        2. Parameter magnitudes within reasonable bounds
        3. Gradient norms not too large (no gradient explosion)
        
        Args:
            model_weights: Model parameters to validate
        
        Returns:
            True if update is valid, False otherwise
        """
        try:
            for name, param in model_weights.items():
                # Check for NaN or Inf
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"❌ Validator {self.node_id}: NaN/Inf detected in {name}")
                    return False
                
                # Check parameter magnitude (prevent extreme values)
                param_abs_max = param.abs().max().item()
                if param_abs_max > 1000:  # Threshold for extreme values
                    print(f"❌ Validator {self.node_id}: Extreme value {param_abs_max} in {name}")
                    return False
                
                # Check gradient norm
                param_norm = param.norm().item()
                if param_norm > 10000:  # Threshold for gradient explosion
                    print(f"❌ Validator {self.node_id}: Large norm {param_norm} in {name}")
                    return False
            
            return True
        
        except Exception as e:
            print(f"❌ Validator {self.node_id}: Validation error: {e}")
            return False
    
    def vote(self, model_weights: Dict[str, torch.Tensor]) -> bool:
        """
        Cast vote on whether to accept the model update
        
        Args:
            model_weights: Model to vote on
        
        Returns:
            True to approve, False to reject
        """
        return self.validate_model_update(model_weights)


class PBFTConsensus:
    """
    PBFT Consensus protocol for DAVS-selected committee
    
    Protocol:
    1. PRE-PREPARE: Primary broadcasts aggregated model to committee
    2. PREPARE: Each validator validates and broadcasts PREPARE message
    3. COMMIT: If node receives 2f+1 PREPARE messages, broadcast COMMIT
    4. EXECUTE: If node receives 2f+1 COMMIT messages, accept update
    """
    
    def __init__(
        self,
        committee: List[int],
        max_faulty: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Args:
            committee: List of node IDs in the committee (selected by DAVS)
            max_faulty: Maximum number of Byzantine nodes tolerable (f)
                       Default: (len(committee) - 1) // 3
            verbose: Print detailed protocol messages
        """
        self.committee = committee
        self.committee_size = len(committee)
        
        # Byzantine fault tolerance: need 3f + 1 nodes minimum
        if max_faulty is None:
            self.f = (self.committee_size - 1) // 3
        else:
            self.f = max_faulty
        
        # Quorum: 2f + 1 (majority needed)
        self.quorum_size = 2 * self.f + 1
        
        # Primary is node with highest DAVS score (first in committee list)
        self.primary_id = committee[0]
        
        self.verbose = verbose
        
        # Validate committee size
        if self.committee_size < 3 * self.f + 1:
            print(f"⚠️  Warning: Committee size {self.committee_size} may be too small for f={self.f}")
    
    def run_consensus(
        self,
        round_num: int,
        aggregated_model: Dict[str, torch.Tensor],
        client_models: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run full PBFT consensus protocol
        
        Args:
            round_num: FL round number
            aggregated_model: The aggregated model weights (from FedAvg on committee)
            client_models: Optional dict of {client_id: model_weights} for validators
        
        Returns:
            (consensus_reached, consensus_details)
            consensus_details contains:
                - reached: bool
                - votes: {validator_id: vote}
                - approve_count: int
                - reject_count: int
                - quorum_size: int
                - phase: str
                - primary_id: int
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"PBFT CONSENSUS - Round {round_num}")
            print(f"{'='*70}")
            print(f"Committee: {self.committee}")
            print(f"Committee size: {self.committee_size}")
            print(f"Primary (highest DAVS score): Node {self.primary_id}")
            print(f"Byzantine tolerance: f={self.f}")
            print(f"Quorum required: {self.quorum_size}/{self.committee_size}")
        
        # Phase 1: PRE-PREPARE
        model_hash = self._hash_model(aggregated_model)
        
        if self.verbose:
            print(f"\n--- Phase 1: PRE-PREPARE ---")
            print(f"Primary {self.primary_id} broadcasts model (hash: {model_hash[:16]}...)")
            print(f"Message size: ~{self._estimate_model_size(aggregated_model):.2f} MB")
            print(f"Recipients: {len(self.committee)} validators")
        
        # Create validators for each committee member
        validators = {}
        for node_id in self.committee:
            is_primary = (node_id == self.primary_id)
            validators[node_id] = PBFTValidator(
                node_id=node_id,
                model_weights=aggregated_model,
                is_primary=is_primary
            )
        
        # Phase 2: PREPARE
        if self.verbose:
            print(f"\n--- Phase 2: PREPARE ---")
        
        prepare_votes = {}
        for node_id, validator in validators.items():
            vote = validator.vote(aggregated_model)
            prepare_votes[node_id] = vote
            
            if self.verbose:
                vote_str = "✅ APPROVE" if vote else "❌ REJECT"
                print(f"Validator {node_id}: {vote_str}")
        
        approve_count = sum(1 for v in prepare_votes.values() if v)
        reject_count = sum(1 for v in prepare_votes.values() if not v)
        
        # Check if quorum reached in PREPARE phase
        prepare_quorum = approve_count >= self.quorum_size
        
        if self.verbose:
            print(f"\nPREPARE Phase Results:")
            print(f"  Approvals: {approve_count}/{self.committee_size}")
            print(f"  Rejections: {reject_count}/{self.committee_size}")
            print(f"  Quorum ({self.quorum_size}): {'✅ REACHED' if prepare_quorum else '❌ NOT REACHED'}")
        
        if not prepare_quorum:
            if self.verbose:
                print(f"\n❌ Consensus FAILED - Insufficient approvals in PREPARE phase")
            
            return False, {
                'reached': False,
                'votes': prepare_votes,
                'approve_count': approve_count,
                'reject_count': reject_count,
                'quorum_size': self.quorum_size,
                'phase': 'prepare',
                'primary_id': self.primary_id,
                'reason': 'prepare_quorum_not_reached'
            }
        
        # Phase 3: COMMIT
        if self.verbose:
            print(f"\n--- Phase 3: COMMIT ---")
            print(f"Quorum reached in PREPARE, validators broadcast COMMIT messages")
        
        # Simulate commit messages (in real system, nodes would broadcast)
        commit_votes = prepare_votes.copy()  # Same votes in commit phase
        commit_quorum = approve_count >= self.quorum_size
        
        if self.verbose:
            print(f"COMMIT messages exchanged: {self.committee_size} × {self.committee_size-1} = {self.committee_size * (self.committee_size-1)}")
            print(f"Quorum ({self.quorum_size}): {'✅ REACHED' if commit_quorum else '❌ NOT REACHED'}")
        
        # Consensus reached if commit quorum achieved
        consensus_reached = commit_quorum
        
        if self.verbose:
            if consensus_reached:
                print(f"\n✅ CONSENSUS REACHED!")
                print(f"Model update APPROVED by {approve_count}/{self.committee_size} validators")
            else:
                print(f"\n❌ CONSENSUS FAILED")
        
        # Calculate communication complexity
        # PRE-PREPARE: 1 primary → k-1 replicas = k-1 messages
        # PREPARE: k replicas × (k-1) = k(k-1) messages
        # COMMIT: k replicas × (k-1) = k(k-1) messages
        # Total: k-1 + 2k(k-1) = k-1 + 2k² - 2k = 2k² - k - 1 ≈ O(k²)
        total_messages = (self.committee_size - 1) + 2 * self.committee_size * (self.committee_size - 1)
        
        if self.verbose:
            print(f"\nCommunication Complexity:")
            print(f"  Total messages: {total_messages} (O(k²) where k={self.committee_size})")
            print(f"  vs Full PBFT (all nodes): O(N²) messages would be ~{len(self.committee)**2 * 3}")
        
        consensus_details = {
            'reached': consensus_reached,
            'votes': prepare_votes,
            'approve_count': approve_count,
            'reject_count': reject_count,
            'quorum_size': self.quorum_size,
            'phase': 'commit' if consensus_reached else 'prepare',
            'primary_id': self.primary_id,
            'total_messages': total_messages,
            'committee_size': self.committee_size
        }
        
        return consensus_reached, consensus_details
    
    def _hash_model(self, model_weights: Dict[str, torch.Tensor]) -> str:
        """
        Compute hash of model weights
        
        Args:
            model_weights: Model parameters
        
        Returns:
            SHA256 hash
        """
        weight_bytes = b''
        for key in sorted(model_weights.keys()):
            tensor = model_weights[key]
            weight_bytes += tensor.cpu().numpy().tobytes()
        
        return hashlib.sha256(weight_bytes).hexdigest()
    
    def _estimate_model_size(self, model_weights: Dict[str, torch.Tensor]) -> float:
        """
        Estimate model size in MB
        
        Args:
            model_weights: Model parameters
        
        Returns:
            Size in megabytes
        """
        total_bytes = 0
        for param in model_weights.values():
            total_bytes += param.nelement() * param.element_size()
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def __repr__(self) -> str:
        return f"PBFTConsensus(committee_size={self.committee_size}, f={self.f}, quorum={self.quorum_size})"
