"""
Blockchain Implementation for Federated Learning
Stores model updates, DAVS scores, and consensus results
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional
import torch


class Block:
    """
    Block in the blockchain containing FL round information
    """
    
    def __init__(
        self,
        index: int,
        timestamp: float,
        data: Dict[str, Any],
        previous_hash: str,
        nonce: int = 0
    ):
        """
        Args:
            index: Block number (round number)
            timestamp: Time when block was created
            data: Dictionary containing:
                - model_hash: Hash of aggregated model weights
                - committee: List of validator node IDs
                - davs_scores: Dict of {node_id: representativeness_score} for ALL nodes
                - consensus: Dict with PBFT consensus details
                - round_metrics: Training/test accuracy and loss
                - attack_info: Information about detected attacks (if any)
            previous_hash: Hash of previous block
            nonce: Proof of work nonce (optional, not used in our PoC)
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """
        Calculate SHA256 hash of block contents
        
        Returns:
            Hexadecimal hash string
        """
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self._serialize_data(self.data),
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize data for hashing (convert non-serializable objects)
        """
        serialized = {}
        for key, value in data.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to dictionary for JSON serialization
        """
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'hash': self.hash,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'data': self.data
        }
    
    def __repr__(self) -> str:
        return f"Block(index={self.index}, hash={self.hash[:8]}..., prev={self.previous_hash[:8]}...)"


class MedicalBlockchain:
    """
    Blockchain for storing federated learning model updates
    """
    
    def __init__(self):
        """
        Initialize blockchain with genesis block
        """
        self.chain: List[Block] = []
        self.create_genesis_block()
    
    def create_genesis_block(self) -> Block:
        """
        Create the first block in the chain
        
        Returns:
            Genesis block
        """
        genesis_data = {
            'model_hash': '0' * 64,
            'committee': [],
            'davs_scores': {},
            'consensus': {
                'reached': True,
                'votes': {},
                'quorum_size': 0
            },
            'round_metrics': {
                'train_loss': 0.0,
                'train_acc': 0.0,
                'test_loss': 0.0,
                'test_acc': 0.0
            },
            'message': 'Genesis Block - MedBlockDFL Initialized'
        }
        
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            data=genesis_data,
            previous_hash='0' * 64,
            nonce=0
        )
        
        self.chain.append(genesis_block)
        return genesis_block
    
    def get_latest_block(self) -> Block:
        """
        Get the most recent block in the chain
        
        Returns:
            Latest block
        """
        return self.chain[-1]
    
    def add_block(
        self,
        round_num: int,
        model_weights: Dict[str, torch.Tensor],
        committee: List[int],
        davs_scores: Dict[int, float],
        consensus_result: Dict[str, Any],
        round_metrics: Dict[str, float],
        attack_info: Optional[Dict[str, Any]] = None
    ) -> Block:
        """
        Add a new block to the chain after successful consensus
        
        Args:
            round_num: FL round number
            model_weights: Aggregated model parameters
            committee: List of validator node IDs
            davs_scores: Representativeness scores for ALL nodes
            consensus_result: PBFT consensus details (votes, quorum, etc.)
            round_metrics: Training and test metrics
            attack_info: Optional information about detected attacks
        
        Returns:
            The newly created block
        """
        # Compute model hash
        model_hash = self._hash_model_weights(model_weights)
        
        # Prepare block data
        block_data = {
            'model_hash': model_hash,
            'committee': committee,
            'davs_scores': {str(k): float(v) for k, v in davs_scores.items()},
            'consensus': consensus_result,
            'round_metrics': round_metrics,
        }
        
        if attack_info:
            block_data['attack_info'] = attack_info
        
        # Create new block
        previous_block = self.get_latest_block()
        new_block = Block(
            index=round_num,
            timestamp=time.time(),
            data=block_data,
            previous_hash=previous_block.hash,
            nonce=0
        )
        
        # Add to chain
        self.chain.append(new_block)
        
        return new_block
    
    def _hash_model_weights(self, model_weights: Dict[str, torch.Tensor]) -> str:
        """
        Compute SHA256 hash of model weights
        
        Args:
            model_weights: Dictionary of model parameters
        
        Returns:
            Hexadecimal hash string
        """
        # Concatenate all parameters and hash
        weight_bytes = b''
        for key in sorted(model_weights.keys()):
            tensor = model_weights[key]
            weight_bytes += tensor.cpu().numpy().tobytes()
        
        return hashlib.sha256(weight_bytes).hexdigest()
    
    def verify_chain(self) -> bool:
        """
        Verify integrity of the entire blockchain
        
        Returns:
            True if chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Verify hash
            if current_block.hash != current_block.calculate_hash():
                print(f"❌ Block {i} hash mismatch!")
                return False
            
            # Verify chain link
            if current_block.previous_hash != previous_block.hash:
                print(f"❌ Block {i} previous_hash doesn't match!")
                return False
        
        return True
    
    def get_audit_trail(self, node_id: int) -> List[Dict[str, Any]]:
        """
        Get audit trail for a specific node (DAVS scores over time)
        
        Args:
            node_id: Node to track
        
        Returns:
            List of {round, davs_score, in_committee} for each round
        """
        audit_trail = []
        
        for block in self.chain[1:]:  # Skip genesis
            davs_scores = block.data.get('davs_scores', {})
            committee = block.data.get('committee', [])
            
            score = davs_scores.get(str(node_id), None)
            if score is not None:
                audit_trail.append({
                    'round': block.index,
                    'davs_score': score,
                    'in_committee': node_id in committee,
                    'timestamp': block.timestamp
                })
        
        return audit_trail
    
    def get_committee_history(self) -> List[Dict[str, Any]]:
        """
        Get history of committee selections
        
        Returns:
            List of {round, committee, committee_size} for each round
        """
        history = []
        
        for block in self.chain[1:]:  # Skip genesis
            committee = block.data.get('committee', [])
            history.append({
                'round': block.index,
                'committee': committee,
                'committee_size': len(committee)
            })
        
        return history
    
    def export_to_json(self, filepath: str):
        """
        Export blockchain to JSON file
        
        Args:
            filepath: Path to save JSON file
        """
        chain_data = [block.to_dict() for block in self.chain]
        
        with open(filepath, 'w') as f:
            json.dump(chain_data, f, indent=2)
    
    def get_chain_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the blockchain
        
        Returns:
            Dictionary with chain statistics
        """
        return {
            'total_blocks': len(self.chain),
            'chain_valid': self.verify_chain(),
            'latest_block_hash': self.get_latest_block().hash,
            'genesis_timestamp': self.chain[0].timestamp,
            'latest_timestamp': self.get_latest_block().timestamp
        }
    
    def __len__(self) -> int:
        return len(self.chain)
    
    def __repr__(self) -> str:
        return f"MedicalBlockchain(blocks={len(self.chain)}, valid={self.verify_chain()})"
