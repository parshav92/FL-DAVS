"""
Phase 4: DAVS + PBFT + Blockchain
Complete system with gradient sketching, DAVS selection, PBFT consensus, and blockchain
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from models.cnn_model import get_model
from data.medmnist_loader import MedMNISTDataLoader, create_client_loaders
from federated.server import FederatedServer
from federated.gradient_sketching import GradientSketcherForDAVS
from federated.davs_selection import DAVSSelector
from federated.aggregation import fedavg
from blockchain.chain import MedicalBlockchain
from consensus.pbft import PBFTConsensus
from attacks.malicious_client import create_mixed_clients, AttackType
from utils.attack_logger import AttackLogger
from utils.metrics import plot_client_contributions


def setup_directories():
    """Create necessary directories"""
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def get_class_distributions(client_datasets, num_classes=9):
    """Calculate class distribution for each client"""
    distributions = {}
    for client_id, dataset in client_datasets.items():
        class_counts = np.zeros(num_classes)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            class_counts[label.item()] += 1
        distributions[client_id] = class_counts
    return distributions


def main():
    """Main training with DAVS + PBFT + Blockchain"""
    
    print("="*70)
    print("MedBlockDFL - DAVS + PBFT + Blockchain")
    print("="*70)
    
    # Setup
    setup_directories()
    
    # Configuration
    USE_GRADIENT_SKETCHING = True
    SKETCH_DIM = 128
    USE_DAVS = True
    COMMITTEE_SIZE = 5  # Fixed for PBFT
    USE_PBFT = True
    USE_BLOCKCHAIN = True
    
    # Attack configuration
    MALICIOUS_RATIO = 0.2  # 20% malicious nodes
    NUM_MALICIOUS = max(1, int(NUM_CLIENTS * MALICIOUS_RATIO))
    MALICIOUS_IDS = list(range(NUM_CLIENTS - NUM_MALICIOUS, NUM_CLIENTS))  # Last N clients are malicious
    ATTACK_TYPE = AttackType.FLIP
    ATTACK_SCALE = 10.0
    
    # Initialize logger
    experiment_name = f"hybrid_davs_pbft_blockchain_{DATASET}_{DATA_SPLIT}_attack{int(MALICIOUS_RATIO*100)}"
    logger = AttackLogger(
        experiment_name=experiment_name,
        save_dir=RESULTS_DIR,
        malicious_nodes=MALICIOUS_IDS,
        total_nodes=NUM_CLIENTS
    )
    
    config = {
        'dataset': DATASET,
        'num_clients': NUM_CLIENTS,
        'malicious_clients': NUM_MALICIOUS,
        'malicious_ratio': MALICIOUS_RATIO,
        'malicious_ids': MALICIOUS_IDS,
        'attack_type': ATTACK_TYPE.value,
        'attack_scale': ATTACK_SCALE,
        'data_split': DATA_SPLIT,
        'local_epochs': LOCAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_rounds': NUM_ROUNDS,
        'device': DEVICE,
        'model': 'SimpleCNN',
        'gradient_sketching': USE_GRADIENT_SKETCHING,
        'sketch_dim': SKETCH_DIM,
        'davs_selection': USE_DAVS,
        'committee_size': COMMITTEE_SIZE,
        'pbft_consensus': USE_PBFT,
        'blockchain': USE_BLOCKCHAIN
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load and partition data
    print(f"\n{'='*70}")
    print("Loading and Partitioning Data")
    print(f"{'='*70}")
    
    data_loader = MedMNISTDataLoader(
        dataset_name=DATASET,
        num_clients=NUM_CLIENTS,
        data_dir=DATA_DIR
    )
    
    if DATA_SPLIT == 'iid':
        client_datasets = data_loader.partition_iid()
    else:
        client_datasets = data_loader.partition_non_iid(alpha=0.5)
    
    client_loaders = create_client_loaders(client_datasets, batch_size=BATCH_SIZE)
    test_loader = data_loader.get_test_loader(batch_size=BATCH_SIZE)
    
    data_sizes = [len(client_datasets[i]) for i in range(NUM_CLIENTS)]
    
    # Plot data distribution
    client_data_sizes = {f"Client {i}": len(client_datasets[i]) for i in range(NUM_CLIENTS)}
    plot_client_contributions(
        client_data_sizes,
        os.path.join(logger.exp_dir, 'data_distribution.png')
    )
    
    # Initialize components
    print(f"\n{'='*70}")
    print("Initializing FL Components")
    print(f"{'='*70}")
    
    # Create mixed clients (honest + malicious)
    clients = create_mixed_clients(
        client_datasets=client_loaders,
        model_fn=get_model,
        malicious_ids=MALICIOUS_IDS,
        attack_type=ATTACK_TYPE,
        attack_scale=ATTACK_SCALE,
        device=DEVICE
    )
    
    server = FederatedServer(get_model(), device=DEVICE)
    
    num_params = sum(p.numel() for p in server.global_model.parameters())
    print(f"✓ Created {len(clients)} federated clients")
    print(f"  Honest: {NUM_CLIENTS - NUM_MALICIOUS}")
    print(f"  Malicious: {NUM_MALICIOUS} (IDs: {MALICIOUS_IDS})")
    print(f"  Attack type: {ATTACK_TYPE.value}, scale: {ATTACK_SCALE}")
    print(f"✓ Initialized global model ({num_params:,} parameters)")
    
    # Initialize gradient sketcher
    gradient_sketcher = GradientSketcherForDAVS(
        model=server.global_model,
        sketch_dim=SKETCH_DIM,
        add_dp_noise=False,
        shared_seed=42
    )
    print(f"\n✓ Random Projection Sketcher initialized")
    print(f"  Sketch dimension: {SKETCH_DIM}")
    print(f"  Compression: {gradient_sketcher.total_params/SKETCH_DIM:.1f}x")
    
    # Initialize DAVS selector
    davs_selector = DAVSSelector(committee_size=COMMITTEE_SIZE)
    print(f"\n✓ DAVS Selector initialized")
    print(f"  Committee size: {COMMITTEE_SIZE} clients")
    
    # Initialize blockchain
    if USE_BLOCKCHAIN:
        blockchain = MedicalBlockchain()
        print(f"\n✓ Blockchain initialized")
        print(f"  Genesis block created: {blockchain.get_latest_block()}")
    
    # Initial evaluation
    print(f"\n{'='*70}")
    print("Initial Model Evaluation")
    print(f"{'='*70}")
    init_loss, init_acc = server.evaluate(test_loader)
    print(f"Initial Test Loss: {init_loss:.4f}, Test Acc: {init_acc:.2f}%")
    
    # Federated training with DAVS + PBFT + Blockchain
    print(f"\n{'='*70}")
    print("Starting Federated Training (DAVS + PBFT + Blockchain)")
    print(f"{'='*70}")
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"Round {round_num}/{NUM_ROUNDS}")
        print(f"{'='*70}")
        
        # Get global parameters
        global_params = server.get_global_parameters()
        
        # Phase 1: Local Training (ALL clients)
        print(f"\n--- Phase 1: Local Training ({NUM_CLIENTS} clients) ---")
        client_gradients = []
        client_losses = []
        client_accuracies = []
        
        for client_id in range(NUM_CLIENTS):
            client = clients[client_id]
            client.set_parameters(global_params)
            num_samples, loss, acc = client.train(epochs=LOCAL_EPOCHS, learning_rate=LEARNING_RATE)
            gradients = client.compute_gradients(global_params)
            client_gradients.append(gradients)
            client_losses.append(loss)
            client_accuracies.append(acc)
        
        print(f"✓ All clients completed local training")
        
        # Phase 2: Gradient Sketching (ALL clients)
        print(f"\n--- Phase 2: Gradient Sketching & Norm Calculation ---")
        client_sketches = {}
        client_grad_norms = {}
        for client_id, gradients in enumerate(client_gradients):
            grad_list = [gradients[name] for name in sorted(gradients.keys())]
            grad_tensor = torch.cat([g.flatten() for g in grad_list])
            
            # NEW: Calculate and store the L2 norm of the full gradient
            client_grad_norms[client_id] = torch.norm(grad_tensor).item()
            
            sketch = gradient_sketcher.sketcher.sketch(grad_tensor)
            client_sketches[client_id] = sketch
        
        print(f"✓ All clients computed sketches ({SKETCH_DIM}-dim) and norms")
        print(f"  Bandwidth: {len(client_sketches)} × {SKETCH_DIM} × 4 bytes = {len(client_sketches)*SKETCH_DIM*4/1024:.2f} KB")
        
        # Phase 3: DAVS Committee Selection
        print(f"\n--- Phase 3: Hybrid DAVS Committee Selection ---")
        committee, representativeness_scores = davs_selector.select_committee(
            client_sketches, client_grad_norms
        )
        
        print(f"✓ Committee selected: {committee}")
        print(f"  Hybrid DAVS scores (top-5):")
        for i in committee[:5]:
            malicious_tag = " 🔴 MALICIOUS" if i in MALICIOUS_IDS else " ✅ HONEST"
            print(f"    Node {i}: {representativeness_scores[i]:.4f}{malicious_tag}")
        
        # Check if malicious nodes were selected
        malicious_in_committee = [nid for nid in committee if nid in MALICIOUS_IDS]
        if malicious_in_committee:
            print(f"  ⚠️  WARNING: {len(malicious_in_committee)} malicious node(s) in committee: {malicious_in_committee}")
        else:
            print(f"  ✅ No malicious nodes selected!")
        
        # Phase 4: Aggregation (Committee only)
        print(f"\n--- Phase 4: FedAvg Aggregation (Committee only) ---")
        selected_gradients = [client_gradients[i] for i in committee]
        selected_weights = [data_sizes[i] for i in committee]
        
        # Convert gradients to parameters
        selected_params = []
        for grad_dict in selected_gradients:
            params = {}
            for name, update in grad_dict.items():
                params[name] = global_params[name] + update
            selected_params.append(params)
        
        aggregated_params = fedavg(selected_params, selected_weights)
        print(f"✓ Aggregated updates from {len(committee)} committee members")
        
        # Phase 5: PBFT Consensus
        if USE_PBFT:
            pbft = PBFTConsensus(
                committee=committee,
                max_faulty=None,  # Auto-calculate f
                verbose=True
            )
            
            consensus_reached, consensus_result = pbft.run_consensus(
                round_num=round_num,
                aggregated_model=aggregated_params
            )
            
            if consensus_reached:
                # Update global model with validated parameters
                server.set_global_parameters(aggregated_params)
            else:
                print(f"\n⚠️  Consensus failed - keeping previous model")
                consensus_result['model_updated'] = False
        else:
            # No PBFT - directly update
            server.set_global_parameters(aggregated_params)
            consensus_reached = True
            consensus_result = {
                'reached': True,
                'votes': {nid: True for nid in committee},
                'approve_count': len(committee),
                'quorum_size': len(committee),
                'phase': 'no_pbft'
            }
        
        # Phase 6: Blockchain Commitment
        if USE_BLOCKCHAIN and consensus_reached:
            print(f"\n--- Phase 6: Blockchain Commitment ---")
            
            # Evaluate model
            train_loss = np.average(client_losses, weights=data_sizes)
            train_acc = np.average(client_accuracies, weights=data_sizes)
            
            if round_num % 5 == 0 or round_num == NUM_ROUNDS:
                test_loss, test_acc = server.evaluate(test_loader)
            else:
                test_loss, test_acc = None, None
            
            # Add block to blockchain
            block = blockchain.add_block(
                round_num=round_num,
                model_weights=aggregated_params,
                committee=committee,
                davs_scores=representativeness_scores,
                consensus_result=consensus_result,
                round_metrics={
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss if test_loss else 0.0,
                    'test_acc': test_acc if test_acc else 0.0
                },
                attack_info={
                    'malicious_in_committee': len(malicious_in_committee),
                    'malicious_ids': malicious_in_committee
                } if malicious_in_committee else None
            )
            
            print(f"✓ Block #{round_num} added to blockchain")
            print(f"  Hash: {block.hash[:16]}...")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            if test_acc:
                print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # Log to attack logger
            logger.log_round(
                round_num=round_num,
                davs_scores=representativeness_scores,
                grad_norms=client_grad_norms,  # Pass norms to logger
                committee=committee,
                consensus_result=consensus_result,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                blockchain_hash=block.hash
            )
        
        # Verify blockchain integrity
        if USE_BLOCKCHAIN and round_num % 10 == 0:
            is_valid = blockchain.verify_chain()
            print(f"\n🔗 Blockchain verification: {'✅ VALID' if is_valid else '❌ INVALID'}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Model Evaluation")
    print(f"{'='*70}")
    final_test_loss, final_test_acc = server.evaluate(test_loader)
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    
    # Save results
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")
    
    # Save model
    model_path = os.path.join(logger.exp_dir, 'final_model.pth')
    server.save_model(model_path)
    
    # Save blockchain
    if USE_BLOCKCHAIN:
        blockchain_path = os.path.join(logger.exp_dir, 'blockchain.json')
        blockchain.export_to_json(blockchain_path)
        print(f"✓ Blockchain saved ({len(blockchain)} blocks)")
        
        # Blockchain summary
        summary = blockchain.get_chain_summary()
        print(f"  Chain valid: {summary['chain_valid']}")
    
    # Save attack logs and generate plots
    logger.save_all()
    
    # Save config
    import json
    config_path = os.path.join(logger.exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"All results saved to: {logger.exp_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
