"""
Phase 2+3 Training Script
Federated Learning with Gradient Sketching and DAVS Committee Selection
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
from federated.client import create_clients
from federated.server import FederatedServer
from federated.gradient_sketching import GradientSketcherForDAVS
from federated.davs_selection import DAVSSelector
from utils.metrics import MetricsLogger, plot_client_contributions


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
    """Main training with gradient sketching and DAVS"""
    
    print("="*70)
    print("FL+DAVS - Phase 2+3: Gradient Sketching + DAVS Selection")
    print("="*70)
    
    # Setup
    setup_directories()
    
    # Configuration
    USE_GRADIENT_SKETCHING = True  # Use Random Projection for DAVS
    SKETCH_DIM = 128  # Sketch dimension for random projection
    USE_DAVS = True
    COMMITTEE_SIZE = max(int(NUM_CLIENTS * 0.7), 5)  # 70% of clients
    
    # Initialize metrics logger
    experiment_name = f"davs_sketch_{DATASET}_{DATA_SPLIT}"
    logger = MetricsLogger(experiment_name=experiment_name, save_dir=RESULTS_DIR)
    
    config = {
        'dataset': DATASET,
        'num_clients': NUM_CLIENTS,
        'data_split': DATA_SPLIT,
        'local_epochs': LOCAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_rounds': NUM_ROUNDS,
        'device': DEVICE,
        'model': 'SimpleCNN',
        'gradient_sketching': USE_GRADIENT_SKETCHING,
        'sketch_dim': SKETCH_DIM if USE_GRADIENT_SKETCHING else None,
        'davs_selection': USE_DAVS,
        'committee_size': COMMITTEE_SIZE if USE_DAVS else NUM_CLIENTS
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
    
    # Get class distributions for DAVS
    class_distributions = get_class_distributions(client_datasets, num_classes=data_loader.n_classes)
    data_sizes = [len(client_datasets[i]) for i in range(NUM_CLIENTS)]
    
    # Plot data distribution
    client_data_sizes = {f"Client {i}": len(client_datasets[i]) for i in range(NUM_CLIENTS)}
    plot_client_contributions(
        client_data_sizes,
        os.path.join(logger.exp_dir, 'data_distribution.png')
    )
    print(f"✓ Data distribution plotted")
    
    # Initialize components
    print(f"\n{'='*70}")
    print("Initializing FL Components")
    print(f"{'='*70}")
    
    clients = create_clients(client_loaders, get_model, device=DEVICE)
    server = FederatedServer(get_model(), device=DEVICE)
    
    num_params = sum(p.numel() for p in server.global_model.parameters())
    print(f"✓ Created {len(clients)} federated clients")
    print(f"✓ Initialized global model ({num_params:,} parameters)")
    
    # Initialize gradient sketcher for DAVS
    gradient_sketcher = None
    if USE_GRADIENT_SKETCHING:
        gradient_sketcher = GradientSketcherForDAVS(
            model=server.global_model,
            sketch_dim=SKETCH_DIM,
            add_dp_noise=False,
            shared_seed=42  # All clients must use same seed!
        )
        print(f"\n✓ Random Projection Sketcher initialized")
        print(f"  Sketch dimension: {SKETCH_DIM}")
        print(f"  Compression: {gradient_sketcher.total_params/SKETCH_DIM:.1f}x")
        print(f"  Bandwidth reduction: {gradient_sketcher.sketcher.bandwidth_reduction:.1f}%")
    
    # Initialize DAVS selector
    if USE_DAVS:
        davs_selector = DAVSSelector(committee_size=COMMITTEE_SIZE)
        print(f"\n✓ DAVS Selector initialized")
        print(f"  Committee size: {davs_selector.committee_size} clients")
        print(f"  Algorithm: Amnesic (no historical tracking)")
    else:
        davs_selector = None
    
    # Initial evaluation
    print(f"\n{'='*70}")
    print("Initial Model Evaluation")
    print(f"{'='*70}")
    init_loss, init_acc = server.evaluate(test_loader)
    logger.log_round(0, init_loss, init_acc, init_loss, init_acc)
    
    # Track previous losses for improvement calculation
    prev_client_losses = [init_loss] * NUM_CLIENTS
    
    # Federated training with DAVS and Gradient Sketching
    print(f"\n{'='*70}")
    print("Starting Federated Training (DAVS + Gradient Sketching)")
    print(f"{'='*70}")
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\nRound {round_num}/{NUM_ROUNDS}")
        print("-" * 70)
        
        # Get global parameters
        global_params = server.get_global_parameters()
        
        # Collect gradients from all clients
        client_gradients = []
        client_losses = []
        client_accuracies = []
        
        for client_id in range(NUM_CLIENTS):
            client = clients[client_id]
            
            # Set global parameters
            client.set_parameters(global_params)
            
            # Local training
            num_samples, loss, acc = client.train(epochs=LOCAL_EPOCHS, learning_rate=LEARNING_RATE)
            
            # Get gradients (updates)
            gradients = client.compute_gradients(global_params)
            client_gradients.append(gradients)
            client_losses.append(loss)
            client_accuracies.append(acc)
        
        # DAVS Committee Selection using Random Projection
        if USE_DAVS and gradient_sketcher:
            # Sketch all client gradients
            client_sketches = {}
            for client_id, gradients in enumerate(client_gradients):
                # Convert gradient dict to tensor for sketching
                grad_list = [gradients[name] for name in sorted(gradients.keys())]
                grad_tensor = torch.cat([g.flatten() for g in grad_list])
                sketch = gradient_sketcher.sketcher.sketch(grad_tensor)
                client_sketches[client_id] = sketch
            
            # Select committee based on representativeness (DAVS way)
            committee, representativeness_scores = davs_selector.select_committee(client_sketches)
            
            print(f"  Selected committee ({len(committee)}): {committee}")
            print(f"  Representativeness: {[f'{representativeness_scores[i]:.3f}' for i in committee[:3]]}")
            
            # Filter to committee members
            selected_gradients = [client_gradients[i] for i in committee]
            selected_weights = [data_sizes[i] for i in committee]
        else:
            # Use all clients (no DAVS)
            committee = list(range(NUM_CLIENTS))
            selected_gradients = client_gradients
            selected_weights = data_sizes
        
        # Aggregate selected gradients (standard FedAvg)
        from federated.aggregation import fedavg
        # Convert gradients to parameters format for fedavg
        # compute_gradients already returns (local_params - global_params)
        selected_params = []
        for grad_dict in selected_gradients:
            params = {}
            for name, update in grad_dict.items():
                # Apply the update: new_params = global_params + update
                params[name] = global_params[name] + update
            selected_params.append(params)
        
        aggregated_params = fedavg(selected_params, selected_weights)
        
        # Update global model
        server.set_global_parameters(aggregated_params)
        
        # Calculate training metrics
        train_loss = np.average(client_losses, weights=data_sizes)
        train_acc = np.average(client_accuracies, weights=data_sizes)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluate on test set
        if round_num % 5 == 0 or round_num == NUM_ROUNDS:
            test_loss, test_acc = server.evaluate(test_loader)
            logger.log_round(round_num, train_loss, train_acc, test_loss, test_acc)
        else:
            logger.log_round(round_num, train_loss, train_acc)
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Model Evaluation")
    print(f"{'='*70}")
    final_test_loss, final_test_acc = server.evaluate(test_loader)
    
    # Save results
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")
    
    model_path = os.path.join(logger.exp_dir, 'final_model.pth')
    server.save_model(model_path)
    
    logger.save_json()
    logger.plot_training_curves()
    logger.save_summary(config)
    
    # Summary (no DAVS statistics since it's amnesic)
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Results:")
    print(f"  Test Loss: {final_test_loss:.4f}")
    print(f"  Test Accuracy: {final_test_acc:.2f}%")
    if USE_GRADIENT_SKETCHING and gradient_sketcher:
        bandwidth_reduction = gradient_sketcher.sketcher.bandwidth_reduction
        print(f"  Bandwidth Reduction: {bandwidth_reduction:.1f}%")
    if USE_DAVS:
        print(f"  Committee Size: {COMMITTEE_SIZE}/{NUM_CLIENTS} clients")
        print(f"  Selection: Amnesic (gradient similarity only)")
    print(f"\nAll results saved to: {logger.exp_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
