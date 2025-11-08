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
from federated.gradient_sketching import GradientCompressor
from federated.davs_selection import DAVSSelector, DataQualityMetrics
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
    USE_GRADIENT_SKETCHING = True
    COMPRESSION_RATE = 0.1  # 10x compression
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
        'compression_rate': COMPRESSION_RATE if USE_GRADIENT_SKETCHING else 1.0,
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
    
    # Initialize gradient compressor
    if USE_GRADIENT_SKETCHING:
        compressor = GradientCompressor(
            server.global_model,
            compression_rate=COMPRESSION_RATE,
            num_hash=3
        )
        stats = compressor.get_compression_stats()
        print(f"\n✓ Gradient Compressor initialized")
        print(f"  Compression rate: {stats['compression_rate']:.2%}")
        print(f"  Bandwidth reduction: {stats['bandwidth_reduction_percent']:.1f}%")
        print(f"  Original: {stats['original_size']:,} → Compressed: {stats['compressed_size']:,}")
    else:
        compressor = None
    
    # Initialize DAVS selector
    if USE_DAVS:
        davs_selector = DAVSSelector(
            num_clients=NUM_CLIENTS,
            committee_size=COMMITTEE_SIZE,
            selection_strategy='weighted'
        )
        print(f"\n✓ DAVS Selector initialized")
        print(f"  Committee size: {davs_selector.committee_size}/{NUM_CLIENTS} clients")
        print(f"  Selection strategy: {davs_selector.selection_strategy}")
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
        
        for client_id, client in enumerate(clients):
            # Set global parameters
            client.set_parameters(global_params)
            
            # Local training
            loss, acc, num_samples = client.train(epochs=LOCAL_EPOCHS, learning_rate=LEARNING_RATE)
            
            # Get gradients (updates)
            gradients = client.compute_gradients(global_params)
            client_gradients.append(gradients)
            client_losses.append(loss)
            client_accuracies.append(acc)
        
        # Calculate loss improvements
        loss_improvements = [
            DataQualityMetrics.calculate_loss_improvement(prev_loss, curr_loss)
            for prev_loss, curr_loss in zip(prev_client_losses, client_losses)
        ]
        prev_client_losses = client_losses
        
        # DAVS Committee Selection
        if USE_DAVS:
            committee = davs_selector.select_committee(
                client_gradients=client_gradients,
                data_sizes=data_sizes,
                loss_improvements=loss_improvements,
                class_distributions=class_distributions
            )
            
            # Detect Byzantine clients
            byzantine = davs_selector.detect_byzantine(client_gradients, threshold=2.5)
            if byzantine:
                print(f"  ⚠️  Byzantine clients detected: {byzantine}")
            
            print(f"  Selected committee ({len(committee)}): {committee}")
            
            # Filter to committee members
            selected_gradients = [client_gradients[i] for i in committee]
            selected_weights = [data_sizes[i] for i in committee]
        else:
            # Use all clients
            committee = list(range(NUM_CLIENTS))
            selected_gradients = client_gradients
            selected_weights = data_sizes
        
        # Gradient Sketching (if enabled)
        if USE_GRADIENT_SKETCHING and compressor:
            # Compress gradients
            compressed_gradients = []
            for grad in selected_gradients:
                compressed = compressor.compress_gradients(grad)
                compressed_gradients.append(compressed)
            
            # Decompress and aggregate
            from federated.gradient_sketching import decompress_and_aggregate
            aggregated_gradients = decompress_and_aggregate(
                compressed_gradients,
                selected_weights,
                compressor
            )
        else:
            # Standard aggregation without compression
            from federated.aggregation import fedavg
            # Convert gradients to parameters format for fedavg
            selected_params = []
            for grad_dict in selected_gradients:
                params = {}
                for name, grad in grad_dict.items():
                    # Gradient as parameter update (subtract from current)
                    params[name] = global_params[name] - grad * LEARNING_RATE
                selected_params.append(params)
            
            aggregated_gradients = fedavg(selected_params, selected_weights)
        
        # Update global model
        server.set_global_parameters(aggregated_gradients)
        
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
        
        # Update DAVS history
        if USE_DAVS:
            for client_id in range(NUM_CLIENTS):
                grad_norm = DataQualityMetrics.calculate_gradient_norm(client_gradients[client_id])
                was_selected = client_id in committee
                davs_selector.update_client_history(
                    client_id, grad_norm, loss_improvements[client_id], was_selected
                )
    
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
    
    # Save DAVS statistics
    if USE_DAVS:
        davs_stats_path = os.path.join(logger.exp_dir, 'davs_stats.txt')
        with open(davs_stats_path, 'w') as f:
            f.write("DAVS Committee Selection Statistics\n")
            f.write("="*60 + "\n\n")
            f.write(f"Committee Size: {COMMITTEE_SIZE}/{NUM_CLIENTS}\n")
            f.write(f"Selection Strategy: {davs_selector.selection_strategy}\n")
            f.write(f"Byzantine Clients Detected: {len(davs_selector.suspected_byzantine)}\n")
            if davs_selector.suspected_byzantine:
                f.write(f"Byzantine IDs: {list(davs_selector.suspected_byzantine)}\n")
            f.write("\nClient Contribution History:\n")
            f.write("-"*60 + "\n")
            for client_id in range(NUM_CLIENTS):
                history = davs_selector.client_history[client_id]
                f.write(f"Client {client_id}:\n")
                f.write(f"  Contributions: {history['contributions']}\n")
                f.write(f"  Reliability Score: {history['reliability_score']:.3f}\n")
                f.write(f"  Avg Gradient Norm: {history['avg_gradient_norm']:.4f}\n\n")
        print(f"✓ DAVS statistics saved to {davs_stats_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Results:")
    print(f"  Test Loss: {final_test_loss:.4f}")
    print(f"  Test Accuracy: {final_test_acc:.2f}%")
    if USE_GRADIENT_SKETCHING:
        print(f"  Bandwidth Reduction: {stats['bandwidth_reduction_percent']:.1f}%")
    print(f"\nAll results saved to: {logger.exp_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
