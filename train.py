"""
Enhanced Main Training Script with Metrics Logging
Phase 1 Baseline - FedAvg with comprehensive tracking
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from models.cnn_model import get_model
from data.medmnist_loader import MedMNISTDataLoader, create_client_loaders
from federated.client import create_clients
from federated.server import FederatedServer
from utils.metrics import MetricsLogger, plot_client_contributions


def setup_directories():
    """Create necessary directories for saving results"""
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def main():
    """Main training function with comprehensive logging"""
    
    print("="*70)
    print("FL+DAVS - Phase 1: Baseline Federated Learning (FedAvg)")
    print("="*70)
    
    # Setup
    setup_directories()
    
    # Initialize metrics logger
    logger = MetricsLogger(
        experiment_name=f"fedavg_{DATASET}_{DATA_SPLIT}",
        save_dir=RESULTS_DIR
    )
    
    # Configuration
    config = {
        'dataset': DATASET,
        'num_clients': NUM_CLIENTS,
        'data_split': DATA_SPLIT,
        'local_epochs': LOCAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_rounds': NUM_ROUNDS,
        'device': DEVICE,
        'model': 'SimpleCNN'
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
    
    # Partition data based on configuration
    if DATA_SPLIT == 'iid':
        client_datasets = data_loader.partition_iid()
    else:  # non-iid
        client_datasets = data_loader.partition_non_iid(alpha=0.5)
    
    # Create client data loaders
    client_loaders = create_client_loaders(client_datasets, batch_size=BATCH_SIZE)
    test_loader = data_loader.get_test_loader(batch_size=BATCH_SIZE)
    
    # Plot data distribution
    client_data_sizes = {f"Client {i}": len(client_datasets[i]) for i in range(NUM_CLIENTS)}
    plot_client_contributions(
        client_data_sizes, 
        os.path.join(logger.exp_dir, 'data_distribution.png')
    )
    print(f"✓ Data distribution plotted")
    
    # Create clients and server
    print(f"\n{'='*70}")
    print("Initializing Federated Learning")
    print(f"{'='*70}")
    
    clients = create_clients(client_loaders, get_model, device=DEVICE)
    server = FederatedServer(get_model(), device=DEVICE)
    
    num_params = sum(p.numel() for p in server.global_model.parameters())
    print(f"✓ Created {len(clients)} federated clients")
    print(f"✓ Initialized global model ({num_params:,} parameters)")
    
    # Initial evaluation
    print(f"\n{'='*70}")
    print("Initial Model Evaluation")
    print(f"{'='*70}")
    init_loss, init_acc = server.evaluate(test_loader)
    logger.log_round(0, init_loss, init_acc, init_loss, init_acc)
    
    # Federated training
    print(f"\n{'='*70}")
    print("Starting Federated Training")
    print(f"{'='*70}")
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\nRound {round_num}/{NUM_ROUNDS}")
        print("-" * 50)
        
        # Training round
        train_loss, train_acc = server.train_round(
            clients=clients,
            epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
            client_fraction=1.0  # Use all clients
        )
        
        # Evaluate on test set every 5 rounds or at the end
        if round_num % 5 == 0 or round_num == NUM_ROUNDS:
            test_loss, test_acc = server.evaluate(test_loader)
            logger.log_round(round_num, train_loss, train_acc, test_loss, test_acc)
        else:
            # Log only training metrics
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
    
    # Save model
    model_path = os.path.join(logger.exp_dir, f'final_model.pth')
    server.save_model(model_path)
    
    # Save metrics and plots
    logger.save_json()
    logger.plot_training_curves()
    logger.save_summary(config)
    
    # Summary
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Results:")
    print(f"  Test Loss: {final_test_loss:.4f}")
    print(f"  Test Accuracy: {final_test_acc:.2f}%")
    print(f"\nAll results saved to: {logger.exp_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
