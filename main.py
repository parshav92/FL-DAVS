"""
Main Training Script - Phase 1 Baseline (FedAvg)
Orchestrates the complete federated learning process
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from models.cnn_model import get_model
from data.medmnist_loader import MedMNISTDataLoader, create_client_loaders
from federated.client import create_clients
from federated.server import FederatedServer


def setup_directories():
    """Create necessary directories for saving results"""
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def plot_results(history, save_path):
    """
    Plot and save training results
    
    Args:
        history: Dictionary with training/test metrics
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = history['rounds']
    
    # Plot loss
    ax1.plot(rounds, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if history['test_loss']:
        ax1.plot(rounds, history['test_loss'], 'r--', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(rounds, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    if history['test_accuracy']:
        ax2.plot(rounds, history['test_accuracy'], 'r--', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    plt.close()


def main():
    """Main training function"""
    
    print("="*70)
    print("FL+DAVS - Phase 1: Baseline Federated Learning (FedAvg)")
    print("="*70)
    
    # Setup
    setup_directories()
    
    # Configuration summary
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Data Split: {DATA_SPLIT}")
    print(f"  Local Epochs: {LOCAL_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  FL Rounds: {NUM_ROUNDS}")
    print(f"  Device: {DEVICE}")
    
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
    
    # Create clients and server
    print(f"\n{'='*70}")
    print("Initializing Federated Learning")
    print(f"{'='*70}")
    
    clients = create_clients(client_loaders, get_model, device=DEVICE)
    server = FederatedServer(get_model(), device=DEVICE)
    
    print(f"✓ Created {len(clients)} federated clients")
    print(f"✓ Initialized global model ({sum(p.numel() for p in server.global_model.parameters()):,} parameters)")
    
    # Initial evaluation
    print(f"\n{'='*70}")
    print("Initial Model Evaluation")
    print(f"{'='*70}")
    server.evaluate(test_loader)
    
    # Federated training
    print(f"\n{'='*70}")
    print("Starting Federated Training")
    print(f"{'='*70}")
    
    for round_num in range(NUM_ROUNDS):
        # Training round
        train_loss, train_acc = server.train_round(
            clients=clients,
            epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
            client_fraction=1.0  # Use all clients
        )
        
        # Evaluate every few rounds
        if (round_num + 1) % 5 == 0 or (round_num + 1) == NUM_ROUNDS:
            server.evaluate(test_loader)
    
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
    model_path = os.path.join(MODEL_SAVE_DIR, f'fedavg_{DATASET}_r{NUM_ROUNDS}.pth')
    server.save_model(model_path)
    
    # Plot results
    plot_path = os.path.join(RESULTS_DIR, f'fedavg_{DATASET}_r{NUM_ROUNDS}.png')
    plot_results(server.history, plot_path)
    
    # Save metrics to text file
    metrics_path = os.path.join(RESULTS_DIR, f'fedavg_{DATASET}_r{NUM_ROUNDS}_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("FL+DAVS - Phase 1 Baseline Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Clients: {NUM_CLIENTS}\n")
        f.write(f"Data Split: {DATA_SPLIT}\n")
        f.write(f"Rounds: {NUM_ROUNDS}\n")
        f.write(f"Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n\n")
        f.write(f"Final Test Loss: {final_test_loss:.4f}\n")
        f.write(f"Final Test Accuracy: {final_test_acc:.2f}%\n")
    
    print(f"✓ Metrics saved to {metrics_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Results:")
    print(f"  Test Loss: {final_test_loss:.4f}")
    print(f"  Test Accuracy: {final_test_acc:.2f}%")
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
