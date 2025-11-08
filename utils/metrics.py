"""
Utilities for metrics logging and visualization
"""

import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class MetricsLogger:
    """Logger for tracking and saving experiment metrics"""
    
    def __init__(self, experiment_name, save_dir='results'):
        """
        Initialize metrics logger
        
        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.start_time = time.time()
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(save_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Metrics storage
        self.metrics = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'timestamps': []
        }
        
        print(f"✓ Metrics logger initialized: {self.exp_dir}")
    
    def log_round(self, round_num, train_loss, train_acc, test_loss=None, test_acc=None):
        """Log metrics for a training round"""
        self.metrics['rounds'].append(round_num)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_accuracy'].append(train_acc)
        self.metrics['test_loss'].append(test_loss if test_loss is not None else 0.0)
        self.metrics['test_accuracy'].append(test_acc if test_acc is not None else 0.0)
        self.metrics['timestamps'].append(time.time() - self.start_time)
    
    def save_json(self):
        """Save metrics to JSON file"""
        json_path = os.path.join(self.exp_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Metrics saved to {json_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        rounds = self.metrics['rounds']
        
        # Loss curves
        axes[0, 0].plot(rounds, self.metrics['train_loss'], 'b-', label='Train', linewidth=2)
        if any(self.metrics['test_loss']):
            axes[0, 0].plot(rounds, self.metrics['test_loss'], 'r--', label='Test', linewidth=2)
        axes[0, 0].set_xlabel('Round', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(rounds, self.metrics['train_accuracy'], 'b-', label='Train', linewidth=2)
        if any(self.metrics['test_accuracy']):
            axes[0, 1].plot(rounds, self.metrics['test_accuracy'], 'r--', label='Test', linewidth=2)
        axes[0, 1].set_xlabel('Round', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training time per round
        time_per_round = np.diff([0] + self.metrics['timestamps'])
        axes[1, 0].bar(rounds, time_per_round, color='steelblue', alpha=0.7)
        axes[1, 0].set_xlabel('Round', fontsize=12)
        axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
        axes[1, 0].set_title('Time per Round', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Final test accuracy bar
        if any(self.metrics['test_accuracy']):
            final_acc = self.metrics['test_accuracy'][-1]
            axes[1, 1].bar(['Final Test Accuracy'], [final_acc], color='green', alpha=0.7)
            axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
            axes[1, 1].set_title('Final Performance', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].text(0, final_acc + 2, f'{final_acc:.2f}%', ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(self.exp_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {plot_path}")
        plt.close()
    
    def save_summary(self, config):
        """Save experiment summary"""
        summary_path = os.path.join(self.exp_dir, 'summary.txt')
        
        total_time = time.time() - self.start_time
        
        with open(summary_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"FL+DAVS Experiment Summary\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"{'-'*60}\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nFinal Results:\n")
            f.write(f"{'-'*60}\n")
            f.write(f"  Final Train Loss: {self.metrics['train_loss'][-1]:.4f}\n")
            f.write(f"  Final Train Accuracy: {self.metrics['train_accuracy'][-1]:.2f}%\n")
            
            if any(self.metrics['test_loss']):
                f.write(f"  Final Test Loss: {self.metrics['test_loss'][-1]:.4f}\n")
                f.write(f"  Final Test Accuracy: {self.metrics['test_accuracy'][-1]:.2f}%\n")
            
            f.write(f"\nBest Results:\n")
            f.write(f"{'-'*60}\n")
            best_train_acc_idx = np.argmax(self.metrics['train_accuracy'])
            f.write(f"  Best Train Accuracy: {self.metrics['train_accuracy'][best_train_acc_idx]:.2f}% ")
            f.write(f"(Round {self.metrics['rounds'][best_train_acc_idx]})\n")
            
            if any(self.metrics['test_accuracy']):
                best_test_acc_idx = np.argmax(self.metrics['test_accuracy'])
                f.write(f"  Best Test Accuracy: {self.metrics['test_accuracy'][best_test_acc_idx]:.2f}% ")
                f.write(f"(Round {self.metrics['rounds'][best_test_acc_idx]})\n")
        
        print(f"✓ Summary saved to {summary_path}")


def plot_client_contributions(client_data_sizes, save_path):
    """
    Plot data distribution across clients
    
    Args:
        client_data_sizes: Dict mapping client_id to number of samples
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    client_ids = list(client_data_sizes.keys())
    sizes = list(client_data_sizes.values())
    
    plt.bar(client_ids, sizes, color='steelblue', alpha=0.7)
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Data Distribution Across Clients', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    mean_size = np.mean(sizes)
    plt.axhline(y=mean_size, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_size:.0f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_experiments(exp_dirs, labels, save_dir):
    """
    Compare multiple experiments
    
    Args:
        exp_dirs: List of experiment directories
        labels: List of experiment labels
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for exp_dir, label in zip(exp_dirs, labels):
        # Load metrics
        metrics_path = os.path.join(exp_dir, 'metrics.json')
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Plot test accuracy
        ax1.plot(metrics['rounds'], metrics['test_accuracy'], linewidth=2, label=label, marker='o', markersize=4)
        
        # Plot test loss
        ax2.plot(metrics['rounds'], metrics['test_loss'], linewidth=2, label=label, marker='o', markersize=4)
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'experiment_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {comparison_path}")
    plt.close()


if __name__ == "__main__":
    # Test the logger
    print("Testing MetricsLogger...")
    
    logger = MetricsLogger("test_experiment", save_dir="results")
    
    # Simulate training
    for round_num in range(1, 11):
        train_loss = 2.0 - round_num * 0.15 + np.random.rand() * 0.1
        train_acc = 20 + round_num * 6 + np.random.rand() * 5
        test_loss = 2.1 - round_num * 0.13 + np.random.rand() * 0.1
        test_acc = 18 + round_num * 6 + np.random.rand() * 5
        
        logger.log_round(round_num, train_loss, train_acc, test_loss, test_acc)
        time.sleep(0.1)  # Simulate training time
    
    # Save results
    config = {
        'dataset': 'pathmnist',
        'num_clients': 10,
        'rounds': 10,
        'local_epochs': 2,
        'batch_size': 32
    }
    
    logger.save_json()
    logger.plot_training_curves()
    logger.save_summary(config)
    
    print("\n✓ All tests passed!")
