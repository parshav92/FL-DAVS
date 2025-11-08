"""
Federated Learning Server
Orchestrates the federated learning process
Phase 1 - Basic Federated Learning
"""

import torch
import torch.nn as nn
from copy import deepcopy
from federated.aggregation import fedavg, aggregate_metrics


class FederatedServer:
    """
    Central server that coordinates federated learning
    """
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: Global model to be trained
            device: 'cpu' or 'cuda'
        """
        self.global_model = model.to(device)
        self.device = device
        self.round_num = 0
        
        # Track training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'rounds': []
        }
    
    def get_global_parameters(self):
        """
        Get current global model parameters
        
        Returns:
            Dictionary of model parameters
        """
        return deepcopy(self.global_model.state_dict())
    
    def set_global_parameters(self, parameters):
        """
        Update global model with new parameters
        
        Args:
            parameters: Dictionary of model parameters
        """
        self.global_model.load_state_dict(parameters)
    
    def select_clients(self, clients, fraction=1.0):
        """
        Select subset of clients for training round
        
        Args:
            clients: Dictionary of all available clients
            fraction: Fraction of clients to select (1.0 = all clients)
        
        Returns:
            Dictionary of selected clients
        """
        import random
        
        num_clients = len(clients)
        num_selected = max(1, int(num_clients * fraction))
        
        selected_ids = random.sample(list(clients.keys()), num_selected)
        selected_clients = {cid: clients[cid] for cid in selected_ids}
        
        return selected_clients
    
    def train_round(self, clients, epochs, learning_rate, client_fraction=1.0):
        """
        Execute one round of federated learning
        
        Args:
            clients: Dictionary of FederatedClient objects
            epochs: Number of local epochs per client
            learning_rate: Learning rate for local training
            client_fraction: Fraction of clients to select
        
        Returns:
            Tuple of (avg_train_loss, avg_train_accuracy)
        """
        # Select clients for this round
        selected_clients = self.select_clients(clients, client_fraction)
        
        print(f"\n--- Round {self.round_num + 1} ---")
        print(f"Selected {len(selected_clients)}/{len(clients)} clients")
        
        # Distribute global model to selected clients
        global_params = self.get_global_parameters()
        for client in selected_clients.values():
            client.set_parameters(global_params)
        
        # Train clients locally
        client_parameters = []
        client_weights = []
        client_metrics = []
        
        for client_id, client in selected_clients.items():
            # Local training
            num_samples, loss, accuracy = client.train(
                epochs=epochs,
                learning_rate=learning_rate
            )
            
            # Collect results
            client_parameters.append(client.get_parameters())
            client_weights.append(num_samples)
            client_metrics.append((loss, accuracy))
            
            print(f"  Client {client_id}: Loss={loss:.4f}, Acc={accuracy:.2f}%, Samples={num_samples}")
        
        # Aggregate client models
        aggregated_params = fedavg(client_parameters, client_weights)
        self.set_global_parameters(aggregated_params)
        
        # Aggregate metrics
        avg_loss, avg_accuracy = aggregate_metrics(client_metrics, client_weights)
        
        print(f"Round {self.round_num + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.2f}%")
        
        # Update history
        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(avg_accuracy)
        self.history['rounds'].append(self.round_num + 1)
        
        self.round_num += 1
        
        return avg_loss, avg_accuracy
    
    def evaluate(self, test_loader):
        """
        Evaluate global model on test data
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        self.global_model.eval()
        
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Flatten target if needed
                if len(target.shape) > 1:
                    target = target.squeeze()
                
                # Forward pass
                output = self.global_model(data)
                loss = criterion(output, target)
                
                # Track metrics
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # Calculate averages
        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        # Update history
        self.history['test_loss'].append(avg_loss)
        self.history['test_accuracy'].append(accuracy)
        
        print(f"\nTest Set Evaluation:")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def save_model(self, path):
        """
        Save the global model
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'round': self.round_num,
            'history': self.history
        }, path)
        print(f"\n✓ Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a saved model
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.round_num = checkpoint['round']
        self.history = checkpoint['history']
        print(f"\n✓ Model loaded from {path}")


if __name__ == "__main__":
    # Test the server module
    print("Testing Federated Server...\n")
    
    from models.cnn_model import get_model
    from federated.client import create_clients
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy data and clients
    dummy_data = torch.randn(300, 3, 28, 28)
    dummy_labels = torch.randint(0, 9, (300,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    # Split into 3 clients
    client_datasets = {
        0: DataLoader(TensorDataset(dummy_data[:100], dummy_labels[:100]), batch_size=32),
        1: DataLoader(TensorDataset(dummy_data[100:200], dummy_labels[100:200]), batch_size=32),
        2: DataLoader(TensorDataset(dummy_data[200:], dummy_labels[200:]), batch_size=32),
    }
    
    # Create clients and server
    clients = create_clients(client_datasets, get_model, device='cpu')
    server = FederatedServer(get_model(), device='cpu')
    
    print(f"Server created with {len(clients)} clients")
    
    # Run one training round
    avg_loss, avg_acc = server.train_round(
        clients=clients,
        epochs=1,
        learning_rate=0.001,
        client_fraction=1.0
    )
    
    print("\n✓ Server module working correctly!")
