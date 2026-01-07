import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class FederatedClient:
    """
    Represents a single federated learning client (e.g., a hospital)
    """
    
    def __init__(self, client_id, model, train_loader, device='cpu'):
        """
        Args:
            client_id: Unique identifier for this client
            model: PyTorch model to train
            train_loader: DataLoader for this client's training data
            device: 'cpu' or 'cuda'
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        
        # Training metrics
        self.train_loss = 0.0
        self.train_accuracy = 0.0
    
    def set_parameters(self, parameters):
        """
        Set model parameters from the global model
        
        Args:
            parameters: Dictionary of model parameters
        """
        self.model.load_state_dict(parameters)
    
    def get_parameters(self):
        """
        Get current model parameters
        
        Returns:
            Dictionary of model parameters
        """
        return deepcopy(self.model.state_dict())
    
    def train(self, epochs, learning_rate=0.001):
        """
        Train the model locally for specified number of epochs
        
        Args:
            epochs: Number of local epochs to train
            learning_rate: Learning rate for optimizer
        
        Returns:
            num_samples: Number of training samples
            loss: Average training loss
            accuracy: Training accuracy
        """
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Flatten target if needed (MedMNIST returns [batch, 1])
                if len(target.shape) > 1:
                    target = target.squeeze(-1)  
                if target.size(0) == 0:
                    continue
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                epoch_total += target.size(0)
                epoch_correct += predicted.eq(target).sum().item()
            
            # Accumulate metrics
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        # Calculate averages
        num_batches = len(self.train_loader) * epochs
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        self.train_loss = avg_loss
        self.train_accuracy = accuracy
        
        num_samples = len(self.train_loader.dataset)
        
        return num_samples, avg_loss, accuracy
    
    def compute_gradients(self, global_params=None):
        """
        Compute and return gradients/updates relative to global parameters
        
        Args:
            global_params: Dictionary of global model parameters (optional)
                          If provided, returns the difference between local and global params
                          If None, returns the accumulated gradients from last backward pass
        
        Returns:
            Dictionary of gradients/updates for each parameter
        """
        if global_params is not None:
            # Compute parameter updates (local_params - global_params)
            gradients = {}
            local_params = self.get_parameters()
            for name, local_param in local_params.items():
                if name in global_params:
                    gradients[name] = local_param - global_params[name]
            return gradients
        else:
            # Return accumulated gradients from last backward pass
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            return gradients


def create_clients(client_datasets, model_fn, device='cpu'):
    """
    Create federated clients from client datasets
    
    Args:
        client_datasets: Dictionary of {client_id: DataLoader}
        model_fn: Function that returns a new model instance
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary of {client_id: FederatedClient}
    """
    clients = {}
    
    for client_id, train_loader in client_datasets.items():
        model = model_fn()
        clients[client_id] = FederatedClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            device=device
        )
    
    return clients


if __name__ == "__main__":
    # Test the client module
    print("Testing Federated Client...\n")
    
    from models.cnn_model import get_model
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy data
    dummy_data = torch.randn(100, 3, 28, 28)
    dummy_labels = torch.randint(0, 9, (100,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create client
    model = get_model()
    client = FederatedClient(
        client_id=0,
        model=model,
        train_loader=train_loader,
        device='cpu'
    )
    
    print(f"Client {client.client_id} created")
    print(f"Training samples: {len(dataset)}")
    
    # Train for 2 epochs
    num_samples, loss, accuracy = client.train(epochs=2, learning_rate=0.001)
    
    print(f"\nTraining complete:")
    print(f"  Samples: {num_samples}")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print("\n✓ Client module working correctly!")
