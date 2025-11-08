"""
MedMNIST Data Loader
Handles loading and partitioning MedMNIST datasets across federated clients
Phase 1 - Basic Federated Learning
"""

import numpy as np
import medmnist
from medmnist import INFO
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


class MedMNISTDataLoader:
    """
    Loads MedMNIST dataset and partitions it across multiple clients
    """
    
    def __init__(self, dataset_name='pathmnist', num_clients=10, data_dir='./data'):
        """
        Args:
            dataset_name: Name of MedMNIST dataset (pathmnist, chestmnist, etc.)
            num_clients: Number of federated clients
            data_dir: Directory to store/load data
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.data_dir = data_dir
        
        # Get dataset information
        self.info = INFO[dataset_name]
        self.task = self.info['task']
        self.n_channels = self.info['n_channels']
        self.n_classes = len(self.info['label'])
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        # Load datasets
        self.train_dataset = None
        self.test_dataset = None
        self.load_data()
    
    def load_data(self):
        """Load the MedMNIST dataset"""
        DataClass = getattr(medmnist, self.info['python_class'])
        
        # Load training and test datasets
        self.train_dataset = DataClass(
            split='train',
            transform=self.transform,
            download=True,
            root=self.data_dir
        )
        
        self.test_dataset = DataClass(
            split='test',
            transform=self.transform,
            download=True,
            root=self.data_dir
        )
        
        print(f"✓ Loaded {self.dataset_name}")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Channels: {self.n_channels}")
    
    def partition_iid(self):
        """
        Partition data in an IID manner across clients
        Each client gets an equal number of samples randomly distributed
        """
        num_samples = len(self.train_dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        
        # Divide indices equally among clients
        samples_per_client = num_samples // self.num_clients
        client_datasets = {}
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < self.num_clients - 1 else num_samples
            client_indices = indices[start_idx:end_idx]
            client_datasets[i] = Subset(self.train_dataset, client_indices)
        
        print(f"\n✓ Partitioned data (IID) across {self.num_clients} clients")
        print(f"  Samples per client: ~{samples_per_client}")
        
        return client_datasets
    
    def partition_non_iid(self, alpha=0.5):
        """
        Partition data in a non-IID manner using Dirichlet distribution
        Each client gets a different distribution of classes
        
        Args:
            alpha: Concentration parameter for Dirichlet distribution
                   (lower = more heterogeneous)
        """
        # Get labels for all training samples
        labels = np.array([self.train_dataset[i][1].item() for i in range(len(self.train_dataset))])
        
        # Sort by label
        label_indices = [np.where(labels == i)[0] for i in range(self.n_classes)]
        
        # Allocate samples to clients using Dirichlet distribution
        client_datasets = {i: [] for i in range(self.num_clients)}
        
        for class_indices in label_indices:
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            
            # Split class indices according to proportions
            class_splits = np.split(class_indices, proportions)
            
            # Assign to clients
            for client_id, indices in enumerate(class_splits):
                client_datasets[client_id].extend(indices.tolist())
        
        # Convert to Subset objects
        for client_id in range(self.num_clients):
            np.random.shuffle(client_datasets[client_id])
            client_datasets[client_id] = Subset(self.train_dataset, client_datasets[client_id])
        
        print(f"\n✓ Partitioned data (Non-IID, α={alpha}) across {self.num_clients} clients")
        for i in range(min(3, self.num_clients)):  # Show first 3 clients
            print(f"  Client {i}: {len(client_datasets[i])} samples")
        
        return client_datasets
    
    def get_test_loader(self, batch_size=32):
        """Get DataLoader for test dataset"""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )


def create_client_loaders(client_datasets, batch_size=32):
    """
    Create DataLoaders for each client
    
    Args:
        client_datasets: Dictionary of client datasets
        batch_size: Batch size for training
    
    Returns:
        Dictionary of DataLoaders for each client
    """
    client_loaders = {}
    for client_id, dataset in client_datasets.items():
        client_loaders[client_id] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
    return client_loaders


if __name__ == "__main__":
    # Test the data loader
    print("Testing MedMNIST Data Loader...\n")
    
    # Load PathMNIST
    data_loader = MedMNISTDataLoader(
        dataset_name='pathmnist',
        num_clients=5,
        data_dir='./data'
    )
    
    # Test IID partitioning
    print("\n" + "="*50)
    print("IID Partitioning:")
    print("="*50)
    iid_datasets = data_loader.partition_iid()
    
    # Test Non-IID partitioning
    print("\n" + "="*50)
    print("Non-IID Partitioning:")
    print("="*50)
    non_iid_datasets = data_loader.partition_non_iid(alpha=0.5)
    
    # Create loaders
    client_loaders = create_client_loaders(non_iid_datasets, batch_size=32)
    print(f"\n✓ Created DataLoaders for {len(client_loaders)} clients")
