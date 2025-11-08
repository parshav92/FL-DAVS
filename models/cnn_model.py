"""
Simple CNN Model for MedMNIST Classification
Phase 1 - Basic Federated Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for MedMNIST image classification
    - 2 convolutional layers
    - 2 fully connected layers
    - Suitable for 28x28 RGB images
    """
    
    def __init__(self, input_channels=3, num_classes=9):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 2 pooling layers: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv layer 1 + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv layer 2 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layer 1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # FC layer 2 (output)
        x = self.fc2(x)
        
        return x


def get_model(input_channels=3, num_classes=9):
    """
    Factory function to create and return a new model instance
    """
    return SimpleCNN(input_channels, num_classes)


if __name__ == "__main__":
    # Test the model
    model = get_model()
    
    # Create a dummy input (batch_size=4, channels=3, height=28, width=28)
    dummy_input = torch.randn(4, 3, 28, 28)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Model architecture:")
    print(model)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
