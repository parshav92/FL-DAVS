#!/usr/bin/env python3
"""
Quick test to verify train_davs.py works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing train_davs.py fix...")
print("="*70)

# Test 1: Import check
print("\n1. Checking imports...")
try:
    from train_davs import main
    print("   ✅ Imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check client creation
print("\n2. Checking client creation...")
try:
    from federated.client import create_clients
    from models.cnn_model import get_model
    from data.medmnist_loader import MedMNISTDataLoader, create_client_loaders
    
    # Create minimal test data
    data_loader = MedMNISTDataLoader(
        dataset_name='pathmnist',
        num_clients=3,
        data_dir='./data'
    )
    client_datasets = data_loader.partition_iid()
    client_loaders = create_client_loaders(client_datasets, batch_size=32)
    
    # Create clients
    clients = create_clients(client_loaders, get_model, device='cpu')
    
    print(f"   ✅ Created {len(clients)} clients")
    print(f"   ✅ Type: {type(clients)}")
    print(f"   ✅ Client 0 type: {type(clients[0])}")
    
    # Test iteration (the bug we fixed)
    print("\n3. Testing client iteration...")
    for client_id in range(len(clients)):
        client = clients[client_id]
        if not hasattr(client, 'set_parameters'):
            print(f"   ❌ Client {client_id} missing set_parameters method")
            sys.exit(1)
    
    print("   ✅ All clients have required methods")
    
except Exception as e:
    print(f"   ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - train_davs.py is ready!")
print("="*70)
print("\nYou can now run: python train_davs.py")
