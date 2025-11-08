#!/usr/bin/env python3
"""
System Verification Script
Checks that all components are properly installed and configured
"""

import sys
import os

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def check_python():
    print_header("Python Version")
    print(f"Python {sys.version}")
    if sys.version_info >= (3, 12):
        print("✅ Python version OK")
        return True
    else:
        print("❌ Python 3.12+ required")
        return False

def check_imports():
    print_header("Checking Required Packages")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'flwr': 'Flower',
        'medmnist': 'MedMNIST',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn'
    }
    
    all_ok = True
    for pkg, name in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name:20s} {version}")
        except ImportError:
            print(f"❌ {name:20s} NOT FOUND")
            all_ok = False
    
    return all_ok

def check_project_structure():
    print_header("Checking Project Structure")
    
    required_files = [
        'config.py',
        'train.py',
        'main.py',
        'quickstart.py',
        'models/cnn_model.py',
        'data/medmnist_loader.py',
        'federated/client.py',
        'federated/server.py',
        'federated/aggregation.py',
        'utils/metrics.py'
    ]
    
    all_ok = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath} MISSING")
            all_ok = False
    
    return all_ok

def check_model():
    print_header("Testing Model")
    
    try:
        from models.cnn_model import get_model
        import torch
        
        model = get_model()
        x = torch.randn(4, 3, 28, 28)
        y = model(x)
        
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Model created successfully")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {y.shape}")
        print(f"   Parameters: {num_params:,}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def check_data():
    print_header("Testing Data Loader")
    
    try:
        from data.medmnist_loader import MedMNISTDataLoader
        
        loader = MedMNISTDataLoader(
            dataset_name='pathmnist',
            num_clients=3,
            data_dir='./data'
        )
        
        print(f"✅ Data loader created successfully")
        print(f"   Dataset: {loader.dataset_name}")
        print(f"   Train size: {len(loader.train_dataset)}")
        print(f"   Test size: {len(loader.test_dataset)}")
        
        # Test partitioning
        client_datasets = loader.partition_iid()
        print(f"   Partitioned into {len(client_datasets)} clients")
        
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def check_config():
    print_header("Configuration")
    
    try:
        import config
        
        settings = {
            'Dataset': config.DATASET,
            'Clients': config.NUM_CLIENTS,
            'Rounds': config.NUM_ROUNDS,
            'Data Split': config.DATA_SPLIT,
            'Local Epochs': config.LOCAL_EPOCHS,
            'Batch Size': config.BATCH_SIZE,
            'Learning Rate': config.LEARNING_RATE,
            'Device': config.DEVICE
        }
        
        for key, value in settings.items():
            print(f"   {key:20s}: {value}")
        
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("FL+DAVS SYSTEM VERIFICATION")
    print("="*70)
    
    checks = [
        ("Python Version", check_python),
        ("Package Imports", check_imports),
        ("Project Structure", check_project_structure),
        ("Configuration", check_config),
        ("Model Test", check_model),
        ("Data Loader Test", check_data)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - SYSTEM READY!")
        print("\nNext steps:")
        print("  1. Run quick test: python quickstart.py")
        print("  2. Run full training: python train.py")
        print("  3. Check NEXT_STEPS.md for more info")
    else:
        print("⚠️  SOME CHECKS FAILED - PLEASE FIX BEFORE RUNNING")
        print("\nTroubleshooting:")
        print("  1. Make sure virtual environment is activated:")
        print("     source fl_davs_env/bin/activate")
        print("  2. Reinstall dependencies:")
        print("     pip install -r requirements.txt")
        print("  3. Check STATUS.txt for more info")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
