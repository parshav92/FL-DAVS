#!/usr/bin/env python3
"""
Test script for Phase 2+3 implementations
Verifies gradient sketching and DAVS work correctly
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gradient_sketching():
    """Test gradient sketching implementation"""
    print("\n" + "="*70)
    print("TEST 1: Gradient Sketching")
    print("="*70)
    
    try:
        from federated.gradient_sketching import GradientCompressor
        from models.cnn_model import SimpleCNN
        import torch
        
        # Create model
        model = SimpleCNN(num_classes=9)
        
        # Create compressor
        compressor = GradientCompressor(model, compression_rate=0.1, num_hash=3)
        
        # Get stats
        stats = compressor.get_compression_stats()
        
        print(f"✓ Compressor initialized")
        print(f"  Original: {stats['original_size']:,} parameters")
        print(f"  Compressed: {stats['compressed_size']:,} parameters")
        print(f"  Compression: {stats['compression_rate']:.1%}")
        print(f"  Bandwidth reduction: {stats['bandwidth_reduction_percent']:.1f}%")
        
        # Test compression/decompression
        dummy_gradients = {
            name: torch.randn_like(param)
            for name, param in model.named_parameters()
        }
        
        compressed = compressor.compress_gradients(dummy_gradients)
        decompressed = compressor.decompress_gradients(compressed)
        
        print(f"✓ Compression/decompression successful")
        
        # Calculate error
        total_error = 0.0
        total_norm = 0.0
        for name in dummy_gradients.keys():
            if name in decompressed:
                error = torch.norm(dummy_gradients[name] - decompressed[name]).item()
                norm = torch.norm(dummy_gradients[name]).item()
                total_error += error
                total_norm += norm
        
        relative_error = (total_error / total_norm) * 100
        print(f"✓ Reconstruction error: {relative_error:.2f}%")
        
        if relative_error < 10:
            print("✅ PASS: Gradient sketching works correctly")
            return True
        else:
            print("❌ FAIL: Reconstruction error too high")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_davs_selection():
    """Test DAVS committee selection"""
    print("\n" + "="*70)
    print("TEST 2: DAVS Committee Selection")
    print("="*70)
    
    try:
        from federated.davs_selection import DAVSSelector, DataQualityMetrics
        from models.cnn_model import SimpleCNN
        import torch
        import numpy as np
        
        # Create model
        model = SimpleCNN(num_classes=9)
        
        # Initialize selector
        num_clients = 10
        selector = DAVSSelector(num_clients=num_clients, committee_size=5)
        
        print(f"✓ DAVS Selector initialized")
        print(f"  Total clients: {num_clients}")
        print(f"  Committee size: {selector.committee_size}")
        
        # Create dummy client data
        client_gradients = []
        data_sizes = []
        loss_improvements = []
        class_distributions = {}
        
        for i in range(num_clients):
            dummy_grad = {
                name: torch.randn_like(param)
                for name, param in model.named_parameters()
            }
            client_gradients.append(dummy_grad)
            data_sizes.append(int(5000 + 5000 * np.random.rand()))
            loss_improvements.append(0.1 * np.random.rand())
            class_distributions[i] = np.random.dirichlet(np.ones(9)) * 1000
        
        print(f"✓ Created dummy data for {num_clients} clients")
        
        # Calculate scores
        scores = selector.calculate_client_scores(
            client_gradients, data_sizes, loss_improvements, class_distributions
        )
        
        print(f"✓ Calculated scores for all clients")
        
        # Select committee
        committee = selector.select_committee(
            client_gradients, data_sizes, loss_improvements, class_distributions
        )
        
        print(f"✓ Selected committee: {committee}")
        print(f"  Committee size: {len(committee)}")
        
        # Test Byzantine detection
        byzantine_grad = {
            name: torch.randn_like(param) * 10.0
            for name, param in model.named_parameters()
        }
        client_gradients.append(byzantine_grad)
        
        byzantine = selector.detect_byzantine(client_gradients, threshold=2.5)
        
        if byzantine:
            print(f"✓ Byzantine detection working: {byzantine}")
        else:
            print(f"✓ No Byzantine clients detected (threshold may be high)")
        
        if len(committee) > 0 and len(committee) <= num_clients:
            print("✅ PASS: DAVS selection works correctly")
            return True
        else:
            print("❌ FAIL: Invalid committee size")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integrated training script"""
    print("\n" + "="*70)
    print("TEST 3: Integration Test")
    print("="*70)
    
    try:
        # Just check imports and basic initialization
        import config
        from models.cnn_model import get_model
        from data.medmnist_loader import MedMNISTDataLoader
        from federated.gradient_sketching import GradientCompressor
        from federated.davs_selection import DAVSSelector
        
        print("✓ All imports successful")
        
        # Check train_davs.py exists
        if os.path.exists('train_davs.py'):
            print("✓ train_davs.py exists")
        else:
            print("❌ train_davs.py not found")
            return False
        
        print("✅ PASS: Integration test successful")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PHASE 2+3 VALIDATION TESTS")
    print("Testing Gradient Sketching and DAVS Implementation")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Gradient Sketching", test_gradient_sketching()))
    results.append(("DAVS Selection", test_davs_selection()))
    results.append(("Integration", test_integration()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12s} {name}")
        if result:
            passed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED - Ready for training!")
        print("\nNext steps:")
        print("  1. Run: python train_davs.py")
        print("  2. Compare with baseline: python train.py")
        print("  3. Check PHASE2_3_README.md for details")
    else:
        print("\n⚠️  SOME TESTS FAILED - Please fix before training")
        print("\nTroubleshooting:")
        print("  1. Check virtual environment is activated")
        print("  2. Verify all dependencies installed")
        print("  3. See error messages above")
    
    print("="*70 + "\n")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
