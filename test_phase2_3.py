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
    """Test gradient sketching implementation with Random Projection"""
    print("\n" + "="*70)
    print("TEST 1: Random Projection Gradient Sketching")
    print("="*70)
    
    try:
        from federated.gradient_sketching import GradientSketcherForDAVS
        from models.cnn_model import SimpleCNN
        import torch
        
        # Create model
        model = SimpleCNN(num_classes=9)
        
        # Create sketcher
        sketcher = GradientSketcherForDAVS(
            model=model,
            sketch_dim=128,
            add_dp_noise=False,
            shared_seed=42
        )
        
        print(f"✓ Sketcher initialized")
        print(f"  Total parameters: {sketcher.total_params:,}")
        print(f"  Sketch dimension: {sketcher.sketch_dim}")
        print(f"  Compression: {sketcher.total_params/sketcher.sketch_dim:.1f}x")
        print(f"  Bandwidth reduction: {sketcher.sketcher.bandwidth_reduction:.1f}%")
        
        # Test gradient sketching
        # Create dummy data and compute gradients
        dummy_input = torch.randn(4, 3, 28, 28)
        dummy_target = torch.randint(0, 9, (4,))
        
        # Forward pass
        output = model(dummy_input)
        loss = torch.nn.CrossEntropyLoss()(output, dummy_target)
        loss.backward()
        
        # Extract and sketch gradients
        sketch = sketcher.sketch_gradients(model)
        
        print(f"✓ Gradient extraction and sketching successful")
        print(f"  Sketch shape: {sketch.shape}")
        
        # Test similarity preservation
        grad1 = torch.randn(sketcher.total_params)
        grad2 = grad1 + torch.randn(sketcher.total_params) * 0.1  # Similar
        grad3 = torch.randn(sketcher.total_params)  # Different
        
        sketch1 = sketcher.sketcher.sketch(grad1)
        sketch2 = sketcher.sketcher.sketch(grad2)
        sketch3 = sketcher.sketcher.sketch(grad3)
        
        sim_12 = sketcher.sketcher.compute_cosine_similarity(sketch1, sketch2)
        sim_13 = sketcher.sketcher.compute_cosine_similarity(sketch1, sketch3)
        
        print(f"✓ Similarity preservation test:")
        print(f"  Similar gradients: {sim_12:.4f}")
        print(f"  Different gradients: {sim_13:.4f}")
        
        if sim_12 > sim_13:
            print("✅ PASS: Random Projection preserves gradient similarity")
            return True
        else:
            print("❌ FAIL: Similarity not preserved correctly")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_davs_selection():
    """Test DAVS committee selection with gradient sketches"""
    print("\n" + "="*70)
    print("TEST 2: DAVS Committee Selection (Amnesic)")
    print("="*70)
    
    try:
        from federated.davs_selection import DAVSSelector
        import torch
        import numpy as np
        
        # Initialize selector
        num_clients = 10
        sketch_dim = 128
        selector = DAVSSelector(committee_size=5)
        
        print(f"✓ DAVS Selector initialized")
        print(f"  Total clients: {num_clients}")
        print(f"  Committee size: {selector.committee_size}")
        print(f"  Algorithm: Amnesic (no history)")
        
        # Create gradient sketches (DAVS works on sketches, not full gradients)
        client_sketches = {}
        
        # Honest clients (similar sketches)
        honest_base = torch.randn(sketch_dim)
        for i in range(7):
            honest_sketch = honest_base + torch.randn(sketch_dim) * 0.2
            client_sketches[i] = honest_sketch
        
        # Malicious clients (very different sketches)  
        for i in range(7, num_clients):
            malicious_sketch = torch.randn(sketch_dim) * 5.0
            client_sketches[i] = malicious_sketch
        
        print(f"✓ Created {num_clients} gradient sketches ({sketch_dim}-dim)")
        
        # Select committee using DAVS (gradient similarity only)
        committee, scores = selector.select_committee(client_sketches)
        
        print(f"✓ Selected committee: {committee}")
        print(f"  Committee size: {len(committee)}")
        
        # Check Byzantine resilience
        malicious_in_committee = [c for c in committee if c >= 7]
        print(f"✓ Malicious clients in committee: {len(malicious_in_committee)}/{len(committee)}")
        
        if len(committee) == 5 and len(malicious_in_committee) <= 1:
            print("✅ PASS: DAVS selection works correctly")
            print("  (Low representativeness filtered out most malicious clients)")
            return True
        else:
            print("❌ FAIL: Committee selection issue")
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
        from federated.gradient_sketching import GradientSketcherForDAVS
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
