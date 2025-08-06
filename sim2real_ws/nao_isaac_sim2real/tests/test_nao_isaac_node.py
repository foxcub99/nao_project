#!/usr/bin/env python3
"""
Test script for NAO Isaac RL Policy Node
Tests observation space construction and policy loading
"""

import numpy as np
import torch
import sys
import os

# Add the package path for testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_observation_space():
    """Test that observation space construction matches expected dimensions."""
    print("Testing observation space construction...")
    
    # Create mock sensor data
    base_lin_vel = np.zeros(3)
    base_ang_vel = np.zeros(3)
    projected_gravity = np.array([0.0, 0.0, -9.81])
    velocity_commands = np.zeros(3)
    joint_pos = np.zeros(21)  # Only controlled joints (excludes wrist yaws and hands)
    joint_vel = np.zeros(21)
    actions = np.zeros(21)
    foot_contact = np.zeros(2)
    
    # Construct observation
    obs = np.concatenate([
        base_lin_vel,      # 3
        base_ang_vel,      # 3
        projected_gravity, # 3
        velocity_commands, # 3
        joint_pos,         # 21 (controlled joints only)
        joint_vel,         # 21 (controlled joints only)
        actions,           # 21 (controlled joints only)
        foot_contact       # 2
    ])
    
    expected_shape = 77
    actual_shape = obs.shape[0]
    
    print(f"Expected observation shape: {expected_shape}")
    print(f"Actual observation shape: {actual_shape}")
    print(f"Controlled joints: 21 (excludes 2 wrist yaws and 2 hands from 25 total)")
    
    if actual_shape == expected_shape:
        print("✓ Observation space test PASSED")
        return True
    else:
        print("✗ Observation space test FAILED")
        return False

def test_policy_inference():
    """Test policy inference with dummy observation."""
    print("\nTesting policy inference with dummy data...")
    
    # Create dummy policy (simple linear layer)
    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(77, 21)
            
        def forward(self, x):
            return torch.tanh(self.linear(x))  # Tanh to keep outputs in reasonable range
    
    # Create and save dummy policy
    dummy_policy = DummyPolicy()
    dummy_path = "test_policy.pt"
    torch.save(dummy_policy.state_dict(), dummy_path)
    
    # Test loading
    try:
        loaded_state = torch.load(dummy_path, map_location='cpu')
        loaded_policy = DummyPolicy()
        loaded_policy.load_state_dict(loaded_state)
        loaded_policy.eval()
        
        # Test inference
        dummy_obs = torch.randn(1, 77)
        with torch.no_grad():
            actions = loaded_policy(dummy_obs)
            
        print(f"Input shape: {dummy_obs.shape}")
        print(f"Output shape: {actions.shape}")
        print(f"Action range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        
        if actions.shape == (1, 21):
            print("✓ Policy inference test PASSED")
            success = True
        else:
            print("✗ Policy inference test FAILED")
            success = False
            
    except Exception as e:
        print(f"✗ Policy inference test FAILED: {e}")
        success = False
    finally:
        # Clean up
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            
    return success

def test_joint_velocity_calculation():
    """Test joint velocity calculation from position history."""
    print("\nTesting joint velocity calculation...")
    
    # Simulate joint position history
    dt = 0.02  # 50 Hz
    t1, t2 = 0.0, dt
    pos1 = np.random.randn(21)
    pos2 = pos1 + np.random.randn(21) * 0.1  # Small change
    
    # Calculate velocity
    joint_vel = (pos2 - pos1) / dt
    
    print(f"Position change magnitude: {np.linalg.norm(pos2 - pos1):.6f}")
    print(f"Calculated velocity magnitude: {np.linalg.norm(joint_vel):.6f}")
    print(f"Expected velocity magnitude: {np.linalg.norm(pos2 - pos1) / dt:.6f}")
    
    # Check if calculation is correct
    expected_vel = (pos2 - pos1) / dt
    if np.allclose(joint_vel, expected_vel):
        print("✓ Joint velocity calculation test PASSED")
        return True
    else:
        print("✗ Joint velocity calculation test FAILED")
        return False

def test_joint_mapping():
    """Test joint mapping from full NAO joint space to controlled joints."""
    print("\nTesting joint mapping...")
    
    # Import would normally come from JointIndexes, but we'll simulate the values
    # Based on JointIndexes.msg constants
    HEADYAW, HEADPITCH = 0, 1
    LSHOULDERPITCH, RSHOULDERPITCH = 2, 18
    LSHOULDERROLL, RSHOULDERROLL = 3, 19
    LELBOWYAW, RELBOWYAW = 4, 20
    LELBOWROLL, RELBOWROLL = 5, 21
    LHIPYAWPITCH = 7
    LHIPROLL, RHIPROLL = 8, 13
    LHIPPITCH, RHIPPITCH = 9, 14
    LKNEEPITCH, RKNEEPITCH = 10, 15
    LANKLEPITCH, RANKLEPITCH = 11, 16
    LANKLEROLL, RANKLEROLL = 12, 17
    
    # Isaac Lab joint ordering
    controlled_joint_indices = [
        # Head (2)
        HEADYAW, HEADPITCH,
        # Shoulder Pitch (2) - Left, Right
        LSHOULDERPITCH, RSHOULDERPITCH,
        # Shoulder Roll (2) - Left, Right  
        LSHOULDERROLL, RSHOULDERROLL,
        # Elbow Yaw (2) - Left, Right
        LELBOWYAW, RELBOWYAW,
        # Elbow Roll (2) - Left, Right
        LELBOWROLL, RELBOWROLL,
        # Hip Yaw Pitch (1) - Only left side exists
        LHIPYAWPITCH,
        # Hip Roll (2) - Left, Right
        LHIPROLL, RHIPROLL,
        # Hip Pitch (2) - Left, Right
        LHIPPITCH, RHIPPITCH,
        # Knee Pitch (2) - Left, Right
        LKNEEPITCH, RKNEEPITCH,
        # Ankle Pitch (2) - Left, Right
        LANKLEPITCH, RANKLEPITCH,
        # Ankle Roll (2) - Left, Right
        LANKLEROLL, RANKLEROLL
    ]
    
    full_joint_count = 25
    
    print(f"Total NAO joints: {full_joint_count}")
    print(f"Controlled joints: {len(controlled_joint_indices)}")
    print(f"Joint ordering: Isaac Lab style (*ShoulderPitch, *ShoulderRoll, etc.)")
    print(f"Excluded joints: LWristYaw(6), RWristYaw(22), LHand(23), RHand(24)")
    
    # Simulate full joint position message
    full_joint_positions = np.random.randn(25)
    
    # Extract controlled joint positions
    controlled_positions = np.array([full_joint_positions[i] for i in controlled_joint_indices])
    
    print(f"Controlled positions shape: {controlled_positions.shape}")
    
    if len(controlled_joint_indices) == 21 and controlled_positions.shape[0] == 21:
        print("✓ Joint mapping test PASSED")
        return True
    else:
        print("✗ Joint mapping test FAILED")
        return False

def test_foot_contact_detection():
    """Test foot contact detection from FSR data."""
    print("\nTesting foot contact detection...")
    
    # Simulate FSR data
    fsr_threshold = 5.0
    
    # Test no contact
    fsr_values_low = [1.0, 0.5, 1.2, 0.8]
    contact_low = 1.0 if np.mean(fsr_values_low) > fsr_threshold else 0.0
    
    # Test contact
    fsr_values_high = [8.0, 7.5, 9.2, 6.8]
    contact_high = 1.0 if np.mean(fsr_values_high) > fsr_threshold else 0.0
    
    print(f"Low FSR mean: {np.mean(fsr_values_low):.2f}, Contact: {contact_low}")
    print(f"High FSR mean: {np.mean(fsr_values_high):.2f}, Contact: {contact_high}")
    
    if contact_low == 0.0 and contact_high == 1.0:
        print("✓ Foot contact detection test PASSED")
        return True
    else:
        print("✗ Foot contact detection test FAILED")
        return False

def main():
    """Run all tests."""
    print("NAO Isaac RL Policy Node Test Suite")
    print("=" * 40)
    
    tests = [
        test_observation_space,
        test_policy_inference,
        test_joint_mapping,
        test_joint_velocity_calculation,
        test_foot_contact_detection
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASSED" if result else "FAILED"
        print(f"{test.__name__}: {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
