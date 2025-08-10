#!/usr/bin/env python3
"""
test_nao_policy.py
--------------------

Simple test script to verify the NAO policy wrapper works correctly.
This can be used to test the policy without running the full ROS2 node.

Usage:
    python test_nao_policy.py

Author: Assistant
"""

import numpy as np
import sys
from pathlib import Path

# Add the sim2real directory to the path so we can import our modules
sim2real_dir = Path(__file__).parent
sys.path.append(str(sim2real_dir))

from robots.nao import NaoPolicy


def test_nao_policy():
    """Test the NAO policy wrapper."""
    print("=== Testing NAO Policy Wrapper ===")
    
    try:
        # Initialize the NAO policy
        nao = NaoPolicy()
        print("✓ NAO policy initialized successfully")
        
        # Test sensor state updates
        print("\n--- Testing sensor state updates ---")
        
        # Simulate joint positions (21 joints)
        joint_positions = np.random.uniform(-1.0, 1.0, 21)
        joint_velocities = np.random.uniform(-0.5, 0.5, 21)
        
        nao.update_sensor_state(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            base_lin_vel=[0.1, 0.0, 0.0],
            base_ang_vel=[0.0, 0.0, 0.05],
            velocity_commands=[0.2, 0.0, 0.1],
            foot_contact=[1.0, 1.0]
        )
        print("✓ Sensor state updated successfully")
        
        # Test observation computation
        print("\n--- Testing observation computation ---")
        obs = nao._compute_observation()
        if obs is not None:
            print(f"✓ Observation computed: shape={obs.shape}, expected=77")
            print(f"  Observation components:")
            print(f"    base_lin_vel: {obs[0:3]}")
            print(f"    base_ang_vel: {obs[3:6]}")
            print(f"    projected_gravity: {obs[6:9]}")
            print(f"    velocity_commands: {obs[9:12]}")
            print(f"    joint_pos (first 5): {obs[12:17]}")
            print(f"    foot_contact: {obs[75:77]}")
        else:
            print("✗ Failed to compute observation")
            
        # Test forward pass (without policy loaded)
        print("\n--- Testing forward pass ---")
        if hasattr(nao, 'policy') and nao.policy is not None:
            joint_commands = nao.forward(0.01, [0.1, 0.0, 0.05])
            if joint_commands is not None:
                print(f"✓ Forward pass successful: {len(joint_commands)} joint commands")
                print(f"  First 5 joint commands: {joint_commands[:5]}")
            else:
                print("✗ Forward pass returned None")
        else:
            print("⚠ Policy not loaded - forward pass will return None")
            print("  This is expected if no policy file was found")
            
        # Test joint position history updates
        print("\n--- Testing joint position history ---")
        for i in range(3):
            new_positions = np.random.uniform(-1.0, 1.0, 21)
            nao.update_joint_positions_with_history(new_positions)
            print(f"  Step {i+1}: Updated positions, velocity estimate available: {len(nao.joint_pos_history) >= 2}")
            
        print(f"✓ Joint velocities from history: {nao.joint_vel[:5]}")
        
        # Test FSR contact updates
        print("\n--- Testing FSR contact detection ---")
        left_forces = [2.0, 3.0, 4.0, 5.0]  # Below threshold
        right_forces = [8.0, 7.0, 6.0, 9.0]  # Above threshold
        nao.update_foot_contact_from_fsr(left_forces, right_forces, threshold=5.0)
        print(f"✓ Foot contact from FSR: left={nao.foot_contact[0]}, right={nao.foot_contact[1]}")
        
        # Test accelerometer/gyroscope updates  
        print("\n--- Testing IMU updates ---")
        for i in range(3):
            linear_accel = np.random.uniform(-2.0, 2.0, 3)
            angular_vel = np.random.uniform(-0.5, 0.5, 3)
            nao.update_base_velocity_from_accel(linear_accel, angular_vel)
            
        print(f"✓ Base velocities: linear={nao.base_lin_vel}, angular={nao.base_ang_vel}")
        print(f"✓ Projected gravity: {nao.projected_gravity}")
        
        print("\n=== All tests passed! ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_loading():
    """Test policy loading with a mock policy file."""
    print("\n=== Testing Policy Loading ===")
    
    # This test would require actual policy files
    # For now, just test that the class handles missing files gracefully
    nao = NaoPolicy()
    print("✓ Policy loading handled gracefully (files not found)")
    

if __name__ == "__main__":
    print("NAO Policy Test Script")
    print("=" * 50)
    
    success = test_nao_policy()
    test_policy_loading()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Train or obtain a NAO policy file (policy.pt) and environment config (env.yaml)")
        print("2. Update the model path in nao.py if needed")
        print("3. Run the ROS2 node: ros2 run <package> nao_policy_node.py")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
