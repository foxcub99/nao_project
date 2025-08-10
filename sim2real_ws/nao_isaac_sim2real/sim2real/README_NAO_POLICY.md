# NAO Policy Controller

This directory contains a simplified implementation for controlling NAO robots using trained policies, extracted from the more complex ROS2 package structure.

## Files

### Core Implementation

- **`robots/nao.py`** - NAO policy wrapper class that extends `PolicyController`
- **`nao_policy_node.py`** - ROS2 node that wraps the NAO policy for real robot control  
- **`test_nao_policy.py`** - Test script to verify the policy wrapper works correctly

### Pattern Based On

- **`robots/gen3.py`** - Original Gen3 robot pattern that NAO implementation follows
- **`run_task_reach.py`** - Original ROS2 node pattern that NAO node follows
- **`controllers/policy_controller.py`** - Base class for policy controllers

## Usage

### 1. Test the Policy Wrapper

First, test that the policy wrapper works correctly:

```bash
cd sim2real_ws/nao_isaac_sim2real/sim2real/
python test_nao_policy.py
```

This will test all the sensor state management and observation computation without requiring ROS2 or actual policy files.

### 2. Setup Policy Files

The NAO policy expects these files:
- `policy.pt` - Trained PyTorch policy (TorchScript format)
- `env.yaml` - Environment configuration with robot parameters

Update the path in `robots/nao.py` if your policy files are in a different location:

```python
model_dir = repo_root / "your_model_directory" / "nao"
```

### 3. Run the ROS2 Node

```bash
# Source your ROS2 workspace
cd sim2real_ws
colcon build
source install/setup.bash

# Run the NAO policy node
python sim2real/nao_policy_node.py

# Or with verbose output
python sim2real/nao_policy_node.py --verbose

# Or to suppress error messages  
python sim2real/nao_policy_node.py --fail-quietly
```

### 4. Send Commands

The node subscribes to velocity commands:

```bash
# Send a velocity command
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}'
```

## ROS2 Topics

### Subscribed Topics

- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands
- `/sensors/joint_positions` (nao_lola_sensor_msgs/JointPositions) - Current joint positions
- `/sensors/fsr` (nao_lola_sensor_msgs/FSR) - Foot pressure sensors
- `/sensors/accelerometer` (nao_lola_sensor_msgs/Accelerometer) - Linear acceleration  
- `/sensors/gyroscope` (nao_lola_sensor_msgs/Gyroscope) - Angular velocity

### Published Topics

- `/effectors/joint_positions` (nao_lola_command_msgs/JointPositions) - Joint position commands
- `/effectors/joint_stiffnesses` (nao_lola_command_msgs/JointStiffnesses) - Joint stiffness commands

## Parameters

- `control_frequency` (default: 83.0) - Control loop frequency in Hz
- `default_stiffness` (default: 1.0) - Default joint stiffness
- `contact_threshold` (default: 5.0) - FSR threshold for foot contact detection

## NAO Robot State

The NAO policy manages a complex observation space with 77 elements:

- **Base linear velocity** (3) - Estimated from accelerometer
- **Base angular velocity** (3) - From gyroscope  
- **Projected gravity vector** (3) - Gravity estimation from accelerometer
- **Velocity commands** (3) - Current movement commands [vx, vy, wz]
- **Joint positions** (21) - Current joint positions
- **Joint velocities** (21) - Estimated from position history
- **Previous actions** (21) - Previous policy output
- **Foot contact** (2) - Contact state for left/right foot

## Key Differences from Original nao_isaac_node.py

1. **Simplified architecture** - Removed complex policy loading logic
2. **Cleaner state management** - Better separation of sensor data handling
3. **Modular design** - Policy logic separated from ROS2 node logic
4. **Better error handling** - Graceful handling of missing policy files
5. **Configurable parameters** - ROS2 parameters for tuning
6. **Debug support** - Verbose mode and error suppression options

## Development Notes

- The observation space matches Isaac Lab's NAO environment requirements
- Joint ordering follows the `JointIndexes` constants from `nao_lola_command_msgs`
- Sensor fusion for base velocity estimation uses simple integration with damping
- Foot contact detection uses moving average of FSR readings
- Policy decimation and action scaling are configurable through the environment YAML

## Troubleshooting

- **Import errors**: Normal in VS Code - ROS2 packages aren't available to the Python language server
- **Policy not found**: Update the model path in `nao.py` or place policy files in the expected location
- **Sensor timeouts**: Check that NAO sensor topics are publishing data
- **Joint command issues**: Verify joint index mapping matches your NAO configuration
