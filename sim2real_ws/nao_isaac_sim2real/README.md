# NAO Isaac Sim2Real RL Policy Node

This ROS2 node loads a PyTorch reinforcement learning policy and applies it to control a real NAO robot. The node bridges the gap between simulation (Isaac Lab) and real robot control.

## Features

- Loads PyTorch (.pt) policy checkpoints
- Constructs 77-dimensional observation space matching Isaac Lab training
- Real-time policy inference and joint control
- Sensor fusion for state estimation
- Configurable control frequency

## Observation Space (77 dimensions)

The node constructs observations matching the training environment. **Note**: The robot has 25 total joints, but only 21 are controlled (excluding left/right wrist yaw and hands):

| Index | Name | Shape | Source | Notes |
|-------|------|-------|--------|-------|
| 0-2 | base_lin_vel | (3,) | Accelerometer integration | Velocity calculated by integrating linear acceleration |
| 3-5 | base_ang_vel | (3,) | Gyroscope (world frame) | Direct angular velocity measurement |
| 6-8 | projected_gravity | (3,) | Accelerometer (low-pass filtered) | Gravity vector estimation |
| 9-11 | velocity_commands | (3,) | /cmd_vel topic | Commanded velocities |
| 12-32 | joint_pos | (21,) | Joint position sensors | Only controlled joints (excludes wrist yaws & hands) |
| 33-53 | joint_vel | (21,) | Calculated from joint positions | Finite difference from position history |
| 54-74 | actions | (21,) | Previous policy actions | Last commanded joint positions |
| 75-76 | foot_contact | (2,) | FSR sensors (running average) | Binary contact detection per foot |

### Controlled Joints (21 total)
Based on Isaac Lab joint ordering and `JointIndexes.msg`:
1. **Head** (2): HeadYaw, HeadPitch
2. **Shoulder Pitch** (2): LShoulderPitch, RShoulderPitch  
3. **Shoulder Roll** (2): LShoulderRoll, RShoulderRoll
4. **Elbow Yaw** (2): LElbowYaw, RElbowYaw
5. **Elbow Roll** (2): LElbowRoll, RElbowRoll
6. **Hip Yaw Pitch** (1): LHipYawPitch (only left side exists)
7. **Hip Roll** (2): LHipRoll, RHipRoll
8. **Hip Pitch** (2): LHipPitch, RHipPitch
9. **Knee Pitch** (2): LKneePitch, RKneePitch
10. **Ankle Pitch** (2): LAnklePitch, RAnklePitch
11. **Ankle Roll** (2): LAnkleRoll, RAnkleRoll

**Excluded joints**: LWristYaw, RWristYaw, LHand, RHand

## Installation

1. Make sure you have the nao_lola packages built and sourced
2. Ensure PyTorch is installed in your ROS2 environment:
   ```bash
   pip install torch torchvision
   ```
3. Build the package:
   ```bash
   cd your_workspace
   colcon build --packages-select nao_isaac_sim2real
   source install/setup.bash
   ```

## Usage

### Method 1: Direct execution
```bash
ros2 run nao_isaac_sim2real nao_isaac_node --ros-args -p pt_filepath:="/path/to/your/policy.pt" -p control_frequency:=50.0
```

### Method 2: Using launch file
```bash
ros2 launch nao_isaac_sim2real launch_nao_isaac_node.py pt_filepath:="/path/to/your/policy.pt" control_frequency:=50.0
```

## Parameters

- `pt_filepath` (string): Path to the PyTorch policy checkpoint file
- `control_frequency` (float): Control loop frequency in Hz (default: 50.0)

## Topics

### Subscribed Topics
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands for the robot
- `/sensors/joint_positions` (nao_lola_sensor_msgs/JointPositions): Joint position feedback
- `/sensors/fsr` (nao_lola_sensor_msgs/FSR): Force sensitive resistor data
- `/sensors/accelerometer` (nao_lola_sensor_msgs/Accelerometer): Accelerometer data
- `/sensors/gyroscope` (nao_lola_sensor_msgs/Gyroscope): Gyroscope data

### Published Topics
- `/effectors/joint_positions` (nao_lola_command_msgs/JointPositions): Joint position commands
- `/effectors/joint_stiffnesses` (nao_lola_command_msgs/JointStiffnesses): Joint stiffness commands

## Policy Checkpoint Format

The node supports various PyTorch checkpoint formats:
- Direct policy models
- Checkpoints with 'model', 'policy', or 'agent' keys
- Both CPU and GPU checkpoints (automatically mapped to available device)

## State Estimation Notes

### Base Velocity Estimation
- Linear velocity is estimated from accelerometer readings minus gravity
- Angular velocity comes directly from gyroscope
- Both are assumed to be in world frame (may need calibration for your setup)

### Joint Velocity Calculation
- Calculated using finite differences from joint position history
- Uses a rolling window of 5 samples for smoothing
- Adjustable history length for better filtering

### Foot Contact Detection
- Uses running average of FSR values over 10 samples
- Configurable contact threshold (default: 5.0)
- Binary contact state (0.0 or 1.0) for each foot

## Safety Considerations

1. **Always test with low stiffness first** - The default stiffness is set to 0.8 (moderate)
2. **Monitor joint limits** - Ensure your policy respects NAO's joint limits
3. **Emergency stop** - Keep the robot's chest button accessible for emergency stops
4. **Gradual deployment** - Start with simple movements before complex behaviors

## Troubleshooting

### Policy Loading Issues
- Check file path and permissions
- Verify PyTorch version compatibility
- Ensure checkpoint contains policy in expected format

### Sensor Issues
- Verify nao_lola nodes are running and publishing sensor data
- Check topic names match your nao_lola configuration
- Monitor sensor data rates with `ros2 topic hz`

### Control Issues
- Check control frequency vs sensor update rates
- Verify observation vector construction
- Monitor for joint limit violations

## Example Usage

1. Start the nao_lola nodes:
   ```bash
   ros2 launch nao_lola nao_lola.launch.py
   ```

2. Run the RL policy node:
   ```bash
   ros2 launch nao_isaac_sim2real launch_nao_isaac_node.py pt_filepath:="/path/to/your/trained_policy.pt"
   ```

3. Send velocity commands:
   ```bash
   ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
   ```

## Contributing

When modifying the observation space or policy interface, ensure compatibility with your Isaac Lab training setup. The observation indices and scaling should match exactly between simulation and real robot.
