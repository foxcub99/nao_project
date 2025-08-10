#!/usr/bin/env python3
"""
nao.py
--------------------

Thin wrapper around a pre-trained NAO policy.
Extends `PolicyController` with:

* State update via `update_sensor_state()` for NAO's complex sensor inputs
* Forward pass (`forward`) that returns target joint-position commands
  every call, computing a new action every ``decimation`` steps.

Manages the complex observation space for NAO including:
- Base linear/angular velocity
- Projected gravity vector
- Velocity commands 
- Joint positions and velocities
- Previous actions
- Foot contact states

Author: Assistant (based on nao_isaac_node.py and gen3.py patterns)
"""

from pathlib import Path
import numpy as np
from collections import deque
import time

from skrl.utils.runner.torch import Runner




class NaoPolicy():
    """Policy controller for NAO using a pre-trained policy model."""

    def __init__(self) -> None:
        """Initialize the NaoPolicy instance."""
        super().__init__()
        
        # NAO joint names in the order expected by the policy (21 joints)
        self.dof_names = [
            "HeadYaw", "HeadPitch",
            "LShoulderPitch", "RShoulderPitch",
            "LShoulderRoll", "RShoulderRoll", 
            "LElbowYaw", "RElbowYaw",
            "LElbowRoll", "RElbowRoll",
            "LHipYawPitch",
            "LHipRoll", "RHipRoll",
            "LHipPitch", "RHipPitch", 
            "LKneePitch", "RKneePitch",
            "LAnklePitch", "RAnklePitch",
            "LAnkleRoll", "RAnkleRoll"
        ]
        
        # Load the pre-trained policy model and environment configuration
        # Update these paths as needed for your NAO model
        repo_root = Path(__file__).resolve().parents[3]
        model_dir = repo_root / "outputs" / "nao"  # Adjust path as needed
        
        # Try to load policy if files exist, otherwise will need to be loaded later
        # policy_path = model_dir / "policy.pt"
        policy_path = "/mnt/c/Users/reill/lab/nao_project/logs/skrl/nao_flat/2025-08-09_18-29-54_ppo_torch/checkpoints/best_agent.pt"
        # env_path = model_dir / "env.yaml"
        env_path = "/mnt/c/Users/reill/lab/nao_project/logs/skrl/nao_flat/2025-08-09_18-29-54_ppo_torch/params/env.yaml"

        # agent_path = "/mnt/c/Users/reill/lab/nao_project/source/nao_project/nao_project/tasks/manager_based/nao_project/agents/skrl_flat_ppo_cfg_long_long.yaml"
        policy = SharedPolicy(obs_size=77, act_size=21)
        policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
        policy.eval()
        # env = wrap_env(env, wrapper="isaaclab")

        self.load_policy(policy, env_path)
        # if policy_path.exists() and env_path.exists():
        # else:
        #     print(f"Warning: Policy files not found at {model_dir}")
        #     print("Policy will need to be loaded manually using load_policy() method")

        self._action_scale = 1.0  # May need tuning for NAO
        self._previous_action = np.zeros(21)
        self._policy_counter = 0

        # NAO sensor state variables
        self.base_lin_vel = np.zeros(3)      # Linear velocity in world frame
        self.base_ang_vel = np.zeros(3)      # Angular velocity in world frame  
        self.projected_gravity = np.array([0.0, 0.0, -9.81])  # Gravity vector
        self.velocity_commands = np.zeros(3)  # Command velocities [vx, vy, wz]
        self.joint_pos = np.zeros(21)        # Current joint positions
        self.joint_vel = np.zeros(21)        # Current joint velocities
        self.foot_contact = np.zeros(2)      # Foot contact states [left, right]
        
        # History for velocity calculation and filtering
        self.joint_pos_history = deque(maxlen=5)
        self.accel_history = deque(maxlen=5)
        self.fsr_history = [deque(maxlen=10), deque(maxlen=10)]  # [left, right]
        
        # State tracking
        self.has_sensor_data = False
        
    def update_sensor_state(self, joint_positions=None, joint_velocities=None, 
                           base_lin_vel=None, base_ang_vel=None, projected_gravity=None,
                           velocity_commands=None, foot_contact=None):
        """
        Update the current sensor state for NAO.
        
        Args:
            joint_positions: Array of 21 joint positions
            joint_velocities: Array of 21 joint velocities
            base_lin_vel: Linear velocity in world frame [3]
            base_ang_vel: Angular velocity in world frame [3] 
            projected_gravity: Gravity vector [3]
            velocity_commands: Velocity commands [3]
            foot_contact: Foot contact states [2]
        """
        if joint_positions is not None:
            self.joint_pos = np.array(joint_positions[:21], dtype=np.float32)
        if joint_velocities is not None:
            self.joint_vel = np.array(joint_velocities[:21], dtype=np.float32)
        if base_lin_vel is not None:
            self.base_lin_vel = np.array(base_lin_vel, dtype=np.float32)
        if base_ang_vel is not None:
            self.base_ang_vel = np.array(base_ang_vel, dtype=np.float32)
        if projected_gravity is not None:
            self.projected_gravity = np.array(projected_gravity, dtype=np.float32)
        if velocity_commands is not None:
            self.velocity_commands = np.array(velocity_commands, dtype=np.float32)
        if foot_contact is not None:
            self.foot_contact = np.array(foot_contact, dtype=np.float32)
            
        self.has_sensor_data = True

    def update_joint_positions_with_history(self, joint_positions):
        """
        Update joint positions and calculate velocities from history.
        
        Args:
            joint_positions: Array of 21 joint positions
        """
        current_time = time.time()
        self.joint_pos = np.array(joint_positions[:21], dtype=np.float32)
        self.joint_pos_history.append((current_time, self.joint_pos.copy()))
        
        # Calculate joint velocities from history
        if len(self.joint_pos_history) >= 2:
            dt = self.joint_pos_history[-1][0] - self.joint_pos_history[-2][0]
            if dt > 0:
                self.joint_vel = (self.joint_pos_history[-1][1] - self.joint_pos_history[-2][1]) / dt
                
        self.has_sensor_data = True

    def update_foot_contact_from_fsr(self, left_foot_forces, right_foot_forces, threshold=5.0):
        """
        Update foot contact states from Force Sensitive Resistor data.
        
        Args:
            left_foot_forces: Array of left foot FSR readings
            right_foot_forces: Array of right foot FSR readings  
            threshold: Contact detection threshold
        """
        left_total = np.sum(left_foot_forces)
        right_total = np.sum(right_foot_forces)
        
        self.fsr_history[0].append(left_total)
        self.fsr_history[1].append(right_total)
        
        # Use moving average for contact detection
        self.foot_contact[0] = 1.0 if np.mean(self.fsr_history[0]) > threshold else 0.0
        self.foot_contact[1] = 1.0 if np.mean(self.fsr_history[1]) > threshold else 0.0

    def update_base_velocity_from_accel(self, linear_accel, angular_vel, alpha=0.98, damping=0.95):
        """
        Update base velocities from accelerometer and gyroscope data.
        
        Args:
            linear_accel: Linear acceleration [3]
            angular_vel: Angular velocity [3]
            alpha: Low-pass filter coefficient for gravity estimation
            damping: Damping factor to prevent velocity drift
        """
        current_time = time.time()
        accel_raw = np.array(linear_accel, dtype=np.float32)
        
        # Update angular velocity directly from gyroscope
        self.base_ang_vel = np.array(angular_vel, dtype=np.float32)
        
        # Estimate gravity vector using low-pass filter
        accel_magnitude = np.linalg.norm(accel_raw)
        if accel_magnitude > 0:
            gravity_estimate = -accel_raw / accel_magnitude * 9.81
            self.projected_gravity = alpha * self.projected_gravity + (1 - alpha) * gravity_estimate
        
        # Remove gravity to get linear acceleration
        linear_accel_corrected = accel_raw - self.projected_gravity
        
        # Add to history for velocity integration
        self.accel_history.append((current_time, linear_accel_corrected.copy()))
        
        # Calculate linear velocity by integrating acceleration
        if len(self.accel_history) >= 2:
            dt = self.accel_history[-1][0] - self.accel_history[-2][0]
            if dt > 0:
                avg_accel = (self.accel_history[-1][1] + self.accel_history[-2][1]) / 2.0
                self.base_lin_vel += avg_accel * dt
                # Apply damping to prevent drift
                self.base_lin_vel *= damping

    def _compute_observation(self) -> np.ndarray:
        """
        Compute the observation vector for the NAO policy network.
        
        Expected observation structure (77 elements):
        - base_lin_vel: 3
        - base_ang_vel: 3  
        - projected_gravity: 3
        - velocity_commands: 3
        - joint_pos: 21
        - joint_vel: 21
        - previous_actions: 21
        - foot_contact: 2
        
        Returns:
            An observation vector if sensor data is available, otherwise None.
        """
        if not self.has_sensor_data:
            return None
            
        obs = np.concatenate([
            self.base_lin_vel,      # 3
            self.base_ang_vel,      # 3
            self.projected_gravity, # 3
            self.velocity_commands, # 3
            self.joint_pos,         # 21
            self.joint_vel,         # 21
            self._previous_action,  # 21
            self.foot_contact       # 2
        ])
        
        return obs

    def forward(self, dt: float, velocity_command: np.ndarray = None) -> np.ndarray:
        """
        Compute the next joint positions based on the NAO policy.
        
        Args:
            dt: Time step for the forward pass.
            velocity_command: Velocity command [vx, vy, wz]. If None, uses current commands.
            
        Returns:
            The computed joint positions if sensor data is available, otherwise None.
        """
        if not self.has_sensor_data:
            return None
            
        # Update velocity commands if provided
        if velocity_command is not None:
            self.velocity_commands = np.array(velocity_command[:3], dtype=np.float32)

        # Compute new action every decimation steps
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation()
            if obs is None:
                return None
                
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

            # Debug logging (can be enabled for debugging)
            if False:  # Set to True for debugging
                print("\n=== NAO Policy Step ===")
                print(f"{'Velocity Command:':<20} {np.round(self.velocity_commands, 4)}")
                print("--- Observation ---")
                print(f"{'Base Lin Vel:':<20} {np.round(self.base_lin_vel, 4)}")
                print(f"{'Base Ang Vel:':<20} {np.round(self.base_ang_vel, 4)}")
                print(f"{'Projected Gravity:':<20} {np.round(self.projected_gravity, 4)}")
                print(f"{'Joint Positions:':<20} {np.round(self.joint_pos[:5], 4)}...")  # First 5 joints
                print(f"{'Joint Velocities:':<20} {np.round(self.joint_vel[:5], 4)}...")
                print(f"{'Foot Contact:':<20} {self.foot_contact}")
                print("--- Action ---")
                print(f"{'Raw Action:':<20} {np.round(self.action[:5], 4)}...")  # First 5 actions

        # Apply action scaling and add to default position
        joint_positions = self.default_pos + (self.action * self._action_scale)
        self._policy_counter += 1
        
        return joint_positions
