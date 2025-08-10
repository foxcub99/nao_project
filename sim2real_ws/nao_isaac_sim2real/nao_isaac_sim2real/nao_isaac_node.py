#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import math
import time
import os
import numpy as np
import torch
from collections import deque
from nao_lola_command_msgs.msg import JointPositions, JointStiffnesses, JointIndexes
from nao_lola_sensor_msgs.msg import JointPositions as SensorJointPositions, FSR, Accelerometer, Gyroscope
from geometry_msgs.msg import Twist


class NaoIsaacNode(Node):
    def __init__(self):
        super().__init__("nao_isaac_node")

        # parameters
        self.declare_parameter("pt_filepath", "")
        self.declare_parameter("control_frequency", 83.0)  # Hz
        self.pt_filepath = self.get_parameter("pt_filepath").get_parameter_value().string_value
        self.control_frequency = self.get_parameter("control_frequency").get_parameter_value().double_value
        
        # Load the PyTorch policy
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_policy()

        # Publishers
        self.joint_positions_pub = self.create_publisher(JointPositions, "/effectors/joint_positions", 10)
        self.joint_stiffnesses_pub = self.create_publisher(JointStiffnesses, "/effectors/joint_stiffnesses", 10)

        # Joint mapping - based on Isaac Lab joint ordering
        self.controlled_joint_indices = [
            JointIndexes.HEADYAW, JointIndexes.HEADPITCH,
            JointIndexes.LSHOULDERPITCH, JointIndexes.RSHOULDERPITCH,
            JointIndexes.LSHOULDERROLL, JointIndexes.RSHOULDERROLL,
            JointIndexes.LELBOWYAW, JointIndexes.RELBOWYAW,
            JointIndexes.LELBOWROLL, JointIndexes.RELBOWROLL,
            JointIndexes.LHIPYAWPITCH,
            JointIndexes.LHIPROLL, JointIndexes.RHIPROLL,
            JointIndexes.LHIPPITCH, JointIndexes.RHIPPITCH,
            JointIndexes.LKNEEPITCH, JointIndexes.RKNEEPITCH,
            JointIndexes.LANKLEPITCH, JointIndexes.RANKLEPITCH,
            JointIndexes.LANKLEROLL, JointIndexes.RANKLEROLL
        ]
        
        # Initialize observation variables
        self.base_lin_vel = np.zeros(3)  # Linear velocity in world frame
        self.base_ang_vel = np.zeros(3)  # Angular velocity in world frame
        self.projected_gravity = np.array([0.0, 0.0, -9.81])  # Gravity vector
        self.velocity_commands = np.zeros(3)  # Command velocities
        self.joint_pos = np.zeros(21)  # Current joint positions (only controlled joints)
        self.joint_vel = np.zeros(21)  # Joint velocities
        self.actions = np.zeros(21)  # Previous actions
        self.foot_contact = np.zeros(2)  # Foot contact states
        
        # History for velocity calculation and filtering
        self.joint_pos_history = deque(maxlen=5)  # Store (time, joint_positions) tuples
        self.accel_history = deque(maxlen=5)  # Store (time, acceleration) tuples for velocity calculation
        self.fsr_history = [deque(maxlen=10), deque(maxlen=10)]  # Store FSR history for contact detection
        self.last_time = time.time()
        
        # Subscribers
        self.command_sub = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )
        self.joint_positions_sub = self.create_subscription(
            SensorJointPositions, "/sensors/joint_positions", self.joint_positions_callback, 10
        )
        self.fsr_sub = self.create_subscription(FSR, "/sensors/fsr", self.fsr_callback, 10)
        self.accelerometer_sub = self.create_subscription(
            Accelerometer, "/sensors/accelerometer", self.accelerometer_callback, 10
        )
        self.gyroscope_sub = self.create_subscription(Gyroscope, "/sensors/gyroscope", self.gyroscope_callback, 10)
        
        # Control timer
        self.control_timer = self.create_timer(1.0 / self.control_frequency, self.control_loop)
        
        # Initialize policy ready flag
        self.sensors_ready = False
        
        self.get_logger().info(f"NAO Isaac Node initialized with policy: {self.pt_filepath}")

    def load_policy(self):
        """Load the PyTorch policy from the checkpoint file."""
        if not self.pt_filepath or not os.path.exists(self.pt_filepath):
            self.get_logger().error(f"Policy file not found: {self.pt_filepath}")
            return
            
        try:
            checkpoint = torch.load(self.pt_filepath, map_location=self.device)
            
            # Extract the policy (this may vary depending on the RL framework used)
            if 'model' in checkpoint:
                self.policy = checkpoint['model']
            elif 'policy' in checkpoint:
                self.policy = checkpoint['policy']
            elif 'agent' in checkpoint:
                self.policy = checkpoint['agent']
            else:
                # Assume the checkpoint is the policy itself
                self.policy = checkpoint
                
            # Set to evaluation mode
            if hasattr(self.policy, 'eval'):
                self.policy.eval()
                
            self.get_logger().info(f"Successfully loaded policy from: {self.pt_filepath}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {str(e)}")

    def cmd_vel_callback(self, msg):
        """Handle velocity commands."""
        self.velocity_commands[0] = msg.linear.x
        self.velocity_commands[1] = msg.linear.y
        self.velocity_commands[2] = msg.angular.z

    def joint_positions_callback(self, msg):
        """Handle joint position updates."""
        current_time = time.time()
        
        # Extract only the controlled joints from the full message
        if len(msg.positions) >= JointIndexes.NUMJOINTS:
            controlled_positions = np.array([msg.positions[i] for i in self.controlled_joint_indices])
            self.joint_pos = controlled_positions
            self.joint_pos_history.append((current_time, self.joint_pos.copy())) # future TODO A timestamp would be better
            
            # Calculate joint velocities
            if len(self.joint_pos_history) >= 2:
                dt = self.joint_pos_history[-1][0] - self.joint_pos_history[-2][0] # last two positions 
                if dt > 0:
                    self.joint_vel = (self.joint_pos_history[-1][1] - self.joint_pos_history[-2][1]) / dt
        else:
            self.get_logger().warn(f"Received {len(msg.positions)} joint positions, expected at least 25")
                    
        self.sensors_ready = True

    def fsr_callback(self, msg):
        """Handle Force Sensitive Resistor (FSR) updates for foot contact."""
        left_foot_force = sum([msg.l_foot_front_left, msg.l_foot_front_right, 
                              msg.l_foot_back_left, msg.l_foot_back_right])
        right_foot_force = sum([msg.r_foot_front_left, msg.r_foot_front_right, 
                               msg.r_foot_back_left, msg.r_foot_back_right])
        
        self.fsr_history[0].append(left_foot_force)
        self.fsr_history[1].append(right_foot_force)
        
        contact_threshold = 5.0 # TODO check 5 works
        self.foot_contact[0] = 1.0 if np.mean(self.fsr_history[0]) > contact_threshold else 0.0
        self.foot_contact[1] = 1.0 if np.mean(self.fsr_history[1]) > contact_threshold else 0.0

    def accelerometer_callback(self, msg):
        """Handle accelerometer updates."""
        current_time = time.time()
        
        accel_raw = np.array([msg.x, msg.y, msg.z])
        
        # Estimate gravity vector using low-pass filter (simple approach)
        # In practice, you might want to use a complementary filter with gyroscope 
        alpha = 0.98  # Low-pass filter coefficient
        accel_magnitude = np.linalg.norm(accel_raw)
        if accel_magnitude > 0:
            # Update gravity estimate with low-pass filter
            gravity_estimate = -accel_raw / accel_magnitude * 9.81
            self.projected_gravity = alpha * self.projected_gravity + (1 - alpha) * gravity_estimate
        
        # Remove gravity to get linear acceleration in world frame
        linear_accel = accel_raw - self.projected_gravity
        
        # Add to history for velocity calculation
        self.accel_history.append((current_time, linear_accel.copy()))
        
        # Calculate linear velocity by integrating acceleration
        if len(self.accel_history) >= 2:
            # Simple integration using trapezoidal rule
            dt = self.accel_history[-1][0] - self.accel_history[-2][0]
            if dt > 0:
                avg_accel = (self.accel_history[-1][1] + self.accel_history[-2][1]) / 2.0
                self.base_lin_vel += avg_accel * dt
                
                # Apply some damping to prevent velocity drift # TODO tune
                damping_factor = 0.95
                self.base_lin_vel *= damping_factor

    def gyroscope_callback(self, msg):
        """Handle gyroscope updates."""
        self.base_ang_vel = np.array([msg.x, msg.y, msg.z])

    def get_observation(self):
        """Construct the observation vector for the policy."""
        obs = np.concatenate([
            self.base_lin_vel,      # 3
            self.base_ang_vel,      # 3
            self.projected_gravity, # 3
            self.velocity_commands, # 3
            self.joint_pos,         # 21
            self.joint_vel,         # 21
            self.actions,           # 21
            self.foot_contact       # 2
        ])
        
        return obs

    def control_loop(self):
        """Main control loop that runs the policy and publishes actions."""
        if not self.sensors_ready or self.policy is None:
            return
            
        try:
            obs = self.get_observation()
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # TODO once I understand the policy, can wipe out a bunch of these ifs
            # Run inference
            with torch.no_grad():
                if hasattr(self.policy, 'act'):
                    # If the policy has an 'act' method (like in some RL frameworks)
                    action_tensor = self.policy.act(obs_tensor)
                    if isinstance(action_tensor, tuple):
                        action_tensor = action_tensor[0]  # Take first element if tuple
                elif hasattr(self.policy, 'forward'):
                    # If it's a standard neural network
                    action_tensor = self.policy.forward(obs_tensor)
                else:
                    # Try calling it directly
                    action_tensor = self.policy(obs_tensor)
                    
                # Convert back to numpy
                if isinstance(action_tensor, torch.Tensor):
                    actions = action_tensor.cpu().numpy().flatten()
                else:
                    actions = np.array(action_tensor).flatten()
                    
            # Ensure we have the right number of actions (21 joints)
            if len(actions) >= 21:
                self.actions = actions[:21]
            else:
                self.get_logger().warn(f"Policy output has {len(actions)} actions, expected 21")
                return
                
            self.publish_joint_commands()
            
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {str(e)}")

    def publish_joint_commands(self):
        """Publish joint position commands."""
        joint_pos_msg = JointPositions()
        joint_pos_msg.indexes = self.controlled_joint_indices
        joint_pos_msg.positions = self.actions.tolist()
        
        joint_stiff_msg = JointStiffnesses()
        joint_stiff_msg.indexes = self.controlled_joint_indices
        joint_stiff_msg.stiffnesses = [1.0] * 21
        
        self.joint_positions_pub.publish(joint_pos_msg)
        self.joint_stiffnesses_pub.publish(joint_stiff_msg)


def main(args=None):
    rclpy.init(args=args)
    node = NaoIsaacNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down NAO Isaac Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
