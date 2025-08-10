#!/usr/bin/env python3
"""
nao_policy_node.py
--------------------

ROS 2 node that drives a NAO robot with a policy-based controller.

* Subscribes to: 
  - /cmd_vel (geometry_msgs/Twist)
  - /sensors/joint_positions (nao_lola_sensor_msgs/JointPositions)
  - /sensors/fsr (nao_lola_sensor_msgs/FSR)
  - /sensors/accelerometer (nao_lola_sensor_msgs/Accelerometer)  
  - /sensors/gyroscope (nao_lola_sensor_msgs/Gyroscope)
* Publishes to:
  - /effectors/joint_positions (nao_lola_command_msgs/JointPositions)
  - /effectors/joint_stiffnesses (nao_lola_command_msgs/JointStiffnesses)
* Runs at: 83 Hz (default NAO control frequency)

Author: Assistant (based on nao_isaac_node.py and run_task_reach.py patterns)
"""

import numpy as np
import rclpy
from rclpy.node import Node

from nao_lola_command_msgs.msg import JointPositions, JointStiffnesses, JointIndexes
from nao_lola_sensor_msgs.msg import JointPositions as SensorJointPositions, FSR, Accelerometer, Gyroscope
from geometry_msgs.msg import Twist

from robots.nao import NaoPolicy


class NaoPolicyNode(Node):
    """ROS2 node for controlling a NAO robot with a policy-based controller."""
    
    # NAO joint mapping - based on Isaac Lab joint ordering
    CONTROLLED_JOINT_INDICES = [
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
    
    # Joint names for debugging/logging
    JOINT_NAMES = [
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

    def __init__(self, fail_quietly: bool = False, verbose: bool = False):
        """Initialize the NaoPolicyNode."""
        super().__init__('nao_policy_node')
        
        # Parameters
        self.declare_parameter("control_frequency", 83.0)  # Hz
        self.declare_parameter("default_stiffness", 1.0)   # Joint stiffness
        self.declare_parameter("contact_threshold", 5.0)   # FSR contact threshold
        
        self.control_frequency = self.get_parameter("control_frequency").get_parameter_value().double_value
        self.default_stiffness = self.get_parameter("default_stiffness").get_parameter_value().double_value
        self.contact_threshold = self.get_parameter("contact_threshold").get_parameter_value().double_value
        
        # Initialize NAO policy controller
        self.robot = NaoPolicy()
        
        # Control timing
        self.step_size = 1.0 / self.control_frequency
        self.timer = self.create_timer(self.step_size, self.control_callback)
        self.i = 0
        
        # Configuration
        self.fail_quietly = fail_quietly
        self.verbose = verbose
        
        # State tracking
        self.sensors_ready = False
        self.current_velocity_command = np.zeros(3)  # [vx, vy, wz]
        
        # Publishers
        self.joint_positions_pub = self.create_publisher(
            JointPositions, "/effectors/joint_positions", 10
        )
        self.joint_stiffnesses_pub = self.create_publisher(
            JointStiffnesses, "/effectors/joint_stiffnesses", 10
        )
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )
        self.joint_positions_sub = self.create_subscription(
            SensorJointPositions, "/sensors/joint_positions", 
            self.joint_positions_callback, 10
        )
        self.fsr_sub = self.create_subscription(
            FSR, "/sensors/fsr", self.fsr_callback, 10
        )
        self.accelerometer_sub = self.create_subscription(
            Accelerometer, "/sensors/accelerometer", 
            self.accelerometer_callback, 10
        )
        self.gyroscope_sub = self.create_subscription(
            Gyroscope, "/sensors/gyroscope", 
            self.gyroscope_callback, 10
        )
        
        self.get_logger().info(f"NAO Policy Node initialized at {self.control_frequency} Hz")

    def cmd_vel_callback(self, msg: Twist):
        """
        Callback for velocity commands.
        
        Args:
            msg: Twist message with linear and angular velocity commands
        """
        self.current_velocity_command[0] = msg.linear.x   # Forward velocity
        self.current_velocity_command[1] = msg.linear.y   # Lateral velocity  
        self.current_velocity_command[2] = msg.angular.z  # Angular velocity
        
        # Update the robot's velocity commands
        self.robot.velocity_commands = self.current_velocity_command.copy()
        
        if self.verbose:
            self.get_logger().info(
                f"Velocity command: vx={msg.linear.x:.3f}, vy={msg.linear.y:.3f}, wz={msg.angular.z:.3f}"
            )

    def joint_positions_callback(self, msg: SensorJointPositions):
        """
        Callback for joint position updates.
        
        Args:
            msg: Joint positions message from NAO sensors
        """
        if len(msg.positions) >= JointIndexes.NUMJOINTS:
            # Extract controlled joint positions
            controlled_positions = [msg.positions[i] for i in self.CONTROLLED_JOINT_INDICES]
            
            # Update robot state with joint positions and calculate velocities from history
            self.robot.update_joint_positions_with_history(controlled_positions)
            
            self.sensors_ready = True
            
            if self.verbose and self.i % 100 == 0:  # Log every 100 cycles
                self.get_logger().info(
                    f"Joint positions updated: {[f'{pos:.3f}' for pos in controlled_positions[:5]]}..."
                )
        else:
            self.get_logger().warn(
                f"Received {len(msg.positions)} joint positions, expected at least {JointIndexes.NUMJOINTS}"
            )

    def fsr_callback(self, msg: FSR):
        """
        Callback for Force Sensitive Resistor (FSR) updates.
        
        Args:
            msg: FSR message with foot pressure sensor data
        """
        # Aggregate FSR readings for each foot
        left_foot_forces = [
            msg.l_foot_front_left, msg.l_foot_front_right,
            msg.l_foot_back_left, msg.l_foot_back_right
        ]
        right_foot_forces = [
            msg.r_foot_front_left, msg.r_foot_front_right, 
            msg.r_foot_back_left, msg.r_foot_back_right
        ]
        
        # Update foot contact in robot
        self.robot.update_foot_contact_from_fsr(
            left_foot_forces, right_foot_forces, self.contact_threshold
        )
        
        if self.verbose and self.i % 200 == 0:  # Log every 200 cycles
            self.get_logger().info(
                f"Foot contact: L={self.robot.foot_contact[0]:.1f}, R={self.robot.foot_contact[1]:.1f}"
            )

    def accelerometer_callback(self, msg: Accelerometer):
        """
        Callback for accelerometer updates.
        
        Args:
            msg: Accelerometer message with linear acceleration data
        """
        linear_accel = [msg.x, msg.y, msg.z]
        
        # Update base velocity estimation in robot (will be combined with gyro data)
        # For now, just store the acceleration - gyroscope_callback will do the full update
        self._last_accel = linear_accel

    def gyroscope_callback(self, msg: Gyroscope):
        """
        Callback for gyroscope updates.
        
        Args:
            msg: Gyroscope message with angular velocity data
        """
        angular_vel = [msg.x, msg.y, msg.z]
        
        # If we have recent accelerometer data, update base velocities
        if hasattr(self, '_last_accel'):
            self.robot.update_base_velocity_from_accel(self._last_accel, angular_vel)
        else:
            # Just update angular velocity
            self.robot.base_ang_vel = np.array(angular_vel, dtype=np.float32)

    def control_callback(self):
        """
        Main control loop callback that runs the policy and publishes commands.
        """
        if not self.sensors_ready:
            if self.verbose and self.i % 100 == 0:
                self.get_logger().info("Waiting for sensor data...")
            self.i += 1
            return
            
        try:
            # Get joint positions from policy
            joint_positions = self.robot.forward(self.step_size, self.current_velocity_command)
            
            if joint_positions is not None:
                if len(joint_positions) != 21:
                    if not self.fail_quietly:
                        self.get_logger().error(
                            f"Policy returned {len(joint_positions)} joint positions, expected 21"
                        )
                    return
                    
                # Publish joint commands
                self.publish_joint_commands(joint_positions)
                
                if self.verbose and self.i % 100 == 0:
                    self.get_logger().info(
                        f"Published joint commands: {[f'{pos:.3f}' for pos in joint_positions[:5]]}..."
                    )
            else:
                if self.verbose and self.i % 100 == 0:
                    self.get_logger().warn("Policy returned None - insufficient sensor data")
                    
        except Exception as e:
            if not self.fail_quietly:
                self.get_logger().error(f"Error in control loop: {str(e)}")
                
        self.i += 1

    def publish_joint_commands(self, joint_positions):
        """
        Publish joint position and stiffness commands.
        
        Args:
            joint_positions: Array of 21 joint position commands
        """
        # Publish joint positions
        joint_pos_msg = JointPositions()
        joint_pos_msg.indexes = self.CONTROLLED_JOINT_INDICES
        joint_pos_msg.positions = joint_positions.tolist()
        self.joint_positions_pub.publish(joint_pos_msg)
        
        # Publish joint stiffnesses
        joint_stiff_msg = JointStiffnesses()
        joint_stiff_msg.indexes = self.CONTROLLED_JOINT_INDICES
        joint_stiff_msg.stiffnesses = [self.default_stiffness] * 21
        self.joint_stiffnesses_pub.publish(joint_stiff_msg)


def main(args=None):
    """Main function to run the NAO policy node."""
    rclpy.init(args=args)
    
    # Allow command line arguments for configuration
    import sys
    fail_quietly = '--fail-quietly' in sys.argv
    verbose = '--verbose' in sys.argv
    
    node = NaoPolicyNode(fail_quietly=fail_quietly, verbose=verbose)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down NAO Policy Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
