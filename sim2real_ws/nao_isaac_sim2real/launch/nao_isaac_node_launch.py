#!/usr/bin/env python3

"""
Launch file for NAO Isaac RL Policy Node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    pt_filepath_arg = DeclareLaunchArgument(
        'pt_filepath',
        default_value='',
        description='Path to the PyTorch policy checkpoint file (.pt)'
    )
    
    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='83.0',
        description='Control loop frequency in Hz'
    )
    
    # Create the node
    nao_isaac_node = Node(
        package='nao_isaac_sim2real',
        executable='nao_isaac_node',
        name='nao_isaac_node',
        parameters=[{
            'pt_filepath': LaunchConfiguration('pt_filepath'),
            'control_frequency': LaunchConfiguration('control_frequency'),
        }],
        output='screen'
    )
    
    return LaunchDescription([
        pt_filepath_arg,
        control_frequency_arg,
        nao_isaac_node
    ])
