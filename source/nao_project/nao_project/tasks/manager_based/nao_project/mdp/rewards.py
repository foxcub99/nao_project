# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# def feet_air_time_heightf(env, command_name: str, sensor_cfg: SceneEntityCfg, height_max_threshold: float, time_threshold: float, max_time_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Reward foot height during single stance for bipeds.

#     This function rewards the agent for lifting the non-contact foot higher during single stance.
#     The reward increases with foot height up to a specified threshold and is scaled by command magnitude.
#     Time reward increases linearly up to time_threshold, then becomes 0 at max_time_threshold.

#     Args:
#         env: The learning environment.
#         command_name: The name of the command term.
#         sensor_cfg: The contact sensor configuration.
#         height_max_threshold: Maximum height threshold for clamping the height reward.
#         time_threshold: Air time threshold for maximum time reward.
#         max_time_threshold: Air time threshold where reward becomes zero.
#         asset_cfg: The robot asset configuration.

#     Returns:
#         The computed reward tensor.
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: Articulation = env.scene[asset_cfg.name]
    
#     # Get contact information
#     air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
#     in_contact = contact_time > 0.0
#     single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
#     # Get foot positions in world frame (z-coordinate for height)
#     foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # z-coordinate for height
    
#     # During single stance, reward the height of the airborne foot
#     # Create mask for airborne feet (not in contact)
#     airborne_feet = ~in_contact
    
#     # Only consider height when in single stance
#     foot_heights_during_single_stance = torch.where(
#         single_stance.unsqueeze(-1) & airborne_feet, 
#         foot_positions, 
#         0.0
#     )
    
#     # Take the maximum height of airborne feet during single stance
#     height_reward = torch.max(foot_heights_during_single_stance, dim=1)[0]
#     height_reward = torch.clamp(height_reward, max=height_max_threshold)

#     # Reward air time specifically when in single stance
#     air_time_during_single_stance = torch.where(single_stance.unsqueeze(-1), air_time, 0.0)
#     # Take max air time instead of min to encourage proper stepping
#     max_air_time = torch.max(air_time_during_single_stance, dim=1)[0]
    
#     # Time reward: increases linearly up to time_threshold, then stays at max
#     air_time_reward = torch.clamp(max_air_time, max=time_threshold)

#     reward = (air_time_reward * height_reward) ** 0.5

#     # Zero out reward if air time exceeds max_time_threshold
#     excessive_air_time = max_air_time >= max_time_threshold
#     reward = torch.where(excessive_air_time, torch.zeros_like(reward), reward)
    
#     return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, max_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    
    # Get air time during single stance to check for excessive air time
    air_time_during_single_stance = torch.where(single_stance.unsqueeze(-1), air_time, 0.0)
    max_air_time_in_single_stance = torch.max(air_time_during_single_stance, dim=1)[0]
    excessive_air_time = max_air_time_in_single_stance >= max_threshold
    reward = torch.where(excessive_air_time, torch.zeros_like(reward), reward)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return reward


def feet_air_height(env, command_name: str, sensor_cfg: SceneEntityCfg, height_max_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward foot height during single stance for bipeds.

    This function rewards the agent for lifting the non-contact foot higher during single stance.
    The reward increases with foot height up to a specified threshold and is scaled by command magnitude.

    Args:
        env: The learning environment.
        command_name: The name of the command term.
        sensor_cfg: The contact sensor configuration.
        height_max_threshold: Maximum height threshold for clamping the reward.
        asset_cfg: The robot asset configuration.

    Returns:
        The computed reward tensor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact information
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
    # Get foot positions in world frame (z-coordinate for height)
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # z-coordinate for height
    
    # During single stance, reward the height of the airborne foot
    # Create mask for airborne feet (not in contact)
    airborne_feet = ~in_contact
    
    # Only consider height when in single stance
    foot_heights_during_single_stance = torch.where(
        single_stance.unsqueeze(-1) & airborne_feet, 
        foot_positions, 
        0.0
    )
    
    # Take the maximum height of airborne feet during single stance
    reward = torch.max(foot_heights_during_single_stance, dim=1)[0]
    reward = torch.clamp(reward, max=height_max_threshold)
    
    # Scale reward based on command magnitude (similar to feet_air_time_positive_biped)
    command = env.command_manager.get_command(command_name)[:, :3]
    command_term = env.command_manager.get_term(command_name)
    
    # Get command ranges
    lin_vel_x_range = command_term.cfg.ranges.lin_vel_x
    lin_vel_y_range = command_term.cfg.ranges.lin_vel_y
    ang_vel_z_range = command_term.cfg.ranges.ang_vel_z
    
    # For each command component, use the appropriate range limit based on sign
    max_x = torch.where(command[:, 0] >= 0, lin_vel_x_range[1], abs(lin_vel_x_range[0]))
    max_y = torch.where(command[:, 1] >= 0, lin_vel_y_range[1], abs(lin_vel_y_range[0]))
    max_z = torch.where(command[:, 2] >= 0, ang_vel_z_range[1], abs(ang_vel_z_range[0]))
    
    # Guard against zero ranges to prevent division by zero
    max_x = torch.clamp(max_x, min=1e-8)
    max_y = torch.clamp(max_y, min=1e-8)
    max_z = torch.clamp(max_z, min=1e-8)
    
    # Normalize each command component by its appropriate range limit
    normalized_commands = torch.stack([
        torch.abs(command[:, 0] / max_x),
        torch.abs(command[:, 1] / max_y),
        torch.abs(command[:, 2] / max_z)
    ], dim=1)
    
    # Calculate overall command magnitude as norm of normalized commands
    command_magnitude = torch.norm(normalized_commands, dim=1) / torch.sqrt(torch.tensor(3.0))
    command_magnitude = torch.clamp(command_magnitude, 0.0, 1.0)
    
    # Apply command scaling to reward (0 for no command, 1 for max command)
    reward *= command_magnitude
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def undesired_contacts_filtered(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold using filtered contact forces
    
    # Try to use force_matrix_w_history first (with history), fallback to force_matrix_w
    filtered_contact_forces = getattr(contact_sensor.data, 'force_matrix_w_history', None)
    if filtered_contact_forces is not None:
        # force_matrix_w_history shape: (N, T, B, M, 3) where T=history_length, M=filtered_bodies
        # Sum over all filtered bodies (M dimension) to get total force per sensor body
        total_forces_per_body = torch.sum(filtered_contact_forces, dim=3)  # Shape: (N, T, B, 3)
        # Select specific sensor bodies, following the same pattern as feet_slide function
        selected_forces = total_forces_per_body[:, :, sensor_cfg.body_ids, :]  # Shape: (N, T, len(body_ids), 3)
        # Take max over history and compute contact threshold
        is_contact = torch.max(torch.norm(selected_forces, dim=-1), dim=1)[0] > threshold
    else:
        # Fallback to force_matrix_w (current frame only)
        filtered_contact_forces = getattr(contact_sensor.data, 'force_matrix_w', None)
        if filtered_contact_forces is not None:
            # force_matrix_w shape: (N, B, M, 3)
            # Sum over all filtered bodies (M dimension) to get total force per sensor body
            total_forces_per_body = torch.sum(filtered_contact_forces, dim=2)  # Shape: (N, B, 3)
            # Select specific sensor bodies
            selected_forces = total_forces_per_body[:, sensor_cfg.body_ids, :]  # Shape: (N, len(body_ids), 3)
            is_contact = torch.norm(selected_forces, dim=-1) > threshold
        else:
            # Final fallback to unfiltered net forces (same pattern as original)
            net_contact_forces = contact_sensor.data.net_forces_w_history
            # Follow the exact same pattern as feet_slide function
            is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1)[0] > threshold
    
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)