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


def feet_air_time_positive_biped(env, command_name: str, time_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
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
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
    # Reward air time specifically when in single stance
    air_time_during_single_stance = torch.where(single_stance.unsqueeze(-1), air_time, 0.0)
    # Take max air time instead of min to encourage proper stepping
    reward = torch.max(air_time_during_single_stance, dim=1)[0]
    reward = torch.clamp(reward, max=time_threshold)
    
    # Scale reward based on command magnitude
    command = env.command_manager.get_command(command_name)[:, :3]
    command_term = env.command_manager.get_term(command_name)
    
    # Get command ranges
    lin_vel_x_range = command_term.cfg.ranges.lin_vel_x
    lin_vel_y_range = command_term.cfg.ranges.lin_vel_y
    ang_vel_z_range = command_term.cfg.ranges.ang_vel_z
    
    # For each command component, use the appropriate range limit based on sign
    # Negative commands scale against minimum (lower bound), positive against maximum (upper bound)
    max_x = torch.where(command[:, 0] >= 0, lin_vel_x_range[1], abs(lin_vel_x_range[0]))
    max_y = torch.where(command[:, 1] >= 0, lin_vel_y_range[1], abs(lin_vel_y_range[0]))
    max_z = torch.where(command[:, 2] >= 0, ang_vel_z_range[1], abs(ang_vel_z_range[0]))
    
    # Normalize each command component by its appropriate range limit
    normalized_commands = torch.stack([
        torch.abs(command[:, 0] / max_x),
        torch.abs(command[:, 1] / max_y),
        torch.abs(command[:, 2] / max_z)
    ], dim=1)
    
    # Calculate overall command magnitude as norm of normalized commands
    # Divide by sqrt(3) to ensure max possible norm equals 1.0
    command_magnitude = torch.norm(normalized_commands, dim=1) / torch.sqrt(torch.tensor(3.0))
    command_magnitude = torch.clamp(command_magnitude, 0.0, 1.0)
    
    # Apply command scaling to reward (0 for no command, 1 for max command)
    reward *= command_magnitude
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


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

