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


def joint_deviation_l1_with_command_scaling(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    command_index: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one, scaled by command magnitude.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    command = env.command_manager.get_command(command_name)[:, command_index]
    
    command_term = env.command_manager.get_term(command_name)
    if command_index == 0:
        min_range, max_range = command_term.cfg.ranges.lin_vel_x
    elif command_index == 1:
        min_range, max_range = command_term.cfg.ranges.lin_vel_y
    elif command_index == 2:
        min_range, max_range = command_term.cfg.ranges.ang_vel_z
    else:
        raise ValueError(f"Unsupported command_index: {command_index}. Supported values are 0 (x), 1 (y), 2 (z).")
    
    max_abs_command = max(abs(min_range), abs(max_range))
    
    scale_factor = torch.abs(command) / max_abs_command
    scale_factor = torch.clamp(scale_factor, 0.0, 1.0)  # Ensure it's between 0 and 1
    
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation_penalty = torch.sum(torch.abs(angle), dim=1)
    
    # Apply scaling based on command magnitude
    return deviation_penalty * scale_factor


# def joint_deviation_l1_with_adaptive_scaling(
#     env: ManagerBasedRLEnv, 
#     command_name: str, 
#     command_index: int,
#     min_scale: float = 0.0,
#     max_scale: float = 1.0,
#     threshold: float = 0.1,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize joint positions that deviate from the default one, with adaptive scaling based on command.
    
#     This is a more flexible version that allows you to set minimum and maximum scaling factors,
#     and includes a threshold below which the penalty is minimized.
    
#     Args:
#         env: The environment instance.
#         command_name: Name of the command to use for scaling.
#         command_index: Index of the command component to use for scaling.
#         min_scale: Minimum scaling factor (when command is below threshold).
#         max_scale: Maximum scaling factor (when command is at range limits).
#         threshold: Command magnitude threshold below which minimum scaling is applied.
#         asset_cfg: Configuration for the asset.
        
#     Returns:
#         Adaptively scaled joint deviation penalty.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
    
#     # Get the current command value
#     command = env.command_manager.get_command(command_name)[:, command_index]
#     command_abs = torch.abs(command)
    
#     # Get the command ranges from the command manager
#     command_term = env.command_manager.get_term(command_name)
#     if command_index == 0:
#         min_range, max_range = command_term.cfg.ranges.lin_vel_x
#     elif command_index == 1:
#         min_range, max_range = command_term.cfg.ranges.lin_vel_y
#     elif command_index == 2:
#         min_range, max_range = command_term.cfg.ranges.ang_vel_z
#     else:
#         raise ValueError(f"Unsupported command_index: {command_index}. Supported values are 0 (x), 1 (y), 2 (z).")
    
#     # Calculate the maximum absolute command value from the range
#     max_abs_command = max(abs(min_range), abs(max_range))
    
#     # Apply threshold: below threshold use min_scale, above threshold scale linearly
#     above_threshold = command_abs > threshold
    
#     # For commands above threshold, scale from min_scale to max_scale
#     linear_scale = min_scale + (max_scale - min_scale) * (command_abs - threshold) / (max_abs_command - threshold)
#     linear_scale = torch.clamp(linear_scale, min_scale, max_scale)
    
#     # Apply threshold logic
#     scale_factor = torch.where(above_threshold, linear_scale, torch.full_like(command_abs, min_scale))
    
#     # compute joint deviation
#     angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     deviation_penalty = torch.sum(torch.abs(angle), dim=1)
    
#     # Apply scaling based on command magnitude
#     return deviation_penalty * scale_factor