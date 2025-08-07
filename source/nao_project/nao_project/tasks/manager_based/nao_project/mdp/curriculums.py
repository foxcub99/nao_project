# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum learning functions for NAO project.

This module contains both standard Isaac Lab curriculum functions and custom
NAO-specific curriculum implementations for reward weights, command difficulty,
and terrain progression.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, Any

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def change_episode_length_s(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, old_value: float, 
    new_length: float, num_steps: int, **kwargs
) -> float:
    """Change the episode length of the environment after specified training steps.
    
    This curriculum function is designed to be used with modify_env_param to change
    the episode_length_s parameter of the environment configuration once training
    has progressed beyond the specified number of steps.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs (not used in this function but required for curriculum interface).
        old_value: Current value of episode_length_s.
        new_length: The new episode length in seconds to set.
        num_steps: Number of training steps after which to change the episode length.
        **kwargs: Additional keyword arguments (unused).
        
    Returns:
        The new episode length if the step threshold is reached, otherwise the current length.
    """
    # Check if we've reached the step threshold
    if env.common_step_counter >= num_steps:
        return new_length
    else:
        # Return current episode length (no change)
        return old_value


def increase_episode_length_gradually(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, old_value: float,
    initial_length: float, final_length: float, start_steps: int, end_steps: int, **kwargs
) -> float:
    """Gradually increase episode length over a range of training steps.
    
    This curriculum function gradually increases the episode length from an initial
    value to a final value over a specified range of training steps.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs (not used in this function but required for curriculum interface).
        old_value: Current value of episode_length_s.
        initial_length: The initial episode length in seconds.
        final_length: The final episode length in seconds.
        start_steps: Number of training steps to start the increase.
        end_steps: Number of training steps to complete the increase.
        **kwargs: Additional keyword arguments (unused).
        
    Returns:
        The interpolated episode length based on current training step.
    """
    current_step = env.common_step_counter
    
    if current_step < start_steps:
        return initial_length
    elif current_step >= end_steps:
        return final_length
    else:
        # Linear interpolation between initial and final length
        progress = (current_step - start_steps) / (end_steps - start_steps)
        return initial_length + progress * (final_length - initial_length)

