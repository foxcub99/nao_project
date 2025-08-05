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


def modify_reward_weights(env: ManagerBasedRLEnv, env_ids: torch.Tensor, initial_weights: Dict[str, float], 
                         final_weights: Dict[str, float]) -> None:
    """Modify reward weights based on curriculum progress.
    
    This function gradually transitions reward weights from initial to final values
    based on the curriculum progress.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to apply curriculum to.
        initial_weights: Initial reward weights for each term.
        final_weights: Final reward weights for each term.
    """
    # Get curriculum progress (0 to 1)
    curriculum_factor = env.curriculum_manager.get_curriculum_factor("reward_weights")
    
    # Interpolate between initial and final weights
    for reward_term_name in initial_weights.keys():
        if hasattr(env.reward_manager.active_terms, reward_term_name):
            initial_weight = initial_weights[reward_term_name]
            final_weight = final_weights[reward_term_name]
            
            # Linear interpolation
            current_weight = initial_weight + curriculum_factor * (final_weight - initial_weight)
            
            # Update the reward term weight
            reward_term = env.reward_manager.active_terms[reward_term_name]
            reward_term.weight = current_weight
            
            # Track curriculum progress in custom metrics
            if hasattr(env, 'extras'):
                env.extras[f"curriculum/reward_weight_{reward_term_name}"] = current_weight


def modify_command_ranges(env: ManagerBasedRLEnv, env_ids: torch.Tensor, 
                         initial_ranges: Dict[str, tuple], final_ranges: Dict[str, tuple]) -> None:
    """Modify command ranges based on curriculum progress.
    
    This function gradually increases the difficulty of velocity commands.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to apply curriculum to.
        initial_ranges: Initial command ranges.
        final_ranges: Final command ranges.
    """
    curriculum_factor = env.curriculum_manager.get_curriculum_factor("command_difficulty")
    
    # Update command manager ranges
    if hasattr(env.command_manager, 'get_term'):
        base_velocity_cmd = env.command_manager.get_term("base_velocity")
        
        for range_name, initial_range in initial_ranges.items():
            if hasattr(base_velocity_cmd.cfg.ranges, range_name):
                final_range = final_ranges[range_name]
                
                # Interpolate range values
                current_min = initial_range[0] + curriculum_factor * (final_range[0] - initial_range[0])
                current_max = initial_range[1] + curriculum_factor * (final_range[1] - initial_range[1])
                
                # Update the range
                setattr(base_velocity_cmd.cfg.ranges, range_name, (current_min, current_max))
                
                # Track in metrics
                if hasattr(env, 'extras'):
                    env.extras[f"curriculum/command_range_{range_name}_min"] = current_min
                    env.extras[f"curriculum/command_range_{range_name}_max"] = current_max


def modify_terrain_difficulty(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Modify terrain difficulty based on curriculum progress.
    
    This function gradually increases terrain complexity.
    """
    curriculum_factor = env.curriculum_manager.get_curriculum_factor("terrain_difficulty")
    
    # This would work with the existing terrain levels curriculum
    # The terrain generator automatically handles this based on curriculum_factor
    
    if hasattr(env, 'extras'):
        env.extras["curriculum/terrain_difficulty"] = curriculum_factor


def adaptive_curriculum_update(env: ManagerBasedRLEnv, env_ids: torch.Tensor, 
                              performance_threshold: float = 0.8) -> None:
    """Update curriculum based on agent performance.
    
    This function monitors agent performance and advances curriculum stages
    when performance criteria are met.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to evaluate.
        performance_threshold: Performance threshold to advance curriculum.
    """
    # Get recent performance metrics
    if hasattr(env, 'reward_manager') and hasattr(env.reward_manager, 'total_reward'):
        recent_rewards = env.reward_manager.total_reward
        avg_performance = torch.mean(recent_rewards).item()
        
        # Normalize performance (this would need to be tailored to your specific rewards)
        normalized_performance = torch.clamp(avg_performance / 100.0, 0.0, 1.0)
        
        # Check if we should advance curriculum
        if normalized_performance > performance_threshold:
            if hasattr(env, 'curriculum_manager'):
                env.curriculum_manager.advance_if_ready()
        
        # Track performance metrics
        if hasattr(env, 'extras'):
            env.extras["curriculum/avg_performance"] = avg_performance
            env.extras["curriculum/normalized_performance"] = normalized_performance.item()


def curriculum_stage_progression(env: ManagerBasedRLEnv, env_ids: torch.Tensor,
                               total_stages: int = 3, episodes_per_stage: int = 1000) -> None:
    """Progress through curriculum stages based on episode count.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs.
        total_stages: Total number of curriculum stages.
        episodes_per_stage: Episodes to spend in each stage.
    """
    # Get current episode count (this would need to be tracked by the environment)
    if hasattr(env, 'episode_count'):
        current_episode = env.episode_count
        current_stage = min(current_episode // episodes_per_stage, total_stages - 1)
        stage_progress = (current_episode % episodes_per_stage) / episodes_per_stage
        
        # Update curriculum manager
        if hasattr(env, 'curriculum_manager'):
            env.curriculum_manager.set_stage(current_stage, stage_progress)
        
        # Track curriculum metrics
        if hasattr(env, 'extras'):
            env.extras["curriculum/current_stage"] = current_stage
            env.extras["curriculum/stage_progress"] = stage_progress
            env.extras["curriculum/total_episodes"] = current_episode


