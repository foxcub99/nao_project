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
    env: ManagerBasedRLEnv, new_length: float
) -> None:
    """Change the episode length in seconds."""
    env.episode_length_s = new_length