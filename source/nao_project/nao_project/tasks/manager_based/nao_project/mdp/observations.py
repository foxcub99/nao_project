# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define observations for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to
specify the observation function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the contact forces of the specified body parts (typically feet).

    This function returns the contact force magnitude for each specified body part,
    which can be used as an observation to help the agent understand ground contact state.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor, specifying which bodies to monitor.

    Returns:
        Contact force magnitudes for each specified body part.
        Shape: (num_envs, len(sensor_cfg.body_ids))
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get the Z-component contact forces for the specified bodies (vertical forces from underneath)
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    
    # Calculate the magnitude of the contact forces and clamp to minimum of 0
    # This simulates real foot sensors that only detect positive forces from underneath
    contact_force_magnitudes = torch.clamp(contact_forces, min=0.0)
    
    return contact_force_magnitudes
