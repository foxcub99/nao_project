# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ....assets.nao import NAO_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class NaoEnvCfg(DirectRLEnvCfg):

    # def __post_init__(self):
    #     self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096

    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21  # Update for NAO's DOFs (typically 25 joints)
    observation_space = 75  # Fixed: 1+3+3+1+1+1+1+1+21+21+21 = 75
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=0,  # Changed from -1 to 0 to ensure priority
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    robot: ArticulationCfg = NAO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # NAO joint gears - must match exact order in actuator configuration in nao.py ?
    joint_gears: list = [
        0.01,  # HeadYaw
        0.01,  # HeadPitch
        0.01,  # LShoulderPitch
        0.01,  # RShoulderPitch
        0.01,  # LShoulderRoll
        0.01,  # RShoulderRoll
        0.01,  # LElbowYaw
        0.01,  # RElbowYaw
        0.01,  # LElbowRoll
        0.01,  # RElbowRoll
        3.0,  # LHipYawPitch
        2.0,  # LHipRoll
        2.0,  # RHipRoll
        5.0,  # LHipPitch
        5.0,  # RHipPitch
        5.0,  # LKneePitch
        5.0,  # RKneePitch
        5.0,  # LAnklePitch
        5.0,  # RAnklePitch
        2.0,  # LAnkleRoll
        2.0,  # RAnkleRoll
    ]

    # Reward weights optimized for velocity-focused locomotion
    heading_weight: float = 2.0  # Increased - important for maintaining direction
    up_weight: float = 0.5  # Increased but still moderate - some stability needed

    # Reduced energy costs to allow more dynamic movement
    energy_cost_scale: float = 0.02  # Reduced to allow more aggressive movement
    actions_cost_scale: float = 0.005  # Reduced to allow larger actions
    alive_reward_scale: float = 1.0  # Reduced - less focus on just surviving
    dof_vel_scale: float = 0.1

    # Lower death penalty since you don't mind occasional falls
    death_cost: float = -0.5  # Reduced penalty for falling
    termination_height: float = (
        0.25  # Reduced for NAO's smaller height (about 58cm tall)
    )

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

    lin_vel_x_goal: float = 1.0
    lin_vel_x_tracking_weight: float = 0.5
