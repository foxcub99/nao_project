# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .nao import NAO_CFG  # Change import to NAO configuration

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .nao_locomotion_env import NaoLocomotionEnv


@configclass
class NaoEnvCfg(DirectRLEnvCfg):  # Rename class to reflect NAO robot

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

    # scene - adjusted for NAO's smaller size
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = NAO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # NAO joint gears - must match exact order in actuator configuration in nao.py
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
        5.0,  # LHipYawPitch
        5.0,  # LHipRoll
        5.0,  # RHipRoll
        5.0,  # LHipPitch
        5.0,  # RHipPitch
        5.0,  # LKneePitch
        5.0,  # RKneePitch
        5.0,  # LAnklePitch
        5.0,  # RAnklePitch
        5.0,  # LAnkleRoll
        5.0,  # RAnkleRoll
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


class NaoEnv(NaoLocomotionEnv):  # Rename class
    cfg: NaoEnvCfg

    def __init__(self, cfg: NaoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Setup scene with proper ground plane handling to avoid conflicts."""
        # First setup the robot
        self.robot = Articulation(self.cfg.robot)

        # Setup terrain using the configuration
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # After cloning, search for and deactivate any ground plane prims
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage:
                # Look for ground plane prims in each environment
                for env_idx in range(self.scene.cfg.num_envs):
                    env_prim_path = f"/World/envs/env_{env_idx}"
                    env_prim = stage.GetPrimAtPath(env_prim_path)
                    if env_prim:
                        # Recursively search for ground plane prims and disable them
                        self._disable_ground_plane_prims(stage, env_prim_path)
        except ImportError:
            # If USD imports fail, just continue - this is mainly for development environments
            print(
                "[NAO ENV] Warning: Could not import USD modules to clean up ground planes"
            )

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _disable_ground_plane_prims(self, stage, env_prim_path: str):
        """Recursively find and disable ground plane prims in the environment."""
        from pxr import Usd, UsdGeom

        # Common ground plane names to look for
        ground_plane_names = [
            "ground",
            "Ground",
            "GroundPlane",
            "groundPlane",
            "floor",
            "Floor",
            "terrain",
            "Terrain",
            "environment",
            "Environment",
            "world",
            "World",
            "plane",
            "Plane",
        ]

        # Search through all prims in this environment
        env_prim = stage.GetPrimAtPath(env_prim_path)
        for prim in Usd.PrimRange(env_prim):
            prim_name = prim.GetName().lower()
            # Check if this prim looks like a ground plane
            if any(
                ground_name.lower() in prim_name for ground_name in ground_plane_names
            ):
                # Check if it's a mesh or has geometry that could be a ground plane
                if (
                    prim.IsA(UsdGeom.Mesh)
                    or prim.IsA(UsdGeom.Xform)
                    or prim.HasAPI(UsdGeom.CollisionAPI)
                ):
                    # Disable the prim by making it invisible and inactive
                    try:
                        imageable = UsdGeom.Imageable(prim)
                        if imageable:
                            imageable.MakeInvisible()
                        prim.SetActive(False)
                        print(
                            f"[NAO ENV] Disabled potential ground plane prim: {prim.GetPath()}"
                        )
                    except Exception as e:
                        print(f"[NAO ENV] Could not disable prim {prim.GetPath()}: {e}")
                        continue
