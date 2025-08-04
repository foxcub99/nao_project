# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from .nao_env_cfg import NaoEnvCfg
from .nao_locomotion_env import NaoLocomotionEnv


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

        # # After cloning, search for and deactivate any ground plane prims
        # try:
        #     import omni.usd
        #     from pxr import Usd, UsdGeom

        #     stage = omni.usd.get_context().get_stage()
        #     if stage:
        #         # Look for ground plane prims in each environment
        #         for env_idx in range(self.scene.cfg.num_envs):
        #             env_prim_path = f"/World/envs/env_{env_idx}"
        #             env_prim = stage.GetPrimAtPath(env_prim_path)
        #             if env_prim:
        #                 # Recursively search for ground plane prims and disable them
        #                 self._disable_ground_plane_prims(stage, env_prim_path)
        # except ImportError:
        #     # If USD imports fail, just continue - this is mainly for development environments
        #     print(
        #         "[NAO ENV] Warning: Could not import USD modules to clean up ground planes"
        #     )

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # def _disable_ground_plane_prims(self, stage, env_prim_path: str):
    #     """Recursively find and disable ground plane prims in the environment."""
    #     from pxr import Usd, UsdGeom

    #     # Common ground plane names to look for
    #     ground_plane_names = [
    #         "ground",
    #         "Ground",
    #         "GroundPlane",
    #         "groundPlane",
    #         "floor",
    #         "Floor",
    #         "terrain",
    #         "Terrain",
    #         "environment",
    #         "Environment",
    #         "world",
    #         "World",
    #         "plane",
    #         "Plane",
    #     ]

    #     # Search through all prims in this environment
    #     env_prim = stage.GetPrimAtPath(env_prim_path)
    #     for prim in Usd.PrimRange(env_prim):
    #         prim_name = prim.GetName().lower()
    #         # Check if this prim looks like a ground plane
    #         if any(
    #             ground_name.lower() in prim_name for ground_name in ground_plane_names
    #         ):
    #             # Check if it's a mesh or has geometry that could be a ground plane
    #             if (
    #                 prim.IsA(UsdGeom.Mesh)
    #                 or prim.IsA(UsdGeom.Xform)
    #                 or prim.HasAPI(UsdGeom.CollisionAPI)
    #             ):
    #                 # Disable the prim by making it invisible and inactive
    #                 try:
    #                     imageable = UsdGeom.Imageable(prim)
    #                     if imageable:
    #                         imageable.MakeInvisible()
    #                     prim.SetActive(False)
    #                     print(
    #                         f"[NAO ENV] Disabled potential ground plane prim: {prim.GetPath()}"
    #                     )
    #                 except Exception as e:
    #                     print(f"[NAO ENV] Could not disable prim {prim.GetPath()}: {e}")
    #                     continue
