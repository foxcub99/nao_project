# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the NAO robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

NAO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"C:/Users/reill/nao_project/source/nao_project/nao_project/assets/nao/nao-project.usd",  # Path to NAO USD file
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.1,  # was 5.0, using to debug
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,  # Increased for better stability
            sleep_threshold=0.005,  # Lowered for more responsive behavior
            stabilization_threshold=0.001,
        ),
        copy_from_source=True,  # This may help with filtering
        # Scale the robot if needed and ensure no ground planes are copied
        scale=None,
        visible=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(
            0.0,
            0.0,
            0.345,
        ),  # NAO is about 58cm tall, starting with feet on ground
        joint_pos={
            # Default joint positions for stable standing pose
            "HeadYaw": 0.0,
            "HeadPitch": 0.0,
            # Arms - positioned to avoid interference with walking
            ".*ShoulderPitch": 0.0,  # Arms slightly forward
            "RShoulderRoll": -0.05,  # Right arm slightly away from body
            "LShoulderRoll": 0.05,  # Left arm slightly away from body
            ".*ElbowYaw": 0.0,
            "RElbowRoll": 0.05,  # Right elbow slightly bent
            "LElbowRoll": -0.05,  # Left elbow slightly bent
            # Legs - neutral standing position
            ".*HipYawPitch": 0.0,
            ".*HipRoll": 0.0,
            ".*HipPitch": 0.0,
            ".*KneePitch": 0.0,
            ".*AnklePitch": 0.0,
            ".*AnkleRoll": 0.0,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[
                # Be more specific - use exact joint names instead of regex
                "HeadYaw",
                "HeadPitch",
                "LShoulderPitch",
                "RShoulderPitch",
                "LShoulderRoll",
                "RShoulderRoll",
                "LElbowYaw",
                "RElbowYaw",
                "LElbowRoll",
                "RElbowRoll",
                "LHipYawPitch",  # NAO shares this joint
                "LHipRoll",
                "RHipRoll",
                "LHipPitch",
                "RHipPitch",
                "LKneePitch",
                "RKneePitch",
                "LAnklePitch",
                "RAnklePitch",
                "LAnkleRoll",
                "RAnkleRoll",
            ],  # Target all joints
            stiffness=None,
            damping=None,
        ),
    },
)
"""Configuration for the NAO robot."""

# You can keep the HUMANOID_CFG if needed or comment it out
"""
HUMANOID_CFG = ArticulationCfg(
    # Original humanoid configuration...
)
"""
