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
        usd_path=f"C:/Users/reill/lab/nao_project/source/nao_project/nao_project/assets/nao/nao-project-contact.usd",  # Path to NAO USD file
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=True,
        scale=None,
        visible=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(
            0.0,
            0.0,
            0.345,
        ),
        joint_pos={
            "HeadYaw": 0.0,
            "HeadPitch": 0.0,
            ".*ShoulderPitch": 0.0,
            "RShoulderRoll": -0.05,
            "LShoulderRoll": 0.05,
            ".*ElbowYaw": 0.0,
            "RElbowRoll": 0.05,
            "LElbowRoll": -0.05,
            "LHipYawPitch": 0.0,
            ".*HipRoll": 0.0,
            ".*HipPitch": 0.0,
            ".*KneePitch": 0.0,
            ".*AnklePitch": 0.0,
            ".*AnkleRoll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "HeadYaw",
                "HeadPitch",
            ],
            stiffness=None,
            damping=None,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*ShoulderPitch",
                ".*ShoulderRoll",
                ".*ElbowYaw",
                ".*ElbowRoll",
            ],
            stiffness=None,
            damping=None,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "LHipYawPitch",
                ".*HipRoll",
                ".*HipPitch",
                ".*KneePitch",
            ],
            stiffness=None,
            damping=None,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*AnklePitch",
                ".*AnkleRoll",
            ],
            stiffness=None,
            damping=None,
        ),
    },
)
"""Configuration for the NAO robot."""


