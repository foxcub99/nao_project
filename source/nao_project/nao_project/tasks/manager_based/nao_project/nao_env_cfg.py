# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg

from . import mdp
from .velocity_env_cfg import LocomotionVelocityEnvCfg
# from .nao_curriculum_env_cfg import NaoEnvCfg 

##
# Pre-defined configs
##
from ....assets.nao import NAO_CFG


@configclass
class NaoEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Scene
        self.scene.robot = NAO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path=f"/World/envs/env_.*/Robot/NaoH25V50/.*"

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Actions
        self.actions.joint_pos.joint_names = [
            "HeadYaw", "HeadPitch",
            ".*ShoulderPitch", ".*ShoulderRoll",
            ".*ElbowYaw", ".*ElbowRoll",
            "LHipYawPitch",
            ".*HipRoll", ".*HipPitch",
            ".*KneePitch",
            ".*AnklePitch", ".*AnkleRoll",
            # Explicitly exclude: .*WristYaw, .*Hand, .*Finger.*, .*Thumb.*
        ]

        # Observations
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.actions.joint_pos.joint_names)
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.actions.joint_pos.joint_names)
        }
        self.observations.policy.foot_contact.params["sensor_cfg"].body_names = [".*_ankle"]

        # Weights and Parameters
        # -- Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.feet_air_time.weight = 2.0 # internal weight scales of command so this is the maximum
        self.rewards.feet_air_time.params["time_threshold"] = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[".*_ankle"]
        )
        self.rewards.feet_air_height.weight = 2.0
        self.rewards.feet_air_height.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[".*_ankle"]
        )
        self.rewards.feet_air_height.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*_ankle"]
        )
        self.rewards.feet_air_height.params["height_max_threshold"] = 0.12  # Max reward at 12cm height
        # -- Penalties
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[".*_ankle"]
        )
        self.rewards.dof_torques_l2.weight = -2.0e-8 # originally -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*Hip.*", ".*Knee.*"]
        )
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*Hip.*", ".*Knee.*"]
        )
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -0.9
        # -- -- Joint Limits and Deviations
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.joint_deviation_hip_roll.weight = -0.04
        self.rewards.joint_deviation_hip_roll.params["asset_cfg"].joint_names = [".*HipRoll"]
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_arms.params["asset_cfg"].joint_names = [
            ".*ShoulderPitch", ".*ShoulderRoll", ".*ElbowRoll", ".*ElbowYaw"
        ]
        self.rewards.joint_deviation_fingers.params["asset_cfg"].joint_names = [".*Finger.*",".*Thumb.*",]
        self.rewards.joint_deviation_fingers = None # I took out fingers
        self.rewards.joint_deviation_torso.weight = -0.04
        self.rewards.joint_deviation_torso.params["asset_cfg"].joint_names = [".*HipYawPitch"]
        # -- Events
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        # -- Terminations
        self.rewards.termination_penalty.weight = -5.0
        self.terminations.base_height.params["asset_cfg"].body_names = ["base_link"]
        self.terminations.base_height.params["minimum_height"] = 0.2

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
