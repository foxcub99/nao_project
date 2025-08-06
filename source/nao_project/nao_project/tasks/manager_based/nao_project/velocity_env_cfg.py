# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg
from isaaclab.managers import ObservationTermCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg

from . import mdp


##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    robot: ArticulationCfg = MISSING

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    contact_forces = ContactSensorCfg(
            prim_path=MISSING,
            history_length=1,
            track_air_time=True,
        )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=AdditiveUniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveUniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=AdditiveUniformNoiseCfg(n_min=-1.5, n_max=1.5))
        actions = ObservationTermCfg(func=mdp.last_action)
        foot_contact = ObservationTermCfg(
            func=mdp.foot_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle"]),
            }
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Startup
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    # Reset
    base_external_force_torque = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    # Interval
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Rewards
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle"]),
            "time_threshold": 0.25, # reward = single stance air time
        },
    )
    feet_air_height = RewardTermCfg(
        func=mdp.feet_air_height,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle"]),
            "height_max_threshold": 0.25,  # clamp reward at this height
        },
    )
    # Penalties
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    feet_slide = RewardTermCfg(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle"]),
        },
    )
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=0.0)
    # -- Joint limits
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*AnklePitch", ".*AnkleRoll"])},
    )
    # -- Joint deviation 
    joint_deviation_hip_roll = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipRoll"]),
        },
    )
    joint_deviation_torso = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipYawPitch"]),
        },
    )
    joint_deviation_arms = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*ShoulderPitch",
                    ".*ShoulderRoll",
                    ".*ElbowRoll",
                    ".*ElbowYaw",
                    ".*WristYaw", # unsure if these are needed
                    ".*Hand.*",
                ],
            )
        },
    )
    joint_deviation_fingers = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*Finger.*", # unsure again
                    ".*Thumb.*",
                ],
            )
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "minimum_height": 0.2,  # Terminate if base goes below 30cm
        },
    )
    


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Scene

    # Commands

    # Actions

    # Observations
    
    # Events

    # Rewards
    # -- Rewards
    # -- Penalties

    
    # # NEW: Reward weight curriculum - gradually shift focus from stability to performance
    # reward_weights = CurriculumTermCfg(
    #     func=mdp.modify_reward_weights,
    #     params={
    #         "initial_weights": {
    #             "track_lin_vel_xy_exp": 0.5,  # Start with lower velocity tracking importance
    #             "track_ang_vel_z_exp": 0.3,   # Lower angular velocity importance
    #             "feet_air_time": 1.0,         # High importance on proper stepping
    #             "flat_orientation_l2": 1.2,   # High importance on stability
    #             "action_rate_l2": 0.02,       # Higher penalty on jerky movements
    #         },
    #         "final_weights": {
    #             "track_lin_vel_xy_exp": 1.3,  # End with higher velocity tracking
    #             "track_ang_vel_z_exp": 1.0,   # Higher angular velocity importance
    #             "feet_air_time": 0.75,        # Reduced stepping importance
    #             "flat_orientation_l2": 0.8,   # Reduced stability penalty
    #             "action_rate_l2": 0.005,      # Lower penalty on movements
    #         },
    #     }
    # )
    
    # # NEW: Command difficulty curriculum - start with easier commands
    # command_difficulty = CurriculumTermCfg(
    #     func=mdp.modify_command_ranges,
    #     params={
    #         "initial_ranges": {
    #             "lin_vel_x": (0.0, 0.5),     # Start with forward walking only
    #             "lin_vel_y": (-0.2, 0.2),    # Limited lateral movement
    #             "ang_vel_z": (-0.5, 0.5),    # Limited turning
    #         },
    #         "final_ranges": {
    #             "lin_vel_x": (-0.5, 1.0),    # Full velocity range
    #             "lin_vel_y": (-0.5, 0.5),    # Full lateral movement
    #             "ang_vel_z": (-1.0, 1.0),    # Full turning range
    #         },
    #     }
    # )


##
# Environment configuration
##


@configclass
class LocomotionVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005 # TODO match this with nao
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
