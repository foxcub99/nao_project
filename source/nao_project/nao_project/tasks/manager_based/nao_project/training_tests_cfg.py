# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import EventTermCfg

from . import mdp
from .nao_env_cfg import NaoEnvCfg


@configclass
class NaoEnvCfg1(NaoEnvCfg):
    # @configclass
    # class CurriculumCfg:
    #     reward_change_1 = None
    #     reward_change_2 = None
    #     episode_change_1 = None
    # curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        super().__post_init__()
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.rewards.track_lin_vel_xy_exp.weight = 3.0
        # self.rewards.track_ang_vel_z_exp.weight = 3.0
        # self.rewards.termination_penalty.weight = -0.1
        # self.rewards.feet_air_time_height.weight = 0.0
        # self.curriculum.episode_change_1 = CurriculumTermCfg(
        #     func=mdp.modify_env_param,
        #     params={
        #         "address": "cfg.episode_length_s",
        #         "modify_fn": mdp.increase_episode_length_gradually,
        #         "modify_params": {
        #             "initial_length": 3.0,   # Starting episode length
        #             "final_length": 15.0,    # Final episode length
        #             "start_steps": 2000,    # Start increasing at step 2000
        #             "end_steps": 32000      # Finish increasing at step 32000
        #         }
        #     }
        # )


@configclass
class NaoEnvCfg2(NaoEnvCfg):
    # @configclass
    # class CurriculumCfg:
    #     reward_change_1 = None
    #     reward_change_2 = None
    #     episode_change_1 = None
    # curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        super().__post_init__()
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.rewards.track_lin_vel_xy_exp.weight = 3.0
        # self.rewards.track_ang_vel_z_exp.weight = 3.0
        # self.rewards.termination_penalty.weight = -0.2
        # self.rewards.feet_air_time_height.weight = 0.0
        # self.curriculum.episode_change_1 = CurriculumTermCfg(
        #     func=mdp.modify_env_param,
        #     params={
        #         "address": "cfg.episode_length_s",
        #         "modify_fn": mdp.increase_episode_length_gradually,
        #         "modify_params": {
        #             "initial_length": 3.0,   # Starting episode length
        #             "final_length": 15.0,    # Final episode length
        #             "start_steps": 2000,    # Start increasing at step 2000
        #             "end_steps": 32000      # Finish increasing at step 32000
        #         }
        #     }
        # )
        # self.curriculum.reward_change_1 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "track_lin_vel_xy_exp",
        #         "weight": 6.0,
        #         "num_steps": 20000,
        #     },
        # )
        # self.curriculum.reward_change_2 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "termination_penalty",
        #         "weight": -0.05,
        #         "num_steps": 26000,
        #     },
        # )


@configclass
class NaoEnvCfg3(NaoEnvCfg):
    # @configclass
    # class CurriculumCfg:
    #     reward_change_1 = None
    #     reward_change_2 = None
    #     episode_change_1 = None
    #     command_change_1 = None
    #     command_change_2 = None
    #     command_change_3 = None
    # curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        super().__post_init__()


#         self.commands.base_velocity.ranges.lin_vel_x = (-0.15, 0.2)
#         self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
#         self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
#         self.rewards.track_lin_vel_xy_exp.weight = 3.0
#         self.rewards.track_ang_vel_z_exp.weight = 3.0
#         self.rewards.termination_penalty.weight = -0.1
#         self.rewards.feet_air_time_height.weight = 0.0
#         self.curriculum.episode_change_1 = CurriculumTermCfg(
#             func=mdp.modify_env_param,
#             params={
#                 "address": "cfg.episode_length_s",
#                 "modify_fn": mdp.increase_episode_length_gradually,
#                 "modify_params": {
#                     "initial_length": 3.0,   # Starting episode length
#                     "final_length": 15.0,    # Final episode length
#                     "start_steps": 2000,    # Start increasing at step 2000
#                     "end_steps": 32000      # Finish increasing at step 32000
#                 }
#             }
#         )
#         # Stage 2
#         self.curriculum.reward_change_1 = CurriculumTermCfg(
#             func=mdp.modify_reward_weight,
#             params={
#                 "term_name": "track_lin_vel_xy_exp",
#                 "weight": 6.0,
#                 "num_steps": 36000,
#             },
#         )
#         self.curriculum.reward_change_2 = CurriculumTermCfg(
#             func=mdp.modify_reward_weight,
#             params={
#                 "term_name": "track_ang_vel_z_exp",
#                 "weight": 6.0,
#                 "num_steps": 36000,
#             },
#         )
#         self.curriculum.command_change_1 = CurriculumTermCfg(
#             func=mdp.modify_term_cfg,
#             params={
#                 "address": "commands.base_velocity.ranges.lin_vel_x",
#                 "modify_fn": mdp.change_command_ranges,
#                 "modify_params": {
#                     "value": (-0.3, 0.4),
#                     "num_steps": 36000,
#                 }
#             }
#         )
#         self.curriculum.command_change_2 = CurriculumTermCfg(
#             func=mdp.modify_term_cfg,
#             params={
#                 "address": "commands.base_velocity.ranges.lin_vel_y",
#                 "modify_fn": mdp.change_command_ranges,
#                 "modify_params": {
#                     "value": (-0.3, 0.3),
#                     "num_steps": 36000,
#                 }
#             }
#         )
#         self.curriculum.command_change_3 = CurriculumTermCfg(
#             func=mdp.modify_term_cfg,
#             params={
#                 "address": "commands.base_velocity.ranges.ang_vel_z",
#                 "modify_fn": mdp.change_command_ranges,
#                 "modify_params": {
#                     "value": (-1.0, 1.0),
#                     "num_steps": 36000,
#                 }
#             }
#         )


@configclass
class NaoEnvCfg4(NaoEnvCfg):
    # @configclass
    # class CurriculumCfg:
    #     reward_change_1 = None
    #     reward_change_2 = None
    #     episode_change_1 = None

    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.rewards.track_lin_vel_xy_exp.weight = 5.0
        # self.rewards.track_ang_vel_z_exp.weight = 5.0
        # self.rewards.termination_penalty.weight = -0.1
        # # Stage1
        # self.curriculum.episode_change_1 = CurriculumTermCfg(
        #     func=mdp.modify_env_param,
        #     params={
        #         "address": "cfg.episode_length_s",
        #         "modify_fn": mdp.increase_episode_length_gradually,
        #         "modify_params": {
        #             "initial_length": 3.0,  # Starting episode length
        #             "final_length": 20.0,  # Final episode length
        #             "start_steps": 2000,  # Start increasing at step 2000
        #             "end_steps": 72000,  # Finish increasing at step 72000
        #         },
        #     },
        # )


@configclass
class NaoEnvCfg5(NaoEnvCfg):
    @configclass
    class CurriculumCfg:
        event_change_1 = None
        event_change_2 = None
        episode_change_1 = None

    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.termination_penalty.weight = -1.0
        self.reset_robot_joints = EventTermCfg(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (-0.7, 0.7),
                "velocity_range": (-0.2, 0.2),
            },
        )
        self.curriculum.episode_change_1 = CurriculumTermCfg(
            func=mdp.modify_env_param,
            params={
                "address": "cfg.episode_length_s",
                "modify_fn": mdp.increase_episode_length_gradually,
                "modify_params": {
                    "initial_length": 3.0,  # Starting episode length
                    "final_length": 15.0,  # Final episode length
                    "start_steps": 2000,  # Start increasing at step 2000
                    "end_steps": 32000,  # Finish increasing at step 32000
                },
            },
        )


@configclass
class NaoEnvCfg6(NaoEnvCfg):
    @configclass
    class CurriculumCfg:
        event_change_1 = None
        event_change_2 = None
        episode_change_1 = None
        episode_change_2 = None
        episode_change_3 = None
        reward_change_1 = None
        reward_change_2 = None
        reward_change_3 = None
        reward_change_4 = None

    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.rewards.termination_penalty.weight = -0.1
        # Stage
        self.curriculum.command_change_1 = CurriculumTermCfg(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.change_command_ranges,
                "modify_params": {
                    "value": (-0.2, 0.3),
                    "num_steps": 8000,
                },
            },
        )
        self.curriculum.episode_change_1 = CurriculumTermCfg(
            func=mdp.modify_env_param,
            params={
                "address": "cfg.episode_length_s",
                "modify_fn": mdp.change_episode_length_s,
                "modify_params": {"new_length": 6.0, "num_steps": 8000},
            },
        )
        self.curriculum.command_change_2 = CurriculumTermCfg(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_y",
                "modify_fn": mdp.change_command_ranges,
                "modify_params": {
                    "value": (-0.1, 0.1),
                    "num_steps": 12000,
                },
            },
        )
        self.curriculum.reward_change_1 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_lin_vel_xy_exp",
                "weight": 2.0,
                "num_steps": 16000,
            },
        )
        self.curriculum.command_change_3 = CurriculumTermCfg(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.ang_vel_z",
                "modify_fn": mdp.change_command_ranges,
                "modify_params": {
                    "value": (-1.0, 1.0),
                    "num_steps": 20000,
                },
            },
        )
        self.curriculum.episode_change_2 = CurriculumTermCfg(
            func=mdp.modify_env_param,
            params={
                "address": "cfg.episode_length_s",
                "modify_fn": mdp.change_episode_length_s,
                "modify_params": {"new_length": 8.0, "num_steps": 20000},
            },
        )
        self.curriculum.reward_change_2 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_ang_vel_z_exp",
                "weight": 2.0,
                "num_steps": 24000,
            },
        )
        self.curriculum.reward_change_3 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "feet_air_time_height",
                "weight": 0.0,
                "num_steps": 28000,
            },
        )
        self.curriculum.command_change_4 = CurriculumTermCfg(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.change_command_ranges,
                "modify_params": {
                    "value": (-0.3, 0.6),
                    "num_steps": 30000,
                },
            },
        )
        self.curriculum.episode_change_3 = CurriculumTermCfg(
            func=mdp.modify_env_param,
            params={
                "address": "cfg.episode_length_s",
                "modify_fn": mdp.change_episode_length_s,
                "modify_params": {"new_length": 10.0, "num_steps": 30000},
            },
        )
        self.curriculum.reward_change_4 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_lin_vel_xy_exp",
                "weight": 3.0,
                "num_steps": 32000,
            },
        )
