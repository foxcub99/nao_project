# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg

from . import mdp

from .nao_env_cfg import NaoEnvCfg


@configclass
class NaoCurriculumEnvCfg(NaoEnvCfg):
    @configclass
    class CurriculumCfg:
        reward_change_1 = None
        reward_change_2 = None
        reward_change_3 = None
        reward_change_4 = None
        reward_change_5 = None
        reward_change_6 = None
        reward_change_air1 = None
        reward_change_air2 = None
        reward_change_air3 = None
        reward_change_air4 = None
        reward_change_air5 = None
        reward_change_air6 = None
        command_change_1 = None
        command_change_2 = None
        command_change_3 = None
        command_change_4 = None
        command_change_5 = None
        command_change_6 = None
        command_change_7 = None
        command_change_8 = None
        episode_change_1 = None
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        """Since we want NaoSpecificEnvCfg post init to run as the initial values for curriculum"""
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.termination_penalty.weight = -0.3
        self.rewards.feet_air_time_height.weight = 0.0
        self.curriculum.episode_change_1 = CurriculumTermCfg(
            func=mdp.modify_env_param,
            params={
                "address": "cfg.episode_length_s",
                "modify_fn": mdp.increase_episode_length_gradually,
                "modify_params": {
                    "initial_length": 3.0,   # Starting episode length
                    "final_length": 15.0,    # Final episode length
                    "start_steps": 2000,    # Start increasing at step 2000
                    "end_steps": 32000      # Finish increasing at step 32000
                }
            }
        )
        # Stage 1
        # self.command_change_1 = CurriculumTermCfg(
        #     func=mdp.modify_term_cfg,
        #     params={
        #         "address": "commands.base_velocity",
        #         "term_name": "ranges.lin_vel_x",
        #         "value": (0, 0.1),
        #         "num_steps": 0,
        #     }
        # )

        # # Stage 2
        # self.episode_change_1 = CurriculumTermCfg(
        #     func=mdp.change_episode_length_s,
        #     params={
        #         "new_length": 6.0,
        #         "num_steps": 8000,
        #     }
        # )
        # self.command_change_2 = CurriculumTermCfg(
        #     func=mdp.modify_term_cfg,
        #     params={
        #         "address": "commands.base_velocity",
        #         "term_name": "ranges.lin_vel_x",
        #         "value": (0, 0.2),
        #         "num_steps": 8000,
        #     }
        # )
        # self.reward_change_1 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "track_lin_vel_xy_exp",
        #         "weight": 2.0,
        #         "num_steps": 8000,
        #     },
        # )
        # self.reward_change_air1 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "feet_air_time",
        #         "weight": 1.5,
        #         "num_steps": 8000,
        #     },
        # )
        # self.reward_change_air2 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "feet_air_height",
        #         "weight": 1.5,
        #         "num_steps": 8000,
        #     },
        # )

        # # Stage 3
        # self.episode_change_2 = CurriculumTermCfg(
        #     func=mdp.change_episode_length_s,
        #     params={
        #         "new_length": 8.0,
        #         "num_steps": 12000,
        #     }
        # )
        # self.command_change_3 = CurriculumTermCfg(
        #     func=mdp.modify_term_cfg,
        #     params={
        #         "address": "commands.base_velocity",
        #         "term_name": "ranges.lin_vel_x",
        #         "value": (0, 0.3),
        #         "num_steps": 12000,
        #     }
        # )
        # self.reward_change_2 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "track_lin_vel_xy_exp",
        #         "weight": 3.0,
        #         "num_steps": 12000,
        #     },
        # )

        # # Stage 4
        # self.episode_change_3 = CurriculumTermCfg(
        #     func=mdp.change_episode_length_s,
        #     params={
        #         "new_length": 10.0,
        #         "num_steps": 18000,
        #     }
        # )
        # self.command_change_4 = CurriculumTermCfg(
        #     func=mdp.modify_term_cfg,
        #     params={
        #         "address": "commands.base_velocity",
        #         "term_name": "ranges.lin_vel_x",
        #         "value": (-0.2, 0.3),
        #         "num_steps": 18000,
        #     }
        # )
        # self.reward_change_air3 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "feet_air_time",
        #         "weight": 1.0,
        #         "num_steps": 18000,
        #     },
        # )
        # self.reward_change_air4 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "feet_air_height",
        #         "weight": 1.0,
        #         "num_steps": 18000,
        #     },
        # )

        # # Stage 5
        # self.command_change_5 = CurriculumTermCfg(
        #     func=mdp.modify_term_cfg,
        #     params={
        #         "address": "commands.base_velocity",
        #         "term_name": "ranges.lin_vel_y",
        #         "value": (-0.2, 0.2),
        #         "num_steps": 24000,
        #     }
        # )
        # self.command_change_6 = CurriculumTermCfg(
        #     func=mdp.modify_term_cfg,
        #     params={
        #         "address": "commands.base_velocity",
        #         "term_name": "ranges.lin_vel_x",
        #         "value": (-0.05, 0.1),
        #         "num_steps": 24000,
        #     }
        # )
        # self.reward_change_3 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "track_lin_vel_xy_exp",
        #         "weight": 3.0,
        #         "num_steps": 24000,
        #     },
        # )
        # self.reward_change_4 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "joint_deviation_hip_roll",
        #         "weight": -4e-7,
        #         "num_steps": 24000,
        #     },
        # )

        # # Stage 6
        # self.episode_change_1 = CurriculumTermCfg(
        #     func=mdp.change_episode_length_s,
        #     params={
        #         "new_length": 15.0,
        #         "num_steps": 30000,
        #     }
        # )
        # self.reward_change_5 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "track_lin_vel_xy_exp",
        #         "weight": 1.0,
        #         "num_steps": 30000,
        #     },
        # )
        # self.reward_change_6 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "track_ang_vel_z_exp",
        #         "weight": 3.0,
        #         "num_steps": 30000,
        #     },
        # )
        # self.command_change_7 = CurriculumTermCfg(
        #     func=mdp.modify_term_cfg,
        #     params={
        #         "address": "commands.base_velocity",
        #         "term_name": "ranges.ang_vel_z",
        #         "value": (-1.0, 1.0),
        #         "num_steps": 30000,
        #     }
        # )
        # self.command_change_8 = CurriculumTermCfg(
        #     func=mdp.modify_term_cfg,
        #     params={
        #         "address": "commands.base_velocity",
        #         "term_name": "ranges.lin_vel_x",
        #         "value": (-0.2, 0.3),
        #         "num_steps": 30000,
        #     }
        # )
        # self.reward_change_air5 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "feet_air_time",
        #         "weight": 1.5,
        #         "num_steps": 30000,
        #     },
        # )
        # self.reward_change_air6 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "feet_air_height",
        #         "weight": 1.5,
        #         "num_steps": 30000,
        #     },
        # )
        # self.reward_change_7 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "joint_deviation_hip_roll",
        #         "weight": -0.04,
        #         "num_steps": 30000,
        #     },
        # )
        # self.reward_change_8 = CurriculumTermCfg(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "joint_deviation_torso",
        #         "weight": 4e-7,
        #         "num_steps": 30000,
        #     },
        # )