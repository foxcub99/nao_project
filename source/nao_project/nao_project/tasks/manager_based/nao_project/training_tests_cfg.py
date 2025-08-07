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
class NaoEnvCfg1(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class NaoEnvCfg2(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.rewards.feet_air_time_height.weight = 0.0

        
@configclass
class NaoEnvCfg3(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)




        
@configclass
class NaoEnvCfg4(NaoEnvCfg):
    class CurriculumCfg:
        def __init__(self):
            reward_change_1 = None
            reward_change_2 = None
            episode_change_1 = None
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Stage1
        self.curriculum.reward_change_1 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_lin_vel_xy_exp",
                "weight": 2.0,
                "num_steps": 8000,
            },
        )
        self.curriculum.reward_change_2 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_ang_vel_z_exp",
                "weight": 2.0,
                "num_steps": 8000,
            },
        )
        self.curriculum.episode_change_1 = CurriculumTermCfg(
            func=mdp.change_episode_length_s,
            params={
                "new_length": 8.0,
                "num_steps": 12000,
            }
        )



@configclass
class NaoEnvCfg5(NaoEnvCfg):
    class CurriculumCfg:
        def __init__(self):
            episode_change_1 = None
            episode_change_2 = None
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.feet_air_time_height.weight = 0.0
        self.curriculum.episode_change_1 = CurriculumTermCfg(
            func=mdp.change_episode_length_s,
            params={
                "new_length": 6.0,
                "num_steps": 8000,
            }
        )
        self.curriculum.episode_change_2 = CurriculumTermCfg(
            func=mdp.change_episode_length_s,
            params={
                "new_length": 12.0,
                "num_steps": 20000,
            }
        )
        
      

        
@configclass
class NaoEnvCfg6(NaoEnvCfg):
    class CurriculumCfg:
        def __init__(self):
            command_change_1 = None
            command_change_2 = None
            command_change_3 = None
            command_change_4 = None
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Stage
        self.curriculum.command_change_1 = CurriculumTermCfg(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity",
                "term_name": "ranges.lin_vel_x",
                "value": (-0.2, 0.3),
                "num_steps": 8000,
            }
        )
        self.curriculum.episode_change_1 = CurriculumTermCfg(
            func=mdp.change_episode_length_s,
            params={
                "new_length": 6.0,
                "num_steps": 8000,
            }
        )
        self.curriculum.command_change_2 = CurriculumTermCfg(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity",
                "term_name": "ranges.lin_vel_y",
                "value": (-0.1, 0.1),
                "num_steps": 12000,
            }
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
                "address": "commands.base_velocity",
                "term_name": "ranges.ang_vel_z",
                "value": (-1.0, 1.0),
                "num_steps": 20000,
            }
        )
        self.curriculum.episode_change_2 = CurriculumTermCfg(
            func=mdp.change_episode_length_s,
            params={
                "new_length": 8.0,
                "num_steps": 20000,
            }
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
                "address": "commands.base_velocity",
                "term_name": "ranges.lin_vel_x",
                "value": (-0.3, 0.6),
                "num_steps": 30000,
            }
        )
        self.curriculum.episode_change_3 = CurriculumTermCfg(
            func=mdp.change_episode_length_s,
            params={
                "new_length": 10.0,
                "num_steps": 30000,
            }
        )
        self.curriculum.reward_change_4 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_lin_vel_xy_exp",
                "weight": 3.0,
                "num_steps": 32000,
            },
        )
