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


@configclass
class NaoEnvCfg2(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
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
        self.curriculum.reward_change_3 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_lin_vel_xy_exp",
                "weight": 3.0,
                "num_steps": 16000,
            },
        )
        self.curriculum.reward_change_4 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_ang_vel_z_exp",
                "weight": 3.0,
                "num_steps": 16000,
            },
        )

        
@configclass
class NaoEnvCfg3(NaoEnvCfg):
    class CurriculumCfg:
        def __init__(self):
            self.reward_change_1 = None
            self.reward_change_2 = None
            self.reward_change_3 = None
            self.reward_change_4 = None
            self.reward_change_5 = None
            self.reward_change_6 = None
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        super().__post_init__()
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
        self.curriculum.reward_change_3 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_lin_vel_xy_exp",
                "weight": 3.0,
                "num_steps": 16000,
            },
        )
        self.curriculum.reward_change_4 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "track_ang_vel_z_exp",
                "weight": 3.0,
                "num_steps": 16000,
            },
        )
        self.curriculum.reward_change_5 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "feet_air_time",
                "weight": 0.0,
                "num_steps": 24000,
            },
        )
        self.curriculum.reward_change_6 = CurriculumTermCfg(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "feet_air_height",
                "weight": 0.0,
                "num_steps": 24000,
            },
        )

        
@configclass
class NaoEnvCfg4(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()



@configclass
class NaoEnvCfg5(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        
@configclass
class NaoEnvCfg6(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        