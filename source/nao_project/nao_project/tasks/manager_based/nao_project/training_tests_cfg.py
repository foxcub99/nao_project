# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .nao_env_cfg import NaoEnvCfg

@configclass
class NaoEnvCfg1(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0


@configclass
class NaoEnvCfg2(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        
@configclass
class NaoEnvCfg3(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0

        
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

        