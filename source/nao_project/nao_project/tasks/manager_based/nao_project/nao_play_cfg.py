# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg

from . import mdp
from .nao_env_cfg import NaoEnvCfg
# from .nao_curriculum_env_cfg import NaoCurriculumEnvCfg

@configclass
class NaoEnvCfg_PLAY(NaoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class NaoEnvCfg_PLAYFORWARD(NaoEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class NaoEnvCfg_PLAYSIDEWAYS(NaoEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
@configclass
class NaoEnvCfg_PLAYTURN(NaoEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)

