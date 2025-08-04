# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

from .nao_env import NaoEnv, NaoEnvCfg

##
# Register Gym environments.
##
gym.register(
    id="Nao-Direct",
    entry_point=f"{__name__}.nao_env:NaoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NaoEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
