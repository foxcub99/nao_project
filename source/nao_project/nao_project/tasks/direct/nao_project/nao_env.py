# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from .nao_env_cfg import NaoEnvCfg
from .nao_locomotion_env import NaoLocomotionEnv


class NaoEnv(NaoLocomotionEnv):  # Rename class
    cfg: NaoEnvCfg

    def __init__(self, cfg: NaoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

