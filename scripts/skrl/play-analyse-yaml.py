# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from matplotlib import pyplot as plt

from isaaclab.app import AppLauncher

import yaml
from copy import deepcopy
import pickle



# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--iteration", type=int, default=0, help="Iteration number for run")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
# from isaaclab.utils.dict import ConfigDict
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import nao_project.tasks  # noqa: F401

from source.nao_project.nao_project.assets.nao import NAO_CFG

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    """Play with skrl agent using saved agent/env configs from a run folder."""
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if not args_cli.checkpoint:
        raise ValueError("You must provide --checkpoint pointing to a trained model checkpoint.")

    import yaml

    # --------------------------------------------------
    # Locate run directory and params dir
    # --------------------------------------------------
    resume_path = os.path.abspath(args_cli.checkpoint)  # e.g., ...\2025-08-07_11-21-51_ppo_torch\checkpoints\agent420.pt
    run_dir = os.path.dirname(os.path.dirname(resume_path))  # goes up from checkpoints -> run folder
    params_dir = os.path.join(run_dir, "params")  # run folder + "params"
    agent_yaml_path = os.path.join(params_dir, "agent.pkl")
    env_yaml_path = os.path.join(params_dir, "env.pkl")

    print("[INFO] Check agent.yaml exists at:", agent_yaml_path)
    print("[INFO] Check env.yaml exists at:", env_yaml_path)

    if not os.path.exists(agent_yaml_path):
        raise FileNotFoundError(f"{agent_yaml_path} does not exist.")
    if not os.path.exists(env_yaml_path):
        raise FileNotFoundError(f"{env_yaml_path} does not exist.")
    log_dir = run_dir

    # --------------------------------------------------
    # Load agent.yaml (exact training config)
    # --------------------------------------------------
    # agent_yaml_path = os.path.join(params_dir, "agent.pkl")
    # with open(agent_yaml_path, "rb") as f:
        # experiment_cfg = pickle.load(f)
    # with open(env_yaml_path, "r") as f:
    # env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")


    # --------------------------------------------------
    # Load env.yaml and override commands
    # --------------------------------------------------
    env_yaml_path = os.path.join(params_dir, "env.pkl")
    with open(env_yaml_path, "rb") as f:
        env_cfg = pickle.load(f)

    # Override ONLY the commands section
    env_cfg.episode_length_s = 10.0

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    # Runtime overrides
    # env_cfg["sim"]["device"] = args_cli.device
    if args_cli.num_envs:
        env_cfg.scene.num_envs = args_cli.num_envs
    # env_cfg.scene.robot.prim_path = "/World/envs/env_.*/Robot"
    env_cfg.scene.robot = NAO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # env_cfg = ConfigDict(env_cfg)

    # --------------------------------------------------
    # Create environment from saved config
    # --------------------------------------------------
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None
    )

    # Convert to single agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv) and args_cli.algorithm.lower() in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    if args_cli.ml_framework.startswith("torch"):
        from skrl.utils.runner.torch import Runner
    elif args_cli.ml_framework.startswith("jax"):
        from skrl.utils.runner.jax import Runner

    # Instantiate agent from config
    # agent_class = getattr(skrl.agents.torch, args_cli.algorithm.upper())
    # agent = agent_class(
    #     models=None,  # built automatically
    #     memory=None,
    #     cfg=experiment_cfg["agent"]["config"],
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     device=args_cli.device
    #     )


    # Load checkpoint
    # agent.load(resume_path)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)


    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------
    # runner = Runner(env=env, agents=agent)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    timesteperooni = 0

    # episode_length_s = 10
    # sdt = 0.004
    # decimation = 3
    # max_timesteps = int(episode_length_s / sdt))
    reset_count = 0
    import matplotlib.pyplot as plt

    def save_plots(speed, planar_speed, total_speed, top_speed):
        # Convert deques of tensors on CUDA to lists of floats on CPU
        speed_list = [x.cpu().item() if hasattr(x, 'cpu') else float(x) for x in speed]
        planar_speed_list = [x.cpu().item() if hasattr(x, 'cpu') else float(x) for x in planar_speed]
        total_speed_list = [x.cpu().item() if hasattr(x, "cpu") else float(x) for x in total_speed]


        average_speed = [sum(speed_list[:i+1])/(i+1) for i in range(len(speed_list))]
        average_planar_speed = [sum(planar_speed_list[:i+1])/(i+1) for i in range(len(planar_speed_list))]
        average_total_speed = [sum(total_speed_list[:i+1])/(i+1) for i in range(len(total_speed_list))]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 2, 1)
        plt.plot(speed_list, label='X Speed')
        plt.title('X Speed')
        plt.xlabel('Timestep')
        plt.ylabel('Speed (m/s)')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(average_speed, label='Average X Speed', color='orange')
        plt.title('Average X Speed')
        plt.xlabel('Timestep')
        plt.ylabel('Speed (m/s)')
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(planar_speed_list, label='Planar Speed', color='green')
        plt.title('Planar Speed')
        plt.xlabel('Timestep')
        plt.ylabel('Speed (m/s)')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(average_planar_speed, label='Average Planar Speed', color='red')
        plt.title('Average Planar Speed')
        plt.xlabel('Timestep')
        plt.ylabel('Speed (m/s)')
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(total_speed_list, label='Total Speed', color='blue')
        plt.title('Total X Speed')
        plt.xlabel('Timestep')
        plt.ylabel('Speed (m/s)')
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(average_total_speed, label='Average Total Speed', color='purple')
        plt.title('Average Total X Speed')
        plt.xlabel('Timestep')
        plt.ylabel('Speed (m/s)')
        plt.legend()

        plt.tight_layout()
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])  # leave bottom 5% for footnote
        plt.figtext(0.5, 0.01, f"Top Speed: {top_speed:.2f}m/s", ha='center')
        plt.suptitle(f"Nao Manager v3.{args_cli.iteration-1}", y=0.98, fontsize=16)
        if args_cli.checkpoint:
            date_dir = os.path.dirname(os.path.dirname(args_cli.checkpoint))
        else:
            date_dir = "plots/mismatched/"
        plt.savefig(f"{date_dir}/plot3.png")
        # plt.savefig(f"plots/mgr-bi2/v5-{args_cli.iteration-1}.png")
        plt.close()
        print("[INFO] Saved speed plots to speed_plots.png")


    from collections import deque
    top_speed = 0
    top_planar_speed = 0
    planar_speed = deque(maxlen=100)  # store last 100 planar
    speed = deque(maxlen=100)  # store last 100 speeds
    total_ep_speed = []

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

            # base_lin_vel = env.unwrapped.vel_loc[0]
            active_terms = env.unwrapped.observation_manager.get_active_iterable_terms(0)

            # Build dict from active terms
            obs_dict = {name: val for name, val in active_terms}

            base_lin_vel = obs_dict["policy-base_lin_vel"]
            
            # Detect environment reset when speed goes to 0 (ignore first time)
            if abs(base_lin_vel[0]) < 1e-3:  # near zero
                save_plots(speed, planar_speed, total_ep_speed, top_speed)
                env.close()
                print("[INFO] Environment closed after second reset.")
                break

            # If mgr
            speed.append(base_lin_vel[0])
            total_ep_speed.append(base_lin_vel[0])
            planar_speed.append((base_lin_vel[0]**2 + base_lin_vel[1]**2)**0.5)
            average_speed = sum(speed) / len(speed) if speed else 0.0
            if base_lin_vel[0] > top_speed:
                top_speed = base_lin_vel[0]

            # If dir
            # if base_lin_vel[0] > top_speed:
            #     top_speed = base_lin_vel[0]
            # if (base_lin_vel[0]**2 + base_lin_vel[1]**2)**0.5 > top_planar_speed:
            #     top_planar_speed = (base_lin_vel[0]**2 + base_lin_vel[1]**2)**0.5
            # speed.append(base_lin_vel[0])
            # total_ep_speed.append(base_lin_vel[0])
            # planar_speed.append((base_lin_vel[0]**2 + base_lin_vel[1]**2)**0.5)
            # average_speed = sum(speed) / len(speed) if speed else 0.0
            # average_planar_speed = sum(planar_speed) / len(planar_speed) if planar_speed else 0.0
            # print(f"Step: {timesteperooni}, Current speed: {base_lin_vel[0]:.2f}, Current y speed: {base_lin_vel[1]:.2f}, Top X Speed: {top_speed:.2f}, Average X Speed: {average_speed:.2f}, Average p speed: {average_planar_speed:.2f}, top p speed: {top_planar_speed:.2f}, hit max avg {len(speed) == speed.maxlen}")



        # increment timestep
        timesteperooni += 1

        # if timestep >= max_timesteps:
        #     env.close()
        #     print("[INFO] Episode finished. Closing environment.")

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
