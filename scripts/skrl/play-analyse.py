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

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
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

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
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
        plt.suptitle(f"Nao Manager v5.{args_cli.iteration-1}", y=0.98, fontsize=16)
        plt.savefig(f"plots/mgr-bi2/v5-{args_cli.iteration-1}.png")
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
