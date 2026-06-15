"""Play a trained DreamWaQ policy with CENet."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config


def resolve_entry_point(entry_point: str):
    module_name, object_name = entry_point.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


parser = argparse.ArgumentParser(
    description="Play DreamWaQ with PPO and CENet."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-DreamWaQ-A1-Play-v0",
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to a DreamWaQ checkpoint.",
)
parser.add_argument("--real-time", action="store_true")
add_launcher_args(parser)

args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    with launch_simulation(env_cfg, args_cli):
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs

        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg,
            installed_version,
        )

        agent_cfg.obs_groups = {
            "actor": ["policy", "cenet"],
            "critic": ["critic"],
        }

        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        agent_cfg.device = env_cfg.sim.device
        env_cfg.seed = agent_cfg.seed

        log_root = os.path.abspath(
            os.path.join(
                "logs",
                "rsl_rl",
                agent_cfg.experiment_name,
            )
        )

        if args_cli.checkpoint is not None:
            checkpoint_path = retrieve_file_path(args_cli.checkpoint)
        else:
            checkpoint_path = get_checkpoint_path(
                log_root,
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )

        env_cfg.log_dir = os.path.dirname(checkpoint_path)

        task_spec = gym.spec(args_cli.task)
        runner_entry_point = task_spec.kwargs.get(
            "dreamwaq_runner_entry_point"
        )

        if runner_entry_point is None:
            raise ValueError(
                f"Task '{args_cli.task}' does not define "
                "'dreamwaq_runner_entry_point'."
            )

        runner_class = resolve_entry_point(runner_entry_point)

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )

        runner = runner_class(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )

        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        print(
            "[INFO] Active runner: "
            f"{runner.__class__.__module__}."
            f"{runner.__class__.__name__}"
        )

        runner.load(checkpoint_path)

        policy = runner.get_inference_policy(
            device=env.unwrapped.device
        )
        runner.cenet.eval()

        obs = env.get_observations().to(agent_cfg.device)

        with torch.inference_mode():
            obs = runner._add_cenet_features(obs)

        dt = env.unwrapped.step_dt

        try:
            while True:
                start_time = time.time()

                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    obs = obs.to(agent_cfg.device)
                    obs = runner._add_cenet_features(obs)
                    policy.reset(dones)

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            env.close()


if __name__ == "__main__":
    main()