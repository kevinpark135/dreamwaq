"""Train a CENet-free PPO baseline on the DreamWaQ A1 rough task."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
import time
from datetime import datetime

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config


parser = argparse.ArgumentParser(
    description="Train a PPO baseline without DreamWaQ CENet features."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-DreamWaQ-A1-v0",
    help="Registered DreamWaQ rough task name.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of parallel environments.",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Number of learning iterations.",
)
parser.add_argument("--seed", type=int, default=None, help="Training seed.")
parser.add_argument(
    "--experiment_name",
    type=str,
    default="baseline_ppo_a1_rough",
    help="Log folder name under logs/rsl_rl.",
)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

# Hydra should only receive arguments that were not consumed by argparse.
sys.argv = [sys.argv[0]] + remaining_args


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> None:
    """Create the environment and train it with plain PPO."""
    with launch_simulation(env_cfg, args_cli):
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = min(args_cli.num_envs, 4096)
            if args_cli.num_envs > env_cfg.scene.num_envs:
                print(
                    f"[INFO] Capping num_envs from {args_cli.num_envs} "
                    f"to {env_cfg.scene.num_envs} for PhysX stability."
                )
        if args_cli.max_iterations is not None:
            agent_cfg.max_iterations = args_cli.max_iterations
        if args_cli.seed is not None:
            agent_cfg.seed = args_cli.seed

        installed_rsl_rl_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, installed_rsl_rl_version
        )

        # Baseline: same task, reward, terrain curriculum, and privileged critic,
        # but no CENet velocity/context features in the actor.
        agent_cfg.obs_groups = {
            "actor": ["policy"],
            "critic": ["critic"],
        }
        agent_cfg.experiment_name = args_cli.experiment_name

        env_cfg.seed = agent_cfg.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        agent_cfg.device = env_cfg.sim.device

        log_root = os.path.abspath(
            os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        )
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root, run_name)
        env_cfg.log_dir = log_dir

        print(f"[INFO] Logging baseline experiment in: {log_dir}")
        print("[INFO] Active runner: rsl_rl.runners.OnPolicyRunner")
        print("[INFO] Actor observations: policy only")
        print("[INFO] Critic observations: privileged critic")

        train_cfg = agent_cfg.to_dict()

        distribution_cfg = train_cfg.get("actor", {}).get("distribution_cfg")
        if isinstance(distribution_cfg, dict):
            distribution_cfg.pop("std_range", None)

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )

        runner = OnPolicyRunner(
            env,
            train_cfg,
            log_dir=log_dir,
            device=agent_cfg.device,
        )

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        start_time = time.time()
        try:
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
            )
            print(f"[INFO] Training time: {time.time() - start_time:.2f} seconds")
        finally:
            env.close()


if __name__ == "__main__":
    main()
