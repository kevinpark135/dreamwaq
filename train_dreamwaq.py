"""Train the DreamWaQ task with its custom RSL-RL runner."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import os
import sys
import time
from datetime import datetime

import gymnasium as gym

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


def resolve_entry_point(entry_point: str):
    """Resolve a ``package.module:ClassName`` entry point."""
    module_name, object_name = entry_point.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


parser = argparse.ArgumentParser(description="Train DreamWaQ with CENet.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-DreamWaQ-A1-v0",
    help="Registered DreamWaQ task name.",
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
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

# Hydra should only receive arguments that were not consumed by argparse.
sys.argv = [sys.argv[0]] + remaining_args


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> None:
    """Create the environment and train it with DreamWaQRunner."""
    with launch_simulation(env_cfg, args_cli):
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.max_iterations is not None:
            agent_cfg.max_iterations = args_cli.max_iterations
        if args_cli.seed is not None:
            agent_cfg.seed = args_cli.seed

        installed_rsl_rl_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, installed_rsl_rl_version
        )
        agent_cfg.obs_groups = {
            "actor": ["policy", "cenet"],
            "critic": ["critic"],
        }

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

        task_spec = gym.spec(args_cli.task)
        runner_entry_point = task_spec.kwargs.get("dreamwaq_runner_entry_point")
        if runner_entry_point is None:
            raise ValueError(
                f"Task '{args_cli.task}' does not define "
                "'dreamwaq_runner_entry_point'."
            )
        runner_class = resolve_entry_point(runner_entry_point)

        print(f"[INFO] Logging experiment in: {log_dir}")
        print(f"[INFO] DreamWaQ runner entry point: {runner_entry_point}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = runner_class(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )

        print(
            "[INFO] Active runner: "
            f"{runner.__class__.__module__}.{runner.__class__.__name__}"
        )
        print(f"[INFO] CENet: {runner.cenet.__class__.__name__}")

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
