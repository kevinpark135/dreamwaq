"""DreamWaQ curriculum terms."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import SceneEntityCfg


def _as_env_ids(env, env_ids: Sequence[int] | slice) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _episode_return(env, env_ids: torch.Tensor) -> torch.Tensor:
    """Return the current per-env episodic reward sum before RewardManager reset."""
    reward_sums = getattr(env.reward_manager, "_episode_sums", None)
    if not reward_sums:
        return torch.zeros(env_ids.numel(), device=env.device)

    total_return = torch.zeros(env_ids.numel(), device=env.device)
    for value in reward_sums.values():
        total_return += value[env_ids]
    return total_return


def dreamwaq_terrain_levels(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_episode_progress_for_down: float = 0.25,
    reward_rate_keep_threshold: float = 0.05,
    reward_rate_up_threshold: float = 0.25,
    low_distance_ratio_for_down: float = 0.20,
    keep_distance_ratio: float = 0.25,
) -> torch.Tensor:
    """Reward-aware terrain curriculum for DreamWaQ.

    Isaac Lab's default velocity curriculum immediately moves environments down
    when they travel less than half of the commanded episode distance. DreamWaQ
    trains an estimator and policy together, so early hard-terrain attempts can
    still be useful. This curriculum keeps those attempts unless the episode was
    both low-reward and low-progress.
    """
    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return torch.mean(env.scene.terrain.terrain_levels.float())

    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.linalg.norm(
        asset.data.root_pos_w.torch[env_ids, :2]
        - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    terrain_length = terrain.cfg.terrain_generator.size[0]
    commanded_distance = (
        torch.linalg.norm(command[env_ids, :2], dim=1)
        * env.max_episode_length_s
    )

    elapsed_steps = env.episode_length_buf[env_ids].float().clamp_min(1.0)
    elapsed_time = elapsed_steps * env.step_dt
    episode_progress = elapsed_steps / float(env.max_episode_length)
    reward_rate = _episode_return(env, env_ids) / elapsed_time

    distance_success = distance > terrain_length / 2.0
    reward_success = (
        (reward_rate > reward_rate_up_threshold)
        & (distance > terrain_length * keep_distance_ratio)
    )
    move_up = distance_success | reward_success

    commanded_distance = commanded_distance.clamp_min(terrain_length * 0.25)
    very_low_distance = distance < commanded_distance * low_distance_ratio_for_down
    poor_return = reward_rate < reward_rate_keep_threshold
    enough_time_to_judge = episode_progress > min_episode_progress_for_down
    move_down = very_low_distance & poor_return & enough_time_to_judge
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
