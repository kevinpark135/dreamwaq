"""DreamWaQ reward terms.

This module contains reward terms that are specific to the DreamWaQ paper's
Table I reward design.  The environment config should only decide which terms
are enabled and what weights they use; the actual reward math lives here.

The functions intentionally sanitize non-finite simulator values before they
reach PPO.  Height scanner misses, unstable contacts, or extreme joint states
can otherwise create NaN/Inf rewards and break training.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase


# Conservative clamps used only inside reward calculations.  They preserve the
# paper's reward structure while preventing one bad physics value from producing
# NaN/Inf or an enormous value target.
_MAX_ABS_TORQUE = 120.0
_MAX_ABS_JOINT_VEL = 80.0
_MAX_FOOT_SPEED = 5.0
_MAX_ABS_BASE_LIN_VEL = 20.0
_MAX_ABS_BASE_ANG_VEL = 20.0
_MAX_ABS_JOINT_ACC = 500.0
_MAX_ACTION_RATE_REWARD = 100.0
_MAX_HEIGHT_ERROR_SQ = 1.0
_MAX_JOINT_POWER_REWARD = 5_000.0
_MAX_FOOT_CLEARANCE_REWARD = 20.0
_MAX_POWER_DISTRIBUTION_REWARD = 10_000.0


def _finite(tensor: torch.Tensor, clamp_abs: float | None = None) -> torch.Tensor:
    """Replace invalid tensor values and optionally clamp magnitude."""
    tensor = torch.nan_to_num(
        tensor,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if clamp_abs is not None:
        tensor = tensor.clamp(-clamp_abs, clamp_abs)
    return tensor


def _mean_terrain_height_from_scanner(
    env,
    sensor_cfg: SceneEntityCfg,
    fallback_height: torch.Tensor,
) -> torch.Tensor:
    """Estimate terrain height while ignoring invalid ray-caster hits."""
    height_scanner = env.scene[sensor_cfg.name]
    ray_heights = height_scanner.data.ray_hits_w[..., 2]
    finite_mask = torch.isfinite(ray_heights)

    safe_ray_heights = torch.where(
        finite_mask,
        ray_heights,
        torch.zeros_like(ray_heights),
    )
    valid_count = finite_mask.sum(dim=1).clamp_min(1)
    terrain_height = safe_ray_heights.sum(dim=1) / valid_count

    terrain_height = torch.where(
        finite_mask.any(dim=1),
        terrain_height,
        fallback_height,
    )
    return _finite(terrain_height)


def _command(env, command_name: str) -> torch.Tensor:
    """Read a finite command tensor from Isaac Lab's command manager."""
    return _finite(env.command_manager.get_command(command_name))


def track_lin_vel_xy_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DreamWaQ Table I: exp(-4 * ||v_xy_cmd - v_xy||^2).

    Isaac Lab exposes the coefficient as exp(-error / std^2).  Passing
    std=0.5 gives the paper's exp(-4 * error).
    """
    robot = env.scene[asset_cfg.name]
    command = _command(env, command_name)[:, :2]
    lin_vel_xy = _finite(
        robot.data.root_lin_vel_b[:, :2],
        clamp_abs=_MAX_ABS_BASE_LIN_VEL,
    )
    error = torch.sum(torch.square(command - lin_vel_xy), dim=1)
    std_sq = max(float(std) ** 2, 1.0e-6)
    return torch.exp(-error / std_sq)


def track_ang_vel_z_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DreamWaQ Table I: exp(-4 * (omega_yaw_cmd - omega_yaw)^2)."""
    robot = env.scene[asset_cfg.name]
    command = _command(env, command_name)[:, 2]
    yaw_rate = _finite(
        robot.data.root_ang_vel_b[:, 2],
        clamp_abs=_MAX_ABS_BASE_ANG_VEL,
    )
    error = torch.square(command - yaw_rate)
    std_sq = max(float(std) ** 2, 1.0e-6)
    return torch.exp(-error / std_sq)


def lin_vel_z_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DreamWaQ Table I: vertical body velocity squared."""
    robot = env.scene[asset_cfg.name]
    lin_vel_z = _finite(
        robot.data.root_lin_vel_b[:, 2],
        clamp_abs=_MAX_ABS_BASE_LIN_VEL,
    )
    return torch.square(lin_vel_z)


def ang_vel_xy_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DreamWaQ Table I: roll/pitch angular velocity squared."""
    robot = env.scene[asset_cfg.name]
    ang_vel_xy = _finite(
        robot.data.root_ang_vel_b[:, :2],
        clamp_abs=_MAX_ABS_BASE_ANG_VEL,
    )
    return torch.sum(torch.square(ang_vel_xy), dim=1)


def orientation_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DreamWaQ Table I: projected gravity xy norm squared."""
    robot = env.scene[asset_cfg.name]
    projected_gravity = _finite(robot.data.projected_gravity_b)
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


def joint_acc_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DreamWaQ Table I: joint acceleration squared."""
    robot = env.scene[asset_cfg.name]
    joint_acc = _finite(
        robot.data.joint_acc[:, asset_cfg.joint_ids],
        clamp_abs=_MAX_ABS_JOINT_ACC,
    )
    return torch.sum(torch.square(joint_acc), dim=1)


def action_rate_l2(env) -> torch.Tensor:
    """DreamWaQ Table I: (a_t - a_{t-1})^2."""
    action = _finite(env.action_manager.action)
    previous_action = _finite(env.action_manager.prev_action)
    reward = torch.sum(torch.square(action - previous_action), dim=1)
    return reward.clamp(max=_MAX_ACTION_RATE_REWARD)


def joint_power(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DreamWaQ Table I: |tau| |joint velocity|."""
    robot = env.scene[asset_cfg.name]
    torque = _finite(
        robot.data.applied_torque[:, asset_cfg.joint_ids],
        clamp_abs=_MAX_ABS_TORQUE,
    )
    joint_vel = _finite(
        robot.data.joint_vel[:, asset_cfg.joint_ids],
        clamp_abs=_MAX_ABS_JOINT_VEL,
    )

    reward = torch.sum(torch.abs(torque) * torch.abs(joint_vel), dim=1)
    return reward.clamp(max=_MAX_JOINT_POWER_REWARD)


def body_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """DreamWaQ Table I: (desired body height - body height)^2."""
    robot = env.scene[asset_cfg.name]
    base_z = _finite(robot.data.root_pos_w[:, 2])
    fallback_terrain_z = base_z - target_height

    terrain_z = _mean_terrain_height_from_scanner(
        env,
        sensor_cfg=sensor_cfg,
        fallback_height=fallback_terrain_z,
    )
    body_height = _finite(base_z - terrain_z)

    height_error_sq = torch.square(target_height - body_height)
    return height_error_sq.clamp(max=_MAX_HEIGHT_ERROR_SQ)


def foot_clearance(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """DreamWaQ Table I: (p_des_f,z - p_f,z)^2 * v_f,xy."""
    robot = env.scene[asset_cfg.name]
    base_z = _finite(robot.data.root_pos_w[:, 2])
    terrain_height = _mean_terrain_height_from_scanner(
        env,
        sensor_cfg=sensor_cfg,
        fallback_height=base_z - 0.42,
    )

    foot_height = (
        _finite(robot.data.body_pos_w[:, asset_cfg.body_ids, 2])
        - terrain_height.unsqueeze(-1)
    )
    foot_lateral_speed = torch.linalg.vector_norm(
        _finite(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]),
        dim=-1,
    ).clamp(max=_MAX_FOOT_SPEED)

    height_error_sq = torch.square(target_height - foot_height).clamp(
        max=_MAX_HEIGHT_ERROR_SQ
    )
    reward = torch.sum(height_error_sq * foot_lateral_speed, dim=1)
    return reward.clamp(max=_MAX_FOOT_CLEARANCE_REWARD)


def power_distribution(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DreamWaQ Table I: var(tau * joint velocity)^2."""
    robot = env.scene[asset_cfg.name]
    torque = _finite(
        robot.data.applied_torque[:, asset_cfg.joint_ids],
        clamp_abs=_MAX_ABS_TORQUE,
    )
    joint_vel = _finite(
        robot.data.joint_vel[:, asset_cfg.joint_ids],
        clamp_abs=_MAX_ABS_JOINT_VEL,
    )

    mechanical_power = torque * joint_vel
    power_variance = torch.var(
        mechanical_power,
        dim=1,
        unbiased=False,
    )
    return torch.square(power_variance).clamp(
        max=_MAX_POWER_DISTRIBUTION_REWARD
    )


class ActionSmoothnessPenalty(ManagerTermBase):
    """DreamWaQ Table I: (a_t - 2 a_{t-1} + a_{t-2})^2."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_previous_action = torch.zeros(
            env.num_envs,
            env.action_manager.total_action_dim,
            device=env.device,
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            self.previous_previous_action.zero_()
        else:
            self.previous_previous_action[env_ids] = 0.0

    def __call__(self, env) -> torch.Tensor:
        action = _finite(env.action_manager.action)
        previous_action = _finite(env.action_manager.prev_action)

        second_difference = (
            action
            - 2.0 * previous_action
            + self.previous_previous_action
        )
        penalty = torch.sum(torch.square(second_difference), dim=1)

        self.previous_previous_action.copy_(previous_action)
        return _finite(penalty, clamp_abs=100.0)


# Compatibility aliases for environment configs that used Isaac Lab naming.
flat_orientation_l2 = orientation_l2
dof_acc_l2 = joint_acc_l2


__all__ = [
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "orientation_l2",
    "flat_orientation_l2",
    "joint_acc_l2",
    "dof_acc_l2",
    "joint_power",
    "body_height_l2",
    "foot_clearance",
    "action_rate_l2",
    "ActionSmoothnessPenalty",
    "power_distribution",
]
