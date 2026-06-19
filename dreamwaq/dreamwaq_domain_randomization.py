"""Domain randomization helpers for the DreamWaQ A1 task."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer


class DelayedJointPositionAction(JointPositionAction):
    """Joint position action with per-environment command delay randomization."""

    max_system_delay_s = 0.015

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._max_delay_steps = max(0, math.ceil(self.max_system_delay_s / env.step_dt))
        self._delay_buffer = DelayBuffer(self._max_delay_steps, self.num_envs, self.device)
        self._sample_delay_lag()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions = self._delay_buffer.compute(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            super().reset(slice(None))
            self._delay_buffer.reset()
            self._sample_delay_lag()
            return

        super().reset(env_ids)
        self._delay_buffer.reset(env_ids)
        self._sample_delay_lag(env_ids)

    def _sample_delay_lag(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if self._max_delay_steps == 0:
            self._delay_buffer.set_time_lag(0, batch_ids=env_ids)
            return

        if env_ids is None:
            num_envs = self.num_envs
        elif isinstance(env_ids, torch.Tensor):
            num_envs = env_ids.numel()
        else:
            num_envs = len(env_ids)

        time_lag = torch.randint(
            0,
            self._max_delay_steps + 1,
            (num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self._delay_buffer.set_time_lag(time_lag, batch_ids=env_ids)


def randomize_motor_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    strength_distribution_params: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Scale actuator torque limits to emulate motor strength variation."""

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(asset.device)

    if isinstance(asset_cfg.joint_ids, slice):
        requested_joint_ids = torch.arange(asset.num_joints, dtype=torch.long, device=asset.device)
    else:
        requested_joint_ids = torch.as_tensor(asset_cfg.joint_ids, dtype=torch.long, device=asset.device)

    default_key = "_dreamwaq_default_motor_strength"
    if not hasattr(asset, default_key):
        defaults = {}
        for name, actuator in asset.actuators.items():
            defaults[name] = {"effort_limit": actuator.effort_limit.clone()}
            if hasattr(actuator, "_saturation_effort"):
                saturation = actuator._saturation_effort
                if not isinstance(saturation, torch.Tensor):
                    saturation = torch.full_like(actuator.effort_limit, float(saturation))
                defaults[name]["saturation_effort"] = saturation.clone()
        setattr(asset, default_key, defaults)

    defaults = getattr(asset, default_key)
    low, high = strength_distribution_params

    for name, actuator in asset.actuators.items():
        actuator_joint_ids = actuator.joint_indices
        if isinstance(actuator_joint_ids, slice):
            global_joint_ids = torch.arange(asset.num_joints, dtype=torch.long, device=asset.device)
            actuator_indices = requested_joint_ids
        else:
            global_joint_ids = actuator_joint_ids.to(asset.device)
            mask = torch.isin(global_joint_ids, requested_joint_ids)
            if not torch.any(mask):
                continue
            actuator_indices = torch.nonzero(mask, as_tuple=False).flatten()

        strength = torch.empty((env_ids.numel(), actuator_indices.numel()), device=asset.device).uniform_(low, high)
        effort_limit = actuator.effort_limit[env_ids].clone()
        effort_limit[:, actuator_indices] = defaults[name]["effort_limit"][env_ids][:, actuator_indices] * strength
        actuator.effort_limit[env_ids] = effort_limit

        if "saturation_effort" in defaults[name]:
            saturation_effort = getattr(actuator, "_saturation_effort")
            if not isinstance(saturation_effort, torch.Tensor):
                saturation_effort = defaults[name]["saturation_effort"].clone()
            else:
                saturation_effort = saturation_effort.clone()
            saturation_effort[env_ids[:, None], actuator_indices] = (
                defaults[name]["saturation_effort"][env_ids[:, None], actuator_indices] * strength
            )
            actuator._saturation_effort = saturation_effort
            actuator._vel_at_effort_lim = actuator.velocity_limit * (1 + actuator.effort_limit / actuator._saturation_effort)

        if isinstance(actuator, ImplicitActuator):
            sim_limits = actuator.effort_limit[env_ids][:, actuator_indices]
            sim_joint_ids = global_joint_ids[actuator_indices]
            asset.write_joint_effort_limit_to_sim_index(limits=sim_limits, joint_ids=sim_joint_ids, env_ids=env_ids)
