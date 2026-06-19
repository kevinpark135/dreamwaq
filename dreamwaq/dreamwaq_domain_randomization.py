"""Domain randomization helpers for the DreamWaQ A1 task."""

from __future__ import annotations

import torch


def randomize_payload_mass(
    env,
    env_ids: torch.Tensor | None,
    mass_distribution_params: tuple[float, float],
    asset_cfg,
    min_mass: float = 1.0,
):
    """Add a bounded payload mass offset to selected bodies.

    Isaac Lab's generic mass randomizer is scale-oriented for the stock A1 task.
    DreamWaQ uses an absolute payload range, so sanitize the sampled masses before
    sending them to PhysX to avoid invalid inertial values during startup.
    """

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)

    if isinstance(asset_cfg.body_ids, slice):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device=asset.device)
    else:
        body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device=asset.device)

    default_key = "_dreamwaq_default_body_mass"
    if not hasattr(asset, default_key):
        setattr(asset, default_key, asset.data.body_mass.torch.clone())
    default_mass = getattr(asset, default_key)

    low, high = mass_distribution_params
    payload = torch.empty((env_ids.numel(), body_ids.numel()), device=asset.device).uniform_(low, high)
    masses = default_mass[env_ids[:, None], body_ids].clone() + payload
    masses = torch.nan_to_num(masses, nan=min_mass, posinf=min_mass, neginf=min_mass)
    masses = masses.clamp_min(min_mass).contiguous()

    asset.set_masses_index(
        masses=masses,
        body_ids=body_ids.to(dtype=torch.int32),
        env_ids=env_ids.to(dtype=torch.int32),
    )


def randomize_motor_strength(
    env,
    env_ids: torch.Tensor | None,
    strength_distribution_params: tuple[float, float],
    asset_cfg,
):
    """Scale actuator torque limits to emulate motor strength variation."""

    from isaaclab.actuators import ImplicitActuator

    asset = env.scene[asset_cfg.name]
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
