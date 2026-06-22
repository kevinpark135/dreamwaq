"""DreamWaQ environment configuration for four-legged locomotion with proprioceptive history and privileged critic observations."""

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from ..rough_env_cfg import UnitreeA1RoughEnvCfg, UnitreeA1RoughEnvCfg_PLAY
from .dreamwaq_curriculums import dreamwaq_terrain_levels
from .dreamwaq_domain_randomization import randomize_motor_strength, randomize_payload_mass
from .dreamwaq_rewards import (
    ActionSmoothnessPenalty,
    action_rate_l2,
    ang_vel_xy_l2,
    body_height_l2,
    dof_acc_l2,
    flat_orientation_l2,
    foot_clearance,
    joint_power,
    lin_vel_z_l2,
    power_distribution,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
)


def configure_physx_gpu_capacity(physics_cfg) -> None:
    """Set moderate PhysX GPU buffers for contact-heavy rough-terrain training."""
    physx_cfgs = [physics_cfg]
    for preset_name in ("default", "physx"):
        if hasattr(physics_cfg, preset_name):
            physx_cfgs.append(getattr(physics_cfg, preset_name))

    for cfg in physx_cfgs:
        if not hasattr(cfg, "gpu_found_lost_aggregate_pairs_capacity"):
            continue

        cfg.gpu_found_lost_pairs_capacity = max(
            cfg.gpu_found_lost_pairs_capacity,
            2**24,
        )
        cfg.gpu_found_lost_aggregate_pairs_capacity = max(
            cfg.gpu_found_lost_aggregate_pairs_capacity,
            2**26,
        )
        cfg.gpu_total_aggregate_pairs_capacity = max(
            cfg.gpu_total_aggregate_pairs_capacity,
            2**22,
        )


def empty_cenet_features(env) -> torch.Tensor:
    return torch.zeros((env.num_envs, 19), device=env.device)


def configure_forward_heading_commands(command_cfg, fixed_heading: bool = False) -> None:
    """Use forward body motion and heading control instead of backward/lateral velocity commands."""

    command_cfg.heading_command = True
    command_cfg.rel_heading_envs = 1.0
    command_cfg.heading_control_stiffness = 1.0
    command_cfg.ranges.lin_vel_x = (0.2, 1.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (-1.2, 1.2)
    if fixed_heading:
        command_cfg.ranges.lin_vel_x = (1.0, 1.0)
        command_cfg.ranges.heading = (0.0, 0.0)


@configclass
class DreamWaQCENetObsCfg(ObsGroup):
    """Placeholder for CENet velocity and latent context."""

    features = ObsTerm(func=empty_cenet_features)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True

@configclass
class DreamWaQCriticObsCfg(ObsGroup):
    """Privileged observations for DreamWaQ critic."""

    # basic proprioception
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)

    # privileged info
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True

@configclass
class DreamWaQA1RoughEnvCfg(UnitreeA1RoughEnvCfg):
    # actor: temporal observation history
    obs_history_length: int = 5
    
    def __post_init__(self):
        super().__post_init__()

        # Keep Isaac Lab's default initial terrain range while using DreamWaQ's curriculum rule.
        self.scene.terrain.max_init_terrain_level = 5

        self.curriculum.terrain_levels = CurrTerm(
            func=dreamwaq_terrain_levels,
        )

        configure_forward_heading_commands(self.commands.base_velocity)

        configure_physx_gpu_capacity(self.sim.physics)

        # DreamWaQ domain randomization ranges from the paper.
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.physics_material.params["num_buckets"] = 64
        self.events.physics_material.params["make_consistent"] = True

        self.events.add_base_mass = EventTerm(
            func=randomize_payload_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "mass_distribution_params": (-1.0, 2.0),
                "min_mass": 1.0,
            },
        )

        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "com_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05),
                },
            },
        )

        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.motor_strength = EventTerm(
            func=randomize_motor_strength,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "strength_distribution_params": (0.9, 1.1),
            },
        )

        # DreamWaQ Table I rewards.
        # Do not only adjust inherited Isaac Lab reward weights here.
        # Rebind every paper reward to the implementation in dreamwaq_rewards.py.
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=track_lin_vel_xy_exp,
            weight=1.35,
            params={
                "command_name": "base_velocity",
                "std": 0.55,
            },
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=track_ang_vel_z_exp,
            weight=0.65,
            params={
                "command_name": "base_velocity",
                "std": 0.55,
            },
        )
        self.rewards.lin_vel_z_l2 = RewTerm(
            func=lin_vel_z_l2,
            weight=-2.0,
        )
        self.rewards.ang_vel_xy_l2 = RewTerm(
            func=ang_vel_xy_l2,
            weight=-0.035,
        )
        self.rewards.flat_orientation_l2 = RewTerm(
            func=flat_orientation_l2,
            weight=-0.2,
        )
        self.rewards.dof_acc_l2 = RewTerm(
            func=dof_acc_l2,
            weight=-2.5e-7,
        )
        self.rewards.action_rate_l2 = RewTerm(
            func=action_rate_l2,
            weight=-0.006,
        )
        self.rewards.dof_torques_l2 = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.dof_pos_limits = None

        self.rewards.joint_power = RewTerm(
            func=joint_power,
            weight=-1.2e-5,
        )
        self.rewards.body_height = RewTerm(
            func=body_height_l2,
            weight=-0.55,
            params={
                "target_height": 0.42,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.foot_clearance = RewTerm(
            func=foot_clearance,
            weight=-0.006,
            params={
                "target_height": 0.08,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=".*_foot",
                ),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.action_smoothness = RewTerm(
            func=ActionSmoothnessPenalty,
            weight=-0.004,
        )
        self.rewards.power_distribution = RewTerm(
            func=power_distribution,
            weight=-5.0e-6,
        )

        # actor: remove exteroceptive observations
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None

        # critic: privileged observation
        self.observations.critic = DreamWaQCriticObsCfg()
        self.observations.cenet = DreamWaQCENetObsCfg()


@configclass
class DreamWaQA1RoughEnvCfg_PLAY(UnitreeA1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None
        self.observations.cenet = DreamWaQCENetObsCfg()
        self.observations.critic = DreamWaQCriticObsCfg()

        configure_forward_heading_commands(self.commands.base_velocity, fixed_heading=True)

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.motor_strength = None
