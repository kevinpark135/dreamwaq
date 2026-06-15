"""DreamWaQ environment configuration for four-legged locomotion with proprioceptive history and privileged critic observations."""

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from ..rough_env_cfg import UnitreeA1RoughEnvCfg, UnitreeA1RoughEnvCfg_PLAY


def empty_cenet_features(env) -> torch.Tensor:
    return torch.zeros((env.num_envs, 19), device=env.device)


def joint_power(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize total absolute mechanical power across all joints."""
    robot = env.scene[asset_cfg.name]
    torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque) * torch.abs(joint_vel), dim=1)


def foot_clearance(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize moving feet that deviate from the desired terrain-relative height."""
    robot = env.scene[asset_cfg.name]
    height_scanner = env.scene[sensor_cfg.name]

    terrain_height = torch.mean(
        height_scanner.data.ray_hits_w[..., 2],
        dim=1,
    )
    foot_height = (
        robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
        - terrain_height.unsqueeze(-1)
    )
    foot_lateral_speed = torch.linalg.vector_norm(
        robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2],
        dim=-1,
    )

    height_error = torch.square(target_height - foot_height)
    return torch.sum(height_error * foot_lateral_speed, dim=1)


def power_distribution(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared variance of mechanical power across joints."""
    robot = env.scene[asset_cfg.name]
    torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    mechanical_power = torque * joint_vel
    power_variance = torch.var(
        mechanical_power,
        dim=1,
        unbiased=False,
    )
    return torch.square(power_variance)


class ActionSmoothnessPenalty(ManagerTermBase):
    """Penalize the second finite difference of consecutive actions."""

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
        action = env.action_manager.action
        previous_action = env.action_manager.prev_action

        second_difference = (
            action
            - 2.0 * previous_action
            + self.previous_previous_action
        )
        penalty = torch.sum(torch.square(second_difference), dim=1)

        self.previous_previous_action.copy_(previous_action)
        return penalty


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

        # DreamWaQ domain randomization ranges from the paper.
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.physics_material.params["num_buckets"] = 64
        self.events.physics_material.params["make_consistent"] = True

        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.events.add_base_mass.params["operation"] = "add"

        self.events.base_com.default.params["com_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05),
        }

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

        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01

        # DreamWaQ reward terms from the paper.
        self.rewards.dof_torques_l2 = None
        self.rewards.feet_air_time = None

        self.rewards.joint_power = RewTerm(
            func=joint_power,
            weight=-2.0e-5,
        )
        self.rewards.body_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-1.0,
            params={
                "target_height": 0.42,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.foot_clearance = RewTerm(
            func=foot_clearance,
            weight=-0.01,
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
            weight=-0.01,
        )
        self.rewards.power_distribution = None

        """ self.rewards.power_distribution = RewTerm(
            func=power_distribution,
            weight=-1.0e-5,
        ) """

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

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
