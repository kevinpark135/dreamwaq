"""DreamWaQ environment configuration for four-legged locomotion with proprioceptive history and privileged critic observations."""

from isaaclab.utils.configclass import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeA1RoughEnvCfg, UnitreeA1RoughEnvCfg_PLAY

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

        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.action_rate_l2.weight = -0.01

        # actor: remove exteroceptive observations
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None

        # critic: privileged observation
        self.observations.critic = DreamWaQCriticObsCfg()


@configclass
class DreamWaQA1RoughEnvCfg_PLAY(UnitreeA1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None

        self.events.base_external_force_torque = None
        self.events.push_robot = None