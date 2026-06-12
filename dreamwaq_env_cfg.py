from isaaclab.utils.configclass import configclass

from .rough_env_cfg import UnitreeA1RoughEnvCfg, UnitreeA1RoughEnvCfg_PLAY


@configclass
class DreamWaQA1RoughEnvCfg(UnitreeA1RoughEnvCfg):
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

        # remove exteroceptive observations
        self.observations.policy.height_scan = None


@configclass
class DreamWaQA1RoughEnvCfg_PLAY(UnitreeA1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan = None

        self.events.base_external_force_torque = None
        self.events.push_robot = None