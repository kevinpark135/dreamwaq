from isaaclab.utils.configclass import configclass
from .rsl_rl_ppo_cfg import UnitreeA1RoughPPORunnerCfg


@configclass
class DreamWaQA1RoughPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "dreamwaq_a1_rough"
        self.max_iterations = 10000

        self.actor.distribution_cfg.init_std = 0.5
        self.actor.distribution_cfg.std_type = "log"

        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.0
        self.algorithm.max_grad_norm = 0.5
