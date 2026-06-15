from isaaclab.utils.configclass import configclass
from .rsl_rl_ppo_cfg import UnitreeA1RoughPPORunnerCfg


@configclass
class DreamWaQA1RoughPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "dreamwaq_a1_rough"
        self.max_iterations = 1000

        # asymmetric actor-critic: critic은 privileged obs 받음
        self.policy.class_name = "ActorCritic"
        self.algorithm.class_name = "PPO"