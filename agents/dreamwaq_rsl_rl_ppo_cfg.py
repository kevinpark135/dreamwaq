from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import UnitreeA1RoughPPORunnerCfg


@configclass
class DreamWaQA1RoughPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # 로그 이름만 DreamWaQ로 분리
        self.experiment_name = "dreamwaq_a1_rough"

        # 처음 테스트용으로 작게
        self.max_iterations = 1000