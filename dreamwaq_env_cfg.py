from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeA1RoughEnvCfg, UnitreeA1RoughEnvCfg_PLAY


@configclass
class DreamWaQA1RoughEnvCfg(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 여기부터 DreamWaQ 전용 변경사항을 하나씩 넣을 자리
        # 처음에는 아무것도 바꾸지 말고, 태스크 등록/학습 실행 확인부터 한다.
        pass


@configclass
class DreamWaQA1RoughEnvCfg_PLAY(UnitreeA1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # play 전용 설정도 나중에 여기서 조정
        pass