"""DreamWaQ custom environment with proprioceptive observation history buffer."""

from __future__ import annotations
from random import seed

from requests import options

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.buffers import CircularBuffer


class DreamWaQEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv extended with a proprioceptive history buffer for DreamWaQ.

    Every step, the current actor observation (pure proprio, no height scan)
    is pushed into a CircularBuffer of length `history_length`.
    The flattened history is exposed as `self.obs_history` for CENet input.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # history 설정 — cfg에서 읽거나 기본값 5
        self.history_length = getattr(cfg, "obs_history_length", 5)

        # actor obs 한 스텝의 dim 확인
        # ObservationManager가 이미 초기화되어 있으므로 바로 읽을 수 있음
        single_obs_dim = self.observation_manager.group_obs_dim["policy"][0]

        # CircularBuffer(history_length, batch_size, device)
        # shape: [history_length, num_envs, single_obs_dim]
        self.obs_history_buf = CircularBuffer(
            max_len=self.history_length,
            batch_size=self.num_envs,
            device=self.device,
        )

        # 첫 obs로 버퍼 초기화 (zeros)
        init_obs = torch.zeros(self.num_envs, single_obs_dim, device=self.device)
        for _ in range(self.history_length):
            self.obs_history_buf.append(init_obs)

    def step(self, action: torch.Tensor):
        # 기본 step 실행
        obs_dict, reward, terminated, truncated, info = super().step(action)

        # 현재 actor obs를 버퍼에 push
        current_obs = obs_dict["policy"]  # [num_envs, single_obs_dim]
        self.obs_history_buf.append(current_obs)

        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs_dict, info = super().reset(seed=seed, options=options)

        # episode_length_buf == 0인 env들만 reset된 것
        reset_env_ids = (self.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            single_obs_dim = self.observation_manager.group_obs_dim["policy"][0]
            zeros = torch.zeros(len(reset_env_ids), single_obs_dim, device=self.device)

            # reset된 env들만 해당 슬롯을 zeros로 덮어씀
            for i in range(self.history_length):
                self.obs_history_buf.buffer[reset_env_ids, i, :] = zeros

        return obs_dict, info

    @property
    def obs_history(self) -> torch.Tensor:
        """Flattened obs history for CENet input.

        Returns:
            Tensor of shape [num_envs, history_length * single_obs_dim]
        """
        # CircularBuffer.buffer shape: [max_len, num_envs, obs_dim]
        # 시간순으로 정렬해서 flatten
        return self.obs_history_buf.buffer.reshape(self.num_envs, -1)