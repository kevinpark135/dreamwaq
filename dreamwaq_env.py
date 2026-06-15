"""DreamWaQ custom environment with proprioceptive observation history buffer."""

from __future__ import annotations

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

        # Read the history length from the environment configuration.
        self.history_length = getattr(cfg, "obs_history_length", 5)

        # The observation manager is already initialized at this point.
        single_obs_dim = self.observation_manager.group_obs_dim["policy"][0]

        self.obs_history_buf = CircularBuffer(
            max_len=self.history_length,
            batch_size=self.num_envs,
            device=self.device,
        )

        # Initialize storage before the first external reset.
        init_obs = torch.zeros(self.num_envs, single_obs_dim, device=self.device)
        self.obs_history_buf.append(init_obs)

    def step(self, action: torch.Tensor):
        obs_dict, reward, terminated, truncated, info = super().step(action)

        # super().step() has already reset terminated environments and returns
        # their first observation from the new episode.
        done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            self.obs_history_buf.reset(done_env_ids)

        self.obs_history_buf.append(obs_dict["policy"])
        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs_dict, info = super().reset(seed=seed, options=options)
        self.obs_history_buf.reset()
        self.obs_history_buf.append(obs_dict["policy"])
        return obs_dict, info

    @property
    def obs_history(self) -> torch.Tensor:
        """Flattened obs history for CENet input.

        Returns:
            Tensor of shape [num_envs, history_length * single_obs_dim]
        """
        # CircularBuffer.buffer is ordered from oldest to newest.
        return self.obs_history_buf.buffer.reshape(self.num_envs, -1)
