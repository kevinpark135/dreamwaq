# dreamwaq_runner.py
"""DreamWaQ PPO runner with CENet integration."""

from __future__ import annotations

import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

from .dreamwaq_cenet import DreamWaQCENet, cenet_loss

class RunningMeanStd(torch.nn.Module):
    def __init__(self, shape, device, epsilon=1e-4, clip=10.0):
        super().__init__()
        self.clip = clip

        self.register_buffer(
            "mean",
            torch.zeros(shape, device=device),
        )
        self.register_buffer(
            "var",
            torch.ones(shape, device=device),
        )
        self.register_buffer(
            "count",
            torch.tensor(epsilon, device=device),
        )

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, *self.mean.shape)

        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            values.shape[0],
            device=values.device,
            dtype=self.count.dtype,
        )

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        old_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        correction = (
            delta.square()
            * self.count
            * batch_count
            / total_count
        )
        new_var = (old_m2 + batch_m2 + correction) / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(1e-6))
        self.count.copy_(total_count)

    def normalize(self, values):
        normalized = (values - self.mean) / torch.sqrt(
            self.var + 1e-8
        )
        return normalized.clamp(-self.clip, self.clip)

class DreamWaQRunner(OnPolicyRunner):
    """OnPolicyRunner extended with CENet update step."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # DreamWaQ-specific state lives on the wrapped Isaac Lab environment.
        self.base_env = env.unwrapped
        if not hasattr(self.base_env, "obs_history"):
            raise AttributeError(
                "DreamWaQRunner requires an environment exposing 'obs_history'."
            )

        single_obs_dim = self.base_env.observation_manager.group_obs_dim["policy"][0]
        history_length = getattr(self.base_env.cfg, "obs_history_length", 5)
        obs_history_dim = single_obs_dim * history_length

        self.cenet = DreamWaQCENet(
            obs_history_dim=obs_history_dim,
            single_obs_dim=single_obs_dim,
            latent_dim=16,
            hidden_dims=(128, 64),
        ).to(device)

        self.cenet_optimizer = torch.optim.Adam(
            self.cenet.parameters(), lr=1e-3
        )

        self.single_obs_dim = single_obs_dim
        self.obs_history_dim = obs_history_dim
        self.cenet_batch_size = 1024
        self.cenet_obs_normalizer = RunningMeanStd(
            shape=(self.single_obs_dim,),
            device=self.device,
        )

    def _normalize_history(self, obs_history):
        original_shape = obs_history.shape

        normalized = self.cenet_obs_normalizer.normalize(
            obs_history.reshape(-1, self.single_obs_dim)
        )

        return normalized.reshape(original_shape)

    def _add_cenet_features(self, obs):
        """Replace placeholder observations with CENet outputs."""
        obs_history = self.base_env.obs_history.to(self.device)
        obs_history = self._normalize_history(obs_history)
        cenet_out = self.cenet(obs_history)

        cenet_features = torch.cat(
            (cenet_out["v_hat"], cenet_out["z"]),
            dim=-1,
        )

        if obs["cenet"].shape != cenet_features.shape:
            raise ValueError(
                "CENet observation shape mismatch: "
                f"placeholder={tuple(obs['cenet'].shape)}, "
                f"features={tuple(cenet_features.shape)}"
            )

        obs["cenet"] = cenet_features
        return obs

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Learn with CENet update after each PPO update."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.cenet.train()

        with torch.inference_mode():
            obs = self._add_cenet_features(obs)

        if self.is_distributed:
            raise NotImplementedError(
                "Distributed training is not yet implemented for DreamWaQ CENet."
            )

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            start = time.time()

            with torch.inference_mode():
                obs = self._add_cenet_features(obs)

            obs_history_buf = []
            next_obs_buf = []
            base_lin_vel_buf = []
            valid_transition_buf = []

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    obs_history_buf.append(self.base_env.obs_history.clone())
                    base_lin_vel_buf.append(
                        self.base_env.scene["robot"].data.root_lin_vel_b.clone()
                    )

                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )

                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)

                    next_obs_buf.append(obs["policy"].clone())
                    valid_transition_buf.append(~dones.bool())
                    obs = self._add_cenet_features(obs)

                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"]["rnd_cfg"]
                        else None
                    )
                    self.logger.process_env_step(
                        rewards, dones, extras, intrinsic_rewards
                    )

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            obs_history_tensor = torch.stack(obs_history_buf).reshape(
                -1, self.obs_history_dim
            ).to(self.device)
            next_obs_tensor = torch.stack(next_obs_buf).reshape(
                -1, self.single_obs_dim
            ).to(self.device)
            base_lin_vel_tensor = torch.stack(base_lin_vel_buf).reshape(
                -1, 3
            ).to(self.device)
            valid_transition_tensor = torch.stack(valid_transition_buf).reshape(
                -1
            ).to(self.device)

            obs_history_tensor = obs_history_tensor[valid_transition_tensor]
            next_obs_tensor = next_obs_tensor[valid_transition_tensor]
            base_lin_vel_tensor = base_lin_vel_tensor[valid_transition_tensor]

            num_valid_samples = obs_history_tensor.shape[0]

            if num_valid_samples > 0:
                with torch.no_grad():
                    self.cenet_obs_normalizer.update(next_obs_tensor)

                obs_history_tensor = self._normalize_history(
                    obs_history_tensor
                )
                next_obs_tensor = self.cenet_obs_normalizer.normalize(
                    next_obs_tensor
                )

                loss_sums = {
                    "total": 0.0,
                    "velocity": 0.0,
                    "reconstruction": 0.0,
                    "kl": 0.0,
                }

                permutation = torch.randperm(
                    num_valid_samples,
                    device=self.device,
                )

                for start_idx in range(
                    0,
                    num_valid_samples,
                    self.cenet_batch_size,
                ):
                    batch_indices = permutation[
                        start_idx : start_idx + self.cenet_batch_size
                    ]

                    batch_history = obs_history_tensor[batch_indices]
                    batch_next_obs = next_obs_tensor[batch_indices]
                    batch_base_lin_vel = base_lin_vel_tensor[batch_indices]

                    cenet_out = self.cenet(batch_history)

                    ce_loss = cenet_loss(
                        cenet_out,
                        target_base_lin_vel=batch_base_lin_vel,
                        target_next_obs=batch_next_obs,
                        beta=1.0,
                    )

                    self.cenet_optimizer.zero_grad(set_to_none=True)
                    ce_loss["total_loss"].backward()
                    self.cenet_optimizer.step()

                    batch_size = batch_indices.shape[0]

                    loss_sums["total"] += (
                        ce_loss["total_loss"].item() * batch_size
                    )
                    loss_sums["velocity"] += (
                        ce_loss["velocity_loss"].item() * batch_size
                    )
                    loss_sums["reconstruction"] += (
                        ce_loss["reconstruction_loss"].item() * batch_size
                    )
                    loss_sums["kl"] += (
                        ce_loss["kl_loss"].item() * batch_size
                    )

                loss_dict["CENet/total"] = (
                    loss_sums["total"] / num_valid_samples
                )
                loss_dict["CENet/velocity"] = (
                    loss_sums["velocity"] / num_valid_samples
                )
                loss_dict["CENet/reconstruction"] = (
                    loss_sums["reconstruction"] / num_valid_samples
                )
                loss_dict["CENet/kl"] = (
                    loss_sums["kl"] / num_valid_samples
                )

            else:
                loss_dict["CENet/total"] = 0.0
                loss_dict["CENet/velocity"] = 0.0
                loss_dict["CENet/reconstruction"] = 0.0
                loss_dict["CENet/kl"] = 0.0

            loss_dict["CENet/valid_samples"] = num_valid_samples

            learn_time = time.time() - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=(
                    self.alg.rnd.weight
                    if self.cfg["algorithm"]["rnd_cfg"]
                    else None
                ),
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )
            self.logger.stop_logging_writer()

    def save(self, path, infos=None):
        """Save PPO + CENet weights."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["cenet"] = self.cenet.state_dict()
        saved_dict["cenet_optimizer"] = self.cenet_optimizer.state_dict()
        saved_dict["cenet_obs_normalizer"] = (
            self.cenet_obs_normalizer.state_dict()
        )
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        """Load PPO + CENet weights."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if "cenet" in loaded_dict:
            self.cenet.load_state_dict(loaded_dict["cenet"])
        if "cenet_optimizer" in loaded_dict:
            self.cenet_optimizer.load_state_dict(loaded_dict["cenet_optimizer"])
        if "cenet_obs_normalizer" in loaded_dict:
            self.cenet_obs_normalizer.load_state_dict(
                loaded_dict["cenet_obs_normalizer"]
            )

        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
