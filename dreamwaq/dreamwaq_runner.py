# dreamwaq_runner.py
"""DreamWaQ PPO runner with CENet integration."""

from __future__ import annotations

import os
import math
import time

import torch
from isaaclab.utils.buffers import DelayBuffer
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
            hidden_dims=(256, 128),
        ).to(device)

        self.cenet_optimizer = torch.optim.AdamW(
            self.cenet.parameters(), lr=2e-3, weight_decay=1.0e-5
        )

        self.single_obs_dim = single_obs_dim
        self.obs_history_dim = obs_history_dim
        self.cenet_batch_size = 1024
        self.cenet_num_epochs = 2
        self.cenet_obs_normalizer = RunningMeanStd(
            shape=(self.single_obs_dim,),
            device=self.device,
        )
        self.max_system_delay_s = 0.015
        self.max_action_delay_steps = max(0, math.ceil(self.max_system_delay_s / self.base_env.step_dt))
        self.action_delay_buffer = None
        self.adaboot_probability = 0.35
        self.adaboot_reward_cv = 0.0
        self.adaboot_target_probability = 0.65
        self.adaboot_ema_alpha = 0.25
        self.adaboot_warmup_iterations = 400
        self.adaboot_ramp_iterations = 200
        self.adaboot_episode_returns = torch.zeros(
            self.base_env.num_envs,
            device=self.device,
        )
        self.adaboot_episode_valid = torch.ones(
            self.base_env.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        distribution_cfg = train_cfg.get("actor", {}).get("distribution_cfg", {})
        self._policy_std_is_log = distribution_cfg.get("std_type") == "log"
        self._policy_std_min = 1.0e-3
        self._policy_std_max = 2.0
        self._patch_actor_std_safety()

    @torch.no_grad()
    def _sanitize_optimizer_state(self):
        """Remove invalid values from PPO optimizer state."""
        optimizer = getattr(self.alg, "optimizer", None)
        if optimizer is None:
            return

        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _sanitize_policy_std(self):
        """Keep Gaussian std parameters valid before actor sampling."""
        actors = [
            getattr(self.alg, "actor", None),
            getattr(self.alg, "actor_critic", None),
            self.alg.get_policy() if hasattr(self.alg, "get_policy") else None,
        ]

        seen = set()
        for actor in actors:
            if actor is None or id(actor) in seen:
                continue
            seen.add(id(actor))

            for name, param in actor.named_parameters():
                if "std" not in name.lower():
                    continue

                param.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if self._policy_std_is_log:
                    param.data.clamp_(
                        math.log(self._policy_std_min),
                        math.log(self._policy_std_max),
                    )
                else:
                    param.data.clamp_(
                        self._policy_std_min,
                        self._policy_std_max,
                    )

    @torch.no_grad()
    def _sanitize_observations(self, obs):
        """Remove invalid values from observation tensors before PPO sees them."""
        if hasattr(obs, "keys"):
            for key in obs.keys():
                value = obs[key]
                if torch.is_tensor(value):
                    obs[key] = torch.nan_to_num(
                        value,
                        nan=0.0,
                        posinf=1.0e6,
                        neginf=-1.0e6,
                    )
        elif torch.is_tensor(obs):
            obs = torch.nan_to_num(
                obs,
                nan=0.0,
                posinf=1.0e6,
                neginf=-1.0e6,
            )
        return obs

    def _patch_actor_std_safety(self):
        """Clamp actor std before every actor forward call."""
        actor = getattr(self.alg, "actor", None)
        if actor is None:
            return
        if getattr(actor, "_dreamwaq_std_safety_patched", False):
            return

        original_forward = actor.forward

        def safe_forward(*args, **kwargs):
            self._sanitize_policy_std()
            self._sanitize_optimizer_state()
            return original_forward(*args, **kwargs)

        actor.forward = safe_forward
        actor._dreamwaq_std_safety_patched = True

    def _normalize_history(self, obs_history):
        original_shape = obs_history.shape

        normalized = self.cenet_obs_normalizer.normalize(
            obs_history.reshape(-1, self.single_obs_dim)
        )

        return normalized.reshape(original_shape)

    def _sample_action_delay_lag(self, env_ids=None):
        if self.action_delay_buffer is None:
            return
        if self.max_action_delay_steps == 0:
            self.action_delay_buffer.set_time_lag(0, batch_ids=env_ids)
            return

        if env_ids is None:
            num_envs = self.base_env.num_envs
        else:
            num_envs = int(env_ids.numel())
        lags = torch.randint(
            0,
            self.max_action_delay_steps + 1,
            (num_envs,),
            dtype=torch.int,
            device=self.device,
        )
        self.action_delay_buffer.set_time_lag(lags, batch_ids=env_ids)

    def _apply_action_delay(self, actions):
        if self.action_delay_buffer is None:
            self.action_delay_buffer = DelayBuffer(
                self.max_action_delay_steps,
                actions.shape[0],
                self.device,
            )
            self._sample_action_delay_lag()
        delayed_actions = self.action_delay_buffer.compute(actions.to(self.device))
        return delayed_actions

    def _reset_action_delay(self, done_mask):
        if self.action_delay_buffer is None or not done_mask.any():
            return
        env_ids = torch.nonzero(done_mask, as_tuple=False).flatten().to(self.device)
        self.action_delay_buffer.reset(env_ids)
        self._sample_action_delay_lag(env_ids)

    @torch.no_grad()
    def _update_adaboot(self, completed_returns):
        if completed_returns.numel() < 2:
            return

        reward_mean = completed_returns.mean()
        reward_std = completed_returns.std(unbiased=False)
        reward_cv = reward_std / reward_mean.abs().clamp_min(1e-6)

        stability = 1.0 - torch.tanh(reward_cv)
        probability_target = self.adaboot_target_probability + 0.20 * (stability - self.adaboot_target_probability)
        probability_target = probability_target.clamp(0.45, 0.75).item()

        self.adaboot_reward_cv = reward_cv.item()
        self.adaboot_probability = (
            (1.0 - self.adaboot_ema_alpha) * self.adaboot_probability
            + self.adaboot_ema_alpha * probability_target
        )

    def _effective_adaboot_probability(self) -> float:
        if self.current_learning_iteration < self.adaboot_warmup_iterations:
            return 0.0

        ramp_progress = (
            self.current_learning_iteration - self.adaboot_warmup_iterations
        ) / max(float(self.adaboot_ramp_iterations), 1.0)
        ramp_progress = min(max(ramp_progress, 0.0), 1.0)
        return self.adaboot_probability * ramp_progress

    def _add_cenet_features(self, obs, use_adaboot=False):
        """Add CENet features, optionally applying AdaBoot."""
        obs_history = self.base_env.obs_history.to(self.device)
        obs_history = torch.nan_to_num(
            obs_history,
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
        )
        obs_history = self._normalize_history(obs_history)
        cenet_out = self.cenet(obs_history)

        estimated_velocity = torch.nan_to_num(
            cenet_out["v_hat"],
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
        )

        if use_adaboot:
            adaboot_probability = self._effective_adaboot_probability()
            ground_truth_velocity = (
                self.base_env.scene["robot"].data.root_lin_vel_b.to(self.device)
            )
            ground_truth_velocity = torch.nan_to_num(
                ground_truth_velocity,
                nan=0.0,
                posinf=1.0e6,
                neginf=-1.0e6,
            )
            bootstrap_mask = torch.rand(
                estimated_velocity.shape[0],
                1,
                device=self.device,
            ) < adaboot_probability
            actor_velocity = torch.where(
                bootstrap_mask,
                estimated_velocity,
                ground_truth_velocity,
            )
        else:
            actor_velocity = estimated_velocity

        cenet_features = torch.cat(
            (
                actor_velocity,
                torch.nan_to_num(
                    # Use the deterministic context for the policy. The VAE
                    # sample is still used when training CENet below.
                    cenet_out["mu"],
                    nan=0.0,
                    posinf=1.0e6,
                    neginf=-1.0e6,
                ),
            ),
            dim=-1,
        )

        if obs["cenet"].shape != cenet_features.shape:
            raise ValueError(
                "CENet observation shape mismatch: "
                f"placeholder={tuple(obs['cenet'].shape)}, "
                f"features={tuple(cenet_features.shape)}"
            )

        obs["cenet"] = cenet_features
        return self._sanitize_observations(obs)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Learn with CENet update after each PPO update."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )
            self.adaboot_episode_valid.zero_()

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.cenet.train()

        with torch.inference_mode():
            obs = self._add_cenet_features(obs, use_adaboot=True)
            obs = self._sanitize_observations(obs)

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
                obs = self._add_cenet_features(obs, use_adaboot=True)
                obs = self._sanitize_observations(obs)

            obs_history_buf = []
            next_obs_buf = []
            base_lin_vel_buf = []
            valid_transition_buf = []
            completed_episode_returns = []
            num_completed_episodes = 0

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    obs_history_buf.append(self.base_env.obs_history.clone())
                    base_lin_vel_buf.append(
                        self.base_env.scene["robot"].data.root_lin_vel_b.clone()
                    )

                    self._sanitize_policy_std()
                    self._sanitize_optimizer_state()
                    obs = self._sanitize_observations(obs)
                    actions = self.alg.act(obs)
                    actions = self._apply_action_delay(actions)
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )

                    step_rewards = rewards.to(self.device).reshape(-1)
                    done_mask = dones.to(self.device).bool().reshape(-1)
                    self._reset_action_delay(done_mask)
                    self.adaboot_episode_returns += step_rewards

                    valid_done_mask = done_mask & self.adaboot_episode_valid
                    if valid_done_mask.any():
                        completed_episode_returns.append(
                            self.adaboot_episode_returns[
                                valid_done_mask
                            ].clone()
                        )

                    self.adaboot_episode_returns[done_mask] = 0.0
                    self.adaboot_episode_valid[done_mask] = True

                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)

                    next_obs_buf.append(obs["policy"].clone())
                    valid_transition_buf.append(~dones.bool())
                    obs = self._add_cenet_features(
                        obs,
                        use_adaboot=True,
                    )
                    obs = self._sanitize_observations(obs)

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

                if completed_episode_returns:
                    completed_returns = torch.cat(completed_episode_returns)
                    num_completed_episodes = completed_returns.numel()
                    self._update_adaboot(completed_returns)

                stop = time.time()
                collect_time = stop - start
                start = stop
                obs = self._sanitize_observations(obs)
                self.alg.compute_returns(obs)

            self._sanitize_policy_std()
            self._sanitize_optimizer_state()
            loss_dict = self.alg.update()
            self._sanitize_policy_std()
            self._sanitize_optimizer_state()

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

                for _ in range(self.cenet_num_epochs):
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
                            beta=0.1,
                            velocity_weight=6.0,
                            reconstruction_weight=0.25,
                        )

                        self.cenet_optimizer.zero_grad(set_to_none=True)
                        ce_loss["total_loss"].backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.cenet.parameters(),
                            max_norm=1.0,
                        )
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

                loss_normalizer = num_valid_samples * self.cenet_num_epochs
                loss_dict["CENet/total"] = (
                    loss_sums["total"] / loss_normalizer
                )
                loss_dict["CENet/velocity"] = (
                    loss_sums["velocity"] / loss_normalizer
                )
                loss_dict["CENet/reconstruction"] = (
                    loss_sums["reconstruction"] / loss_normalizer
                )
                loss_dict["CENet/kl"] = (
                    loss_sums["kl"] / loss_normalizer
                )

            else:
                loss_dict["CENet/total"] = 0.0
                loss_dict["CENet/velocity"] = 0.0
                loss_dict["CENet/reconstruction"] = 0.0
                loss_dict["CENet/kl"] = 0.0

            loss_dict["CENet/valid_samples"] = num_valid_samples
            loss_dict["AdaBoot/probability"] = self._effective_adaboot_probability()
            loss_dict["AdaBoot/internal_probability"] = self.adaboot_probability
            loss_dict["AdaBoot/reward_cv"] = self.adaboot_reward_cv
            loss_dict["AdaBoot/completed_episodes"] = num_completed_episodes

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
        saved_dict["adaboot_probability"] = self.adaboot_probability
        saved_dict["adaboot_reward_cv"] = self.adaboot_reward_cv
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
        self.adaboot_probability = loaded_dict.get(
            "adaboot_probability",
            0.0,
        )
        self.adaboot_reward_cv = loaded_dict.get(
            "adaboot_reward_cv",
            0.0,
        )

        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
